"""Rollout-aligned future-state supervision for an attention actor."""

from __future__ import annotations

import torch
from .base import PPOPlugin


def build_future_targets(states, last_state, dones, future_steps):
    """s_t -> s_(t+1:t+H), excluding resets and unavailable rollout tails.

    states: [T,N,D], last_state: [N,D], dones[t]: transition t resets env.
    A timeout is also a boundary. No terminal/reset state is used as a label.
    """
    if states.ndim != 3 or tuple(last_state.shape) != tuple(states.shape[1:]):
        raise ValueError("Expected states [T,N,D] and last_state [N,D]")
    t, n, d = states.shape
    if future_steps < 1 or future_steps > t:
        raise ValueError("future_steps must be between 1 and the rollout length")
    done = dones.reshape(t, n).bool()
    sequence = torch.cat((states.detach(), last_state.detach().unsqueeze(0)), dim=0)
    targets = states.new_zeros(t, n, future_steps, d)
    valid = torch.zeros(t, n, future_steps, dtype=torch.bool, device=states.device)
    for horizon in range(1, future_steps + 1):
        count = t + 1 - horizon
        mask = torch.ones(count, n, dtype=torch.bool, device=states.device)
        for offset in range(horizon):
            mask &= ~done[offset:offset + count]
        valid[:count, :, horizon - 1] = mask
        targets[:count, :, horizon - 1] = torch.where(
            mask.unsqueeze(-1), sequence[horizon:horizon + count], 0.0
        )
    return targets, valid


class FuturePredictionPlugin(PPOPlugin):
    TARGET_KEY = "_future_prediction_targets"
    MASK_KEY = "_future_prediction_mask"

    def __init__(self, state_group="future_state", state_layout="v1",
                 loss_coef=1.0, negative_advantage_weight=2.0):
        if loss_coef < 0 or negative_advantage_weight < 1:
            raise ValueError("Prediction coefficient must be nonnegative and negative-advantage weight >= 1")
        self.state_group = state_group
        self.state_layout = state_layout
        self.loss_coef = float(loss_coef)
        self.negative_advantage_weight = float(negative_advantage_weight)
        self._has_last_state = False

    def on_init(self, ppo, env):
        model = ppo.actor.backbone
        if not hasattr(model, "predict_future") or model.is_recurrent:
            raise ValueError("FuturePredictionPlugin requires a non-recurrent future-predicting backbone")
        if ppo.critic.is_recurrent:
            raise ValueError("FuturePredictionPlugin currently supports feedforward critics")
        if self.state_group in model.obs_groups:
            raise ValueError("Clean supervision state must not be an actor input")
        self.future_steps = model.future_steps
        state = ppo.storage.observations[self.state_group]
        if state.shape[-1] != model.future_state_dim:
            raise ValueError("Clean state width differs from the prediction head")
        if self.future_steps > ppo.storage.num_transitions_per_env:
            raise ValueError("Prediction horizon exceeds rollout length")
        self._last_state = torch.zeros_like(state[0])
        self._metadata = {"version": 1, "state_group": self.state_group,
                          "state_layout": self.state_layout, "state_dim": model.future_state_dim,
                          "future_steps": self.future_steps,
                          "history_length": model.core.history_length,
                          "frame_dim": model.core.frame_dim,
                          "token_dim": model.core.token_dim,
                          "num_heads": model.core.blocks[0].attention.num_heads,
                          "num_layers": len(model.core.blocks),
                          "history_order": "oldest_to_newest"}

    def on_after_step(self, runner, obs, rewards, dones, extras):
        self._last_state.copy_(obs[self.state_group].detach())
        self._has_last_state = True
        return rewards

    def on_update_start(self, ppo):
        if not self._has_last_state or ppo.storage.step != ppo.storage.num_transitions_per_env:
            raise RuntimeError("Future supervision needs a full rollout and its final next observation")
        targets, valid = build_future_targets(
            ppo.storage.observations[self.state_group], self._last_state,
            ppo.storage.dones, self.future_steps,
        )
        # The standard generator shuffles these labels with exactly the same
        # indices as actor observations. They are removed before the next rollout.
        ppo.storage.observations.set(self.TARGET_KEY, targets)
        ppo.storage.observations.set(self.MASK_KEY, valid)
        self._valid_fraction = valid.float().mean(dim=(0, 1))
        self._squared_error = torch.zeros(self.future_steps, device=ppo.device)
        self._valid_count = torch.zeros_like(self._squared_error)

    def on_per_batch_extra_loss(self, ppo, batch):
        # PPO supplies the original observation forward, including its graph.
        # Sharing it avoids a second history encoder pass and backward graph.
        # It remains valid even if another plugin runs the mirrored actor first.
        actor_extra = getattr(batch, "actor_extra", None)
        if actor_extra is None:
            # Keep standalone hook callers compatible with the plugin API.
            prediction = ppo.actor.backbone.predict_future(batch.observations)
        else:
            prediction = actor_extra["predicted_future"]
        target = batch.observations[self.TARGET_KEY].detach()
        valid = batch.observations[self.MASK_KEY].bool()
        difference = torch.where(valid.unsqueeze(-1), prediction - target, 0.0)
        error = difference.square().mean(dim=-1)
        weight = torch.where(batch.advantages.detach().reshape(-1, 1) < 0,
                             self.negative_advantage_weight, 1.0)
        loss = (error * weight).sum() / valid.sum().clamp_min(1)
        self._squared_error += error.detach().sum(dim=0)
        self._valid_count += valid.sum(dim=0)
        return {"future_prediction": self.loss_coef * loss}

    def on_post_update(self, ppo):
        metrics = {}
        for k in range(self.future_steps):
            metrics[f"future_mse_step_{k + 1}"] = float(
                self._squared_error[k] / self._valid_count[k].clamp_min(1))
            metrics[f"future_valid_step_{k + 1}"] = float(self._valid_fraction[k])
        ppo.storage.observations.pop(self.TARGET_KEY)
        ppo.storage.observations.pop(self.MASK_KEY)
        self._has_last_state = False
        return metrics

    def on_save(self, ppo, saved_dict):
        saved_dict["future_prediction_metadata"] = self._metadata.copy()

    def on_load(self, ppo, loaded_dict):
        if loaded_dict.get("future_prediction_metadata") != self._metadata:
            raise ValueError("Checkpoint future-state layout/horizon differs from this task")
        self._has_last_state = False
