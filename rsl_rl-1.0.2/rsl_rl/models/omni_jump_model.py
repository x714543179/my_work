"""History-conditioned actor used by the manager-based Omni-Jump task."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import HiddenState, MLP
from rsl_rl.utils import unpad_trajectories


class OmniJumpActorBackbone(nn.Module):
    """Estimate jump state from history and condition the policy on it."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        actor_hidden_dims=(512, 256, 128),
        estimator_hidden_dims=(258, 128),
        state_dim=10,
        state_target_group="critic_state_target",
        activation="elu",
        estimator_loss_coef=1.0,
        **_,
    ) -> None:
        super().__init__()
        if state_dim != 10:
            raise ValueError(f"OmniJumpActorBackbone expects a 10-D state target, got {state_dim}.")
        if len(estimator_hidden_dims) != 2:
            raise ValueError("estimator_hidden_dims must contain the two hidden widths used by GenHis.")

        self.obs_groups = obs_groups[obs_set]
        self.state_dim = state_dim
        self.state_target_group = state_target_group
        self.estimator_loss_coef = estimator_loss_coef
        actor_obs_dim = sum(obs[group].shape[-1] for group in self.obs_groups)
        history_dim = obs["history"].shape[-1]

        self.estimator = nn.Sequential(
            nn.Linear(history_dim, estimator_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(estimator_hidden_dims[0], estimator_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(estimator_hidden_dims[1], state_dim),
        )
        self.actor = MLP(actor_obs_dim + state_dim, output_dim, actor_hidden_dims, activation)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        state_estimate = self.estimator(obs["history"])
        actor_observation = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        output = {"actions": self.actor(torch.cat((state_estimate, actor_observation), dim=-1))}
        if train_mode:
            output["aux_losses"] = self._estimator_losses(state_estimate, obs[self.state_target_group])
        return output

    def _estimator_losses(self, estimate, target):
        target = target.detach()
        coefficient = self.estimator_loss_coef
        return {
            "root_position_estimation": coefficient * F.mse_loss(estimate[..., 0:3], target[..., 0:3]),
            "foot_height_estimation": coefficient * F.mse_loss(estimate[..., 3:7], target[..., 3:7]),
            "linear_velocity_estimation": coefficient * F.mse_loss(estimate[..., 7:10], target[..., 7:10]),
        }

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    def as_jit(self) -> nn.Module:
        return _OmniJumpJitWrapper(self)


class _OmniJumpJitWrapper(nn.Module):
    def __init__(self, backbone: OmniJumpActorBackbone) -> None:
        super().__init__()
        self.estimator = copy.deepcopy(backbone.estimator)
        self.actor = copy.deepcopy(backbone.actor)

    def forward(self, observations: torch.Tensor, history_observations: torch.Tensor) -> torch.Tensor:
        state_estimate = self.estimator(history_observations)
        return self.actor(torch.cat((state_estimate, observations), dim=-1))
