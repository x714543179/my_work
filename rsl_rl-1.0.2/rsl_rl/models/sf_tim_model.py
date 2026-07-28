from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import HiddenState, MLP
from rsl_rl.utils import unpad_trajectories


class SFTIMActorBackbone(nn.Module):
    """SF-TIM actor backbone with CENet and terrain-feature autoencoders.

    The policy input follows the paper's structure:
    proprioception ``o_t`` + CENet velocity/context ``[v_t, z_p]`` +
    terrain feature ``z_e``.  The rollout history buffer in this codebase stores
    the previous actor observations, so the CENet decoder is trained to predict
    the current proprioceptive observation from that history.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        actor_hidden_dims=(512, 256, 128),
        cenet_hidden_dims=(128, 64),
        cenet_decoder_hidden_dims=(64, 128),
        terrain_encoder_hidden_dims=(128, 64),
        terrain_decoder_hidden_dims=(64, 128),
        latent_dim=19,
        velocity_dim=3,
        terrain_latent_dim=16,
        terrain_group="actor_terrain",
        proprio_target_group="actor_proprio",
        proprio_groups=None,
        velocity_target_group="prev_critic_base_lin_vel",
        terrain_target_group="actor_terrain",
        activation="elu",
        aux_loss_coef=1.0,
        velocity_loss_coef=1.0,
        reconstruction_loss_coef=1.0,
        kl_loss_coef=1.0,
        terrain_loss_coef=1.0,
        kl_loss_reduction="sum",
        **_,
    ) -> None:
        super().__init__()
        self.obs_groups = obs_groups[obs_set]
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.velocity_dim = velocity_dim
        self.terrain_latent_dim = terrain_latent_dim
        self.terrain_group = terrain_group
        self.proprio_target_group = proprio_target_group
        self.proprio_groups = list(proprio_groups) if proprio_groups is not None else [proprio_target_group]
        self.velocity_target_group = velocity_target_group
        self.terrain_target_group = terrain_target_group
        self.aux_loss_coef = aux_loss_coef
        self.velocity_loss_coef = velocity_loss_coef
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.kl_loss_coef = kl_loss_coef
        self.terrain_loss_coef = terrain_loss_coef
        self.kl_loss_reduction = kl_loss_reduction

        self.actor_obs_dim = self._obs_dim(obs, self.obs_groups)
        self.proprio_dim = self._obs_dim(obs, self.proprio_groups)
        self.actor_frame_dim = obs["actor"].shape[-1]
        history_dim = obs["history"].shape[-1]
        if history_dim % self.actor_frame_dim != 0:
            raise ValueError(
                f"History dim {history_dim} is not divisible by actor obs dim {self.actor_frame_dim}."
            )
        self.history_len = history_dim // self.actor_frame_dim
        self.proprio_history_dim = self.history_len * self.proprio_dim
        self.terrain_dim = obs[terrain_group].shape[-1]

        proprio_latent_dim = latent_dim - velocity_dim
        if proprio_latent_dim <= 0:
            raise ValueError("latent_dim must be greater than velocity_dim.")
        self.proprio_latent_dim = proprio_latent_dim

        cenet_output_dim = cenet_hidden_dims[-1]
        terrain_output_dim = terrain_encoder_hidden_dims[-1]
        self.cenet_encoder = MLP(self.proprio_history_dim, cenet_output_dim, cenet_hidden_dims[:-1], activation)
        self.encode_mean_latent = nn.Linear(cenet_output_dim, proprio_latent_dim)
        self.encode_logvar_latent = nn.Linear(cenet_output_dim, proprio_latent_dim)
        self.encode_mean_vel = nn.Linear(cenet_output_dim, velocity_dim)
        self.encode_logvar_vel = nn.Linear(cenet_output_dim, velocity_dim)
        self.cenet_decoder = MLP(latent_dim, self.proprio_dim, cenet_decoder_hidden_dims, activation)

        self.terrain_encoder = MLP(self.terrain_dim, terrain_output_dim, terrain_encoder_hidden_dims[:-1], activation)
        self.terrain_latent = nn.Linear(terrain_output_dim, terrain_latent_dim)
        self.terrain_decoder = MLP(terrain_latent_dim, self.terrain_dim, terrain_decoder_hidden_dims, activation)

        actor_input_dim = self.actor_obs_dim + latent_dim + terrain_latent_dim
        self.actor = MLP(actor_input_dim, output_dim, actor_hidden_dims, activation)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        train_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        del hidden_state

        proprio_history = self._extract_proprio_history(obs["history"])
        cenet_code, cenet_aux, proprio_reconstruction = self.cenet_forward(proprio_history)
        terrain_code, terrain_reconstruction = self.terrain_forward(obs[self.terrain_group])
        actor_obs = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        actions = self.actor(torch.cat((actor_obs, cenet_code, terrain_code), dim=-1))

        output = {"actions": actions}
        if train_mode:
            output["aux_losses"] = self._auxiliary_losses(
                cenet_aux,
                proprio_reconstruction,
                terrain_reconstruction,
                obs,
            )
        return output

    def cenet_forward(self, proprio_history: torch.Tensor):
        distribution = self.cenet_encoder(proprio_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = torch.clamp(self.encode_logvar_latent(distribution), min=-5.0, max=5.0)
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = torch.clamp(self.encode_logvar_vel(distribution), min=-5.0, max=5.0)
        sample = self.training
        code_vel = self._reparameterize(mean_vel, logvar_vel, sample)
        code_latent = self._reparameterize(mean_latent, logvar_latent, sample)
        code = torch.cat((code_vel, code_latent), dim=-1)
        decode = self.cenet_decoder(code)
        aux = {
            "code_vel": code_vel,
            "mean_latent": mean_latent,
            "logvar_latent": logvar_latent,
        }
        return code, aux, decode

    def terrain_forward(self, terrain_obs: torch.Tensor):
        terrain_features = self.terrain_encoder(terrain_obs)
        terrain_code = self.terrain_latent(terrain_features)
        terrain_reconstruction = self.terrain_decoder(terrain_code)
        return terrain_code, terrain_reconstruction

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    def as_jit(self) -> nn.Module:
        return _SFTIMJitWrapper(self)

    def _extract_proprio_history(self, history: torch.Tensor) -> torch.Tensor:
        frames = history.reshape(*history.shape[:-1], self.history_len, self.actor_frame_dim)
        proprio = frames[..., : self.proprio_dim]
        return proprio.reshape(*history.shape[:-1], self.proprio_history_dim)

    def _auxiliary_losses(self, cenet_aux, proprio_reconstruction, terrain_reconstruction, obs):
        losses = {}
        velocity_target = obs[self.velocity_target_group][..., : self.velocity_dim].detach()
        proprio_target = torch.cat([obs[group] for group in self.proprio_groups], dim=-1).detach()
        terrain_target = obs[self.terrain_target_group].detach()

        losses["velocity_estimation"] = self.velocity_loss_coef * F.mse_loss(
            cenet_aux["code_vel"], velocity_target
        )
        losses["proprio_reconstruction"] = self.reconstruction_loss_coef * F.mse_loss(
            proprio_reconstruction, proprio_target
        )
        losses["terrain_reconstruction"] = self.terrain_loss_coef * F.mse_loss(
            terrain_reconstruction, terrain_target
        )

        mean_latent = cenet_aux["mean_latent"]
        logvar_latent = cenet_aux["logvar_latent"]
        kl_per_sample = torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp(), dim=-1)
        if self.kl_loss_reduction == "mean":
            kl_loss = -0.5 * torch.mean(kl_per_sample)
        elif self.kl_loss_reduction == "sum":
            kl_loss = -0.5 * torch.sum(kl_per_sample)
        else:
            raise ValueError(f"Unsupported kl_loss_reduction: {self.kl_loss_reduction}")
        losses["kl"] = self.kl_loss_coef * kl_loss
        return {name: self.aux_loss_coef * loss for name, loss in losses.items()}

    @staticmethod
    def _obs_dim(obs: TensorDict, groups: list[str]) -> int:
        return sum(obs[group].shape[-1] for group in groups)

    @staticmethod
    def _reparameterize(mean: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mean
        std = torch.exp(logvar * 0.5)
        return mean + std * torch.randn_like(std)


class _SFTIMJitWrapper(nn.Module):
    """TorchScript wrapper for deployment with measured elevation maps."""

    __constants__ = [
        "actor_frame_dim",
        "proprio_dim",
        "history_len",
        "proprio_history_dim",
    ]

    def __init__(self, backbone: SFTIMActorBackbone) -> None:
        super().__init__()
        self.actor = copy.deepcopy(backbone.actor)
        self.cenet_encoder = copy.deepcopy(backbone.cenet_encoder)
        self.encode_mean_latent = copy.deepcopy(backbone.encode_mean_latent)
        self.encode_mean_vel = copy.deepcopy(backbone.encode_mean_vel)
        self.terrain_encoder = copy.deepcopy(backbone.terrain_encoder)
        self.terrain_latent = copy.deepcopy(backbone.terrain_latent)
        self.actor_frame_dim = backbone.actor_frame_dim
        self.proprio_dim = backbone.proprio_dim
        self.history_len = backbone.history_len
        self.proprio_history_dim = backbone.proprio_history_dim

    def forward(
        self,
        observations: torch.Tensor,
        history_observations: torch.Tensor,
        terrain_observations: torch.Tensor,
    ) -> torch.Tensor:
        frames = history_observations.reshape(-1, self.history_len, self.actor_frame_dim)
        proprio_history = frames[:, :, : self.proprio_dim].reshape(-1, self.proprio_history_dim)
        distribution = self.cenet_encoder(proprio_history)
        cenet_code = torch.cat((self.encode_mean_vel(distribution), self.encode_mean_latent(distribution)), dim=-1)
        terrain_code = self.terrain_latent(self.terrain_encoder(terrain_observations))
        return self.actor(torch.cat((observations, cenet_code, terrain_code), dim=-1))
