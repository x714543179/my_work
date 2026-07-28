"""Actor and critic observations for Go1 Omni-Jump."""

from __future__ import annotations

import torch


def base_angular_velocity(env):
    return env.base_ang_vel * env.obs_scales.ang_vel


def projected_gravity(env):
    return env.projected_gravity


def velocity_commands(env):
    return env.commands[:, :3] * env.commands_scale


def jump_height_command(env):
    return env.commands[:, 3:4] * env.cfg.commands.height_observation_scale


def joint_position(env):
    return (env.dof_pos - env.default_dof_pos) * env.obs_scales.dof_pos


def joint_velocity(env):
    return env.dof_vel * env.obs_scales.dof_vel


def previous_action(env):
    return env.actions


def policy_observation(env):
    return env.obs_buf


def estimator_target(env):
    """Ten-dimensional state supervised by the history encoder.

    OmniNet estimates ``[z, x, y]``, four foot heights, and body-frame linear
    velocity. Positions are relative to the environment origin so parallel
    environment tiling does not leak into the regression target.
    """
    root_position = env.root_states[:, :3] - env.env_origins
    root_zxy = root_position[:, [2, 0, 1]]
    foot_heights = env.foot_positions[:, :, 2] - env.env_origins[:, 2:3]
    body_linear_velocity = env.base_lin_vel * env.obs_scales.lin_vel
    return torch.cat((root_zxy, foot_heights, body_linear_velocity), dim=-1)


def environment_context(env):
    """Paper-defined 187-point egocentric height map for the critic."""
    if torch.is_tensor(env.measured_heights):
        measured_heights = env.measured_heights
    else:
        measured_heights = torch.zeros(
            env.num_envs, env.num_height_points, device=env.device, dtype=env.root_states.dtype
        )
    terrain_heights = torch.clip(
        env.root_states[:, 2:3] - env.cfg.rewards.stance_height - measured_heights,
        -1.0,
        1.0,
    ) * env.obs_scales.height_measurements
    return terrain_heights
