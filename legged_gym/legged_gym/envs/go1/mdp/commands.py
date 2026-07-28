"""Command generators for omnidirectional jumping."""

from __future__ import annotations

import torch
from isaacgym.torch_utils import torch_rand_float


def resample_omni_jump_commands(env, env_ids):
    """Sample one jump target while exposing the stance/off command."""
    if len(env_ids) == 0:
        return

    count = len(env_ids)
    env.jump_target_commands[env_ids, 0] = torch_rand_float(
        *env.command_ranges["lin_vel_x"], (count, 1), device=env.device
    ).squeeze(1)
    env.jump_target_commands[env_ids, 1] = torch_rand_float(
        *env.command_ranges["lin_vel_y"], (count, 1), device=env.device
    ).squeeze(1)
    env.jump_target_commands[env_ids, 2] = torch_rand_float(
        *env.command_ranges["ang_vel_yaw"], (count, 1), device=env.device
    ).squeeze(1)

    env.jump_target_commands[env_ids, 3] = torch_rand_float(
        *env.command_ranges["height_z"], (count, 1), device=env.device
    ).squeeze(1)

    planar_speed = torch.norm(env.jump_target_commands[env_ids, :2], dim=1)
    env.jump_target_commands[env_ids, :2] *= (
        planar_speed > env.cfg.commands.small_command_threshold
    ).unsqueeze(1)

    env.commands[env_ids, :3] = 0.0
    env.commands[env_ids, 3] = env.cfg.rewards.stance_height
