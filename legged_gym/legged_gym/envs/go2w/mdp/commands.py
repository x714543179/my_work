"""Command terms for the go2w task."""

from __future__ import annotations

import torch
from isaacgym.torch_utils import torch_rand_float


def resample_commands(env, env_ids):
    if len(env_ids) == 0:
        return
    env.commands[env_ids, 0] = torch_rand_float(
        env.command_ranges["lin_vel_x"][0], env.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=env.device
    ).squeeze(1)
    env.commands[env_ids, 1] = torch_rand_float(
        env.command_ranges["lin_vel_y"][0], env.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=env.device
    ).squeeze(1)
    if env.cfg.commands.heading_command:
        env.commands[env_ids, 3] = torch_rand_float(
            env.command_ranges["heading"][0], env.command_ranges["heading"][1], (len(env_ids), 1), device=env.device
        ).squeeze(1)
    else:
        env.commands[env_ids, 2] = torch_rand_float(
            env.command_ranges["ang_vel_yaw"][0],
            env.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1),
            device=env.device,
        ).squeeze(1)
    env.commands[env_ids, :2] *= (torch.norm(env.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


def sf_tim_resample_commands(
    env,
    env_ids,
    walk_lin_vel_x=(-1.2, 1.2),
    walk_lin_vel_y=(-1.2, 1.2),
    walk_ang_vel_yaw=(-2.0, 2.0),
    jump_lin_vel_x=(0.3, 1.2),
    jump_terrain_types=(3, 4),
):
    """Sample SF-TIM commands with tau4/tau5 restricted to forward velocity."""
    if len(env_ids) == 0:
        return

    env.commands[env_ids, 0] = torch_rand_float(
        walk_lin_vel_x[0], walk_lin_vel_x[1], (len(env_ids), 1), device=env.device
    ).squeeze(1)
    env.commands[env_ids, 1] = torch_rand_float(
        walk_lin_vel_y[0], walk_lin_vel_y[1], (len(env_ids), 1), device=env.device
    ).squeeze(1)
    env.commands[env_ids, 2] = torch_rand_float(
        walk_ang_vel_yaw[0], walk_ang_vel_yaw[1], (len(env_ids), 1), device=env.device
    ).squeeze(1)

    if hasattr(env, "env_class"):
        terrain_type = env.env_class[env_ids].long()
        jump_mask = torch.zeros_like(terrain_type, dtype=torch.bool)
        for terrain_id in jump_terrain_types:
            jump_mask |= terrain_type == int(terrain_id)
        if torch.any(jump_mask):
            jump_env_ids = env_ids[jump_mask]
            env.commands[jump_env_ids, 0] = torch_rand_float(
                jump_lin_vel_x[0], jump_lin_vel_x[1], (len(jump_env_ids), 1), device=env.device
            ).squeeze(1)
            env.commands[jump_env_ids, 1] = 0.0
            env.commands[jump_env_ids, 2] = 0.0

    env.commands[env_ids, :2] *= (torch.norm(env.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


def mgdp_mixed_resample_commands(
    env,
    env_ids,
    easy_terrain_types=(0, 1, 2),
    first_stage_lin_vel_x=(-1.2, 1.2),
    first_stage_lin_vel_y=(-1.0, 1.0),
    first_stage_ang_vel_yaw=(-1.0, 1.0),
    first_stage_heading=(-3.14, 3.14),
    parkour_lin_vel_x=(0.0, 1.5),
):
    """Sample full velocity commands on rough replacements and vx-only commands on MGDP parkour."""
    if len(env_ids) == 0:
        return

    terrain_type = None
    if hasattr(env, "env_class"):
        terrain_type = env.env_class[env_ids].long()
    easy_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    if terrain_type is not None:
        for terrain_id in easy_terrain_types:
            easy_mask |= terrain_type == int(terrain_id)

    easy_ids = env_ids[easy_mask]
    parkour_ids = env_ids[~easy_mask]

    if len(easy_ids) > 0:
        env.commands[easy_ids, 0] = torch_rand_float(
            first_stage_lin_vel_x[0],
            first_stage_lin_vel_x[1],
            (len(easy_ids), 1),
            device=env.device,
        ).squeeze(1)
        env.commands[easy_ids, 1] = torch_rand_float(
            first_stage_lin_vel_y[0],
            first_stage_lin_vel_y[1],
            (len(easy_ids), 1),
            device=env.device,
        ).squeeze(1)
        if env.cfg.commands.heading_command:
            env.commands[easy_ids, 3] = torch_rand_float(
                first_stage_heading[0],
                first_stage_heading[1],
                (len(easy_ids), 1),
                device=env.device,
            ).squeeze(1)
        else:
            env.commands[easy_ids, 2] = torch_rand_float(
                first_stage_ang_vel_yaw[0],
                first_stage_ang_vel_yaw[1],
                (len(easy_ids), 1),
                device=env.device,
            ).squeeze(1)

    if len(parkour_ids) > 0:
        env.commands[parkour_ids, 0] = torch_rand_float(
            parkour_lin_vel_x[0],
            parkour_lin_vel_x[1],
            (len(parkour_ids), 1),
            device=env.device,
        ).squeeze(1)
        env.commands[parkour_ids, 1:] = 0.0

    env.commands[env_ids, :2] *= (torch.norm(env.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
