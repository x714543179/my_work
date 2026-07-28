"""Curriculum terms for the go2w task."""

from __future__ import annotations

import numpy as np
import torch


def terrain_levels(env, env_ids):
    if not env.init_done:
        return
    distance = torch.norm(env.root_states[env_ids, :2] - env.env_origins[env_ids, :2], dim=1)
    move_up = distance > env.terrain.env_length / 2
    move_down = (
        distance < torch.norm(env.commands[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    ) * ~move_up
    if hasattr(env, "terrain_importer"):
        env.terrain_importer.update_env_origins(env, env_ids, move_up, move_down)
        return
    env.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
    env.terrain_levels[env_ids] = torch.where(
        env.terrain_levels[env_ids] >= env.max_terrain_level,
        torch.randint_like(env.terrain_levels[env_ids], env.max_terrain_level),
        torch.clip(env.terrain_levels[env_ids], 0),
    )
    env.env_origins[env_ids] = env.terrain_origins[env.terrain_levels[env_ids], env.terrain_types[env_ids]]


def sf_tim_terrain_levels(
    env,
    env_ids,
    jump_terrain_types=(3, 4),
    success_distance_ratio=0.6,
    walk_success_distance_ratio=0.5,
    failure_distance_ratio=None,
    log_metrics=True,
):
    """SF-TIM terrain curriculum.

    For gap/platform terrains the paper advances difficulty after the robot
    travels forward more than 0.6 of the terrain length.  Other terrains keep
    the standard game-inspired distance rule.
    """
    if not env.init_done:
        return

    distance = torch.norm(env.root_states[env_ids, :2] - env.env_origins[env_ids, :2], dim=1)
    level_before = env.terrain_levels[env_ids].float().clone()
    move_up = distance > walk_success_distance_ratio * env.terrain.env_length
    jump_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    if hasattr(env, "env_class"):
        terrain_type = env.env_class[env_ids].long()
        for terrain_id in jump_terrain_types:
            jump_mask |= terrain_type == int(terrain_id)
        forward_distance = env.root_states[env_ids, 0] - env.env_origins[env_ids, 0]
        jump_move_up = forward_distance > success_distance_ratio * env.terrain.env_length
        move_up = torch.where(jump_mask, jump_move_up, move_up)

    expected_distance = torch.norm(env.commands[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    if failure_distance_ratio is not None:
        failure_distance = torch.full_like(distance, failure_distance_ratio * env.terrain.env_length)
        expected_distance = torch.minimum(expected_distance, failure_distance)
    move_down = (distance < expected_distance) * ~move_up
    if hasattr(env, "terrain_importer"):
        env.terrain_importer.update_env_origins(env, env_ids, move_up, move_down)
        if log_metrics:
            _log_curriculum_metrics(
                env,
                env_ids,
                move_up,
                move_down,
                distance,
                expected_distance,
                level_before,
                jump_mask,
            )
        return

    env.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
    env.terrain_levels[env_ids] = torch.where(
        env.terrain_levels[env_ids] >= env.max_terrain_level,
        torch.randint_like(env.terrain_levels[env_ids], env.max_terrain_level),
        torch.clip(env.terrain_levels[env_ids], 0),
    )
    env.env_origins[env_ids] = env.terrain_origins[env.terrain_levels[env_ids], env.terrain_types[env_ids]]
    if log_metrics:
        _log_curriculum_metrics(
            env,
            env_ids,
            move_up,
            move_down,
            distance,
            expected_distance,
            level_before,
            jump_mask,
        )


def mgdp_parkour_terrain_levels(
    env,
    env_ids,
    success_distance_ratio=0.5,
    log_metrics=True,
):
    """MGDP-style parkour curriculum with distance measured from the configured start offset."""
    if not env.init_done:
        return

    init_offset = torch.tensor(
        env.cfg.init_state.pos[:2],
        dtype=env.root_states.dtype,
        device=env.device,
    )
    relative_distance = env.root_states[env_ids, :2] - env.env_origins[env_ids, :2] - init_offset
    distance = torch.norm(relative_distance, dim=1)
    level_before = env.terrain_levels[env_ids].float().clone()
    move_up = distance > success_distance_ratio * env.terrain.env_length
    expected_distance = torch.norm(env.commands[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down = (distance < expected_distance) * ~move_up

    if hasattr(env, "terrain_importer"):
        env.terrain_importer.update_env_origins(env, env_ids, move_up, move_down)
    else:
        env.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        env.terrain_levels[env_ids] = torch.where(
            env.terrain_levels[env_ids] >= env.max_terrain_level,
            torch.randint_like(env.terrain_levels[env_ids], env.max_terrain_level),
            torch.clip(env.terrain_levels[env_ids], 0),
        )
        env.env_origins[env_ids] = env.terrain_origins[env.terrain_levels[env_ids], env.terrain_types[env_ids]]

    if log_metrics:
        jump_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        _log_curriculum_metrics(
            env,
            env_ids,
            move_up,
            move_down,
            distance,
            expected_distance,
            level_before,
            jump_mask,
        )


def _log_curriculum_metrics(
    env,
    env_ids,
    move_up,
    move_down,
    distance,
    expected_distance,
    level_before,
    jump_mask,
):
    step_metrics = env.extras.setdefault("step_metrics", {})
    level_after = env.terrain_levels[env_ids].float()
    values = {
        "curriculum/move_up": move_up.float(),
        "curriculum/move_down": move_down.float(),
        "curriculum/distance": distance,
        "curriculum/failure_distance": expected_distance,
        "curriculum/level_before": level_before,
        "curriculum/level_after": level_after,
        "curriculum/jump_terrain": jump_mask.float(),
    }
    if hasattr(env, "time_out_buf"):
        values["curriculum/time_out"] = env.time_out_buf[env_ids].float()
    for name, value in values.items():
        metric = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        metric[env_ids] = value.float()
        step_metrics[name] = metric


def command_ranges(env, env_ids):
    if torch.mean(env.episode_sums["tracking_lin_vel"][env_ids]) / env.max_episode_length > (
        0.8 * env.reward_scales["tracking_lin_vel"]
    ):
        env.command_ranges["lin_vel_x"][0] = np.clip(
            env.command_ranges["lin_vel_x"][0] - 0.5, -env.cfg.commands.max_curriculum, 0.0
        )
        env.command_ranges["lin_vel_x"][1] = np.clip(
            env.command_ranges["lin_vel_x"][1] + 0.5, 0.0, env.cfg.commands.max_curriculum
        )
