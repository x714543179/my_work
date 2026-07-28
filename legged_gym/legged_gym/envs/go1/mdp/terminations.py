"""Termination terms for Go1 Omni-Jump."""

from __future__ import annotations

import torch


def illegal_contact(env, force_threshold=1.0):
    if len(env.termination_contact_indices) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.any(
        torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1)
        > force_threshold,
        dim=1,
    )


def excessive_roll(env, roll_limit=2.4):
    return torch.abs(env.base_euler_xyz[:, 0]) > roll_limit


def fallen_below_terrain(env, clearance=-0.5):
    if torch.is_tensor(env.measured_heights):
        ground_height = torch.mean(env.measured_heights, dim=1)
    else:
        ground_height = torch.zeros(env.num_envs, device=env.device, dtype=env.root_states.dtype)
    return env.root_states[:, 2] - ground_height < clearance


def jump_cycle_complete(env):
    landing_finished = env.jump_completed & (env.landing_phase_steps <= 0)
    return env.jump_failed | landing_finished
