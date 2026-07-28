"""Domain-randomization and periodic event terms for Go1 Omni-Jump."""

from __future__ import annotations

import numpy as np
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import torch_rand_float


def randomize_friction(env, props=None, env_id=None, env_ids=None, friction_range=(0.2, 1.25)):
    if props is None or env_id is None:
        return props
    if env_id == 0:
        bucket_ids = torch.randint(0, 64, (env.num_envs, 1))
        buckets = torch_rand_float(*friction_range, (64, 1), device="cpu")
        env.friction_coeffs = buckets[bucket_ids]
    friction = float(env.friction_coeffs[env_id].item())
    for shape_prop in props:
        shape_prop.friction = friction
    return props


def randomize_rigid_body_properties(
    env,
    props=None,
    env_id=None,
    env_ids=None,
    added_base_mass_range=(-2.0, 2.0),
    base_com_range=(-0.05, 0.05),
):
    if props is None or env_id is None:
        return props
    props[0].mass += np.random.uniform(*added_base_mass_range)
    props[0].com.x += np.random.uniform(*base_com_range)
    props[0].com.y += np.random.uniform(*base_com_range)
    return props


def initialize_dof_properties(env, props=None, env_id=None, env_ids=None):
    if props is None or env_id is None:
        return props
    if env_id == 0:
        env.dof_pos_limits = torch.zeros(env.num_dof, 2, device=env.device)
        env.dof_vel_limits = torch.zeros(env.num_dof, device=env.device)
        env.torque_limits = torch.zeros(env.num_dof, device=env.device)
        for index in range(len(props)):
            env.dof_pos_limits[index, 0] = props["lower"][index].item()
            env.dof_pos_limits[index, 1] = props["upper"][index].item()
            env.dof_vel_limits[index] = props["velocity"][index].item()
            env.torque_limits[index] = props["effort"][index].item()
            midpoint = torch.mean(env.dof_pos_limits[index])
            dof_range = env.dof_pos_limits[index, 1] - env.dof_pos_limits[index, 0]
            env.dof_pos_limits[index, 0] = midpoint - 0.5 * dof_range * env.cfg.rewards.soft_dof_pos_limit
            env.dof_pos_limits[index, 1] = midpoint + 0.5 * dof_range * env.cfg.rewards.soft_dof_pos_limit
    props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    return props


def randomize_motor_strength(
    env,
    env_ids=None,
    env_id=None,
    strength_range=(0.9, 1.1),
):
    if env_id is not None:
        env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
    if env_ids is None or len(env_ids) == 0:
        return
    strength = torch_rand_float(*strength_range, (len(env_ids), env.num_actions), device=env.device)
    env.p_gains_multiplier[env_ids] = strength
    env.d_gains_multiplier[env_ids] = strength


def randomize_system_delay(env, env_ids=None, delay_range_ms=(0.0, 4.0)):
    """Approximate the paper's sub-step system delay by first-step interpolation."""
    if env_ids is None or len(env_ids) == 0:
        return
    sim_step_ms = env.cfg.sim.dt * 1000.0
    delay_ms = torch_rand_float(*delay_range_ms, (len(env_ids), 1), device=env.device).squeeze(1)
    env.action_delay_fraction[env_ids] = torch.clamp(delay_ms / sim_step_ms, 0.0, 1.0)


def update_height_measurements(env, env_ids=None):
    if env.cfg.terrain.measure_heights:
        env.measured_heights = env._get_heights()
