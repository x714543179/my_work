"""Domain randomization, latency, and jump-assistance events."""

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float

from legged_gym.envs.go2w.mdp.events import (
    randomize_friction,
    randomize_motor_zero_offset,
    randomize_pd_gains,
)


def randomize_rigid_body_properties(
    env,
    props=None,
    env_id=None,
    env_ids=None,
    added_base_mass_range=(-1.0, 1.0),
    link_mass_scale_range=(0.9, 1.1),
    base_com_range=(-0.03, 0.03),
    base_body_name="base",
):
    if props is None or env_id is None:
        return props

    base_body_index = env.gym.find_asset_rigid_body_index(
        env.robot_asset, base_body_name
    )
    if base_body_index < 0:
        raise RuntimeError(
            f"Asset does not contain the rigid body {base_body_name!r}."
        )

    props[base_body_index].mass += np.random.uniform(*added_base_mass_range)
    link_body_indices = [
        body_id for body_id in range(len(props)) if body_id != base_body_index
    ]
    link_scales = torch_rand_float(
        link_mass_scale_range[0],
        link_mass_scale_range[1],
        (1, len(link_body_indices)),
        device="cpu",
    )
    for scale_index, body_id in enumerate(link_body_indices):
        props[body_id].mass *= link_scales[0, scale_index].item()

    base_com = props[base_body_index].com
    props[base_body_index].com = gymapi.Vec3(
        base_com.x + np.random.uniform(*base_com_range),
        base_com.y + np.random.uniform(*base_com_range),
        base_com.z + np.random.uniform(*base_com_range),
    )
    return props


def init_dof_properties(env, props=None, env_id=None, env_ids=None):
    if props is None or env_id is None:
        return props
    if env_id != 0:
        return props

    env.dof_pos_limits = torch.zeros(
        env.num_dof, 2, dtype=torch.float, device=env.device
    )
    env.dof_vel_limits = torch.zeros(
        env.num_dof, dtype=torch.float, device=env.device
    )
    env.torque_limits = torch.zeros(
        env.num_dof, dtype=torch.float, device=env.device
    )
    for dof_id in range(len(props)):
        lower = props["lower"][dof_id].item()
        upper = props["upper"][dof_id].item()
        midpoint = 0.5 * (lower + upper)
        dof_range = upper - lower
        env.dof_pos_limits[dof_id, 0] = (
            midpoint - 0.5 * dof_range * env.cfg.rewards.soft_dof_pos_limit
        )
        env.dof_pos_limits[dof_id, 1] = (
            midpoint + 0.5 * dof_range * env.cfg.rewards.soft_dof_pos_limit
        )
        env.dof_vel_limits[dof_id] = props["velocity"][dof_id].item()
        env.torque_limits[dof_id] = props["effort"][dof_id].item()
    return props


def _observation_term_params(env, term_name):
    term = getattr(env.cfg.observations.actor, term_name)
    return dict(term.params)


def update_observation_latency(env, env_ids=None):
    motor_params = _observation_term_params(env, "motor")
    if motor_params.get("latency_enabled", False):
        motor_latency_steps = int(motor_params["latency_range"][1])
        dof_error = env.dof_pos - env.default_dof_pos
        dof_error = dof_error.clone()
        dof_error[:, env.wheel_indices] = 0.0
        q = dof_error * env.obs_scales.dof_pos
        dq = env.dof_vel * env.obs_scales.dof_vel
        env.obs_motor_latency_buffer[:, :, 1:] = env.obs_motor_latency_buffer[
            :, :, :motor_latency_steps
        ].clone()
        env.obs_motor_latency_buffer[:, :, 0] = torch.cat((q, dq), dim=1)

    imu_params = _observation_term_params(env, "imu")
    if imu_params.get("latency_enabled", False):
        imu_latency_steps = int(imu_params["latency_range"][1])
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.base_quat[:] = env.root_states[:, 3:7]
        env.base_ang_vel[:] = quat_rotate_inverse(
            env.base_quat, env.root_states[:, 10:13]
        )
        env.base_euler_xyz = env.get_euler_xyz_tensor(env.base_quat)
        env.obs_imu_latency_buffer[:, :, 1:] = env.obs_imu_latency_buffer[
            :, :, :imu_latency_steps
        ].clone()
        env.obs_imu_latency_buffer[:, :, 0] = torch.cat(
            (
                env.base_ang_vel * env.obs_scales.ang_vel,
                env.base_euler_xyz * env.obs_scales.quat,
            ),
            dim=1,
        )


def reset_latency_buffers(
    env,
    env_ids=None,
):
    if env_ids is None or len(env_ids) == 0:
        return

    env.cmd_action_latency_buffer[env_ids] = 0.0
    env.obs_motor_latency_buffer[env_ids] = 0.0
    env.obs_imu_latency_buffer[env_ids] = 0.0

    action_params = dict(env.cfg.actions.command_latency.params)
    _sample_latency(
        env,
        env.cmd_action_latency_simstep,
        env_ids,
        action_params,
    )
    _sample_latency(
        env,
        env.obs_motor_latency_simstep,
        env_ids,
        _observation_term_params(env, "motor"),
    )
    _sample_latency(
        env,
        env.obs_imu_latency_simstep,
        env_ids,
        _observation_term_params(env, "imu"),
    )


def _sample_latency(env, latency_steps, env_ids, params):
    if not params.get("latency_enabled", params.get("enabled", False)):
        latency_steps[env_ids] = 0
        return
    latency_range = params.get("latency_range", (1, 3))
    if params.get("randomize_latency", params.get("randomize", False)):
        latency_steps[env_ids] = torch.randint(
            int(latency_range[0]),
            int(latency_range[1]) + 1,
            (len(env_ids),),
            device=env.device,
        )
    else:
        latency_steps[env_ids] = int(latency_range[1])


def assist_jump(env, candidates):
    if not env.cfg.jump_assist.enabled or not torch.any(candidates):
        return torch.zeros_like(candidates), 0

    probability_tenths = max(
        env.cfg.jump_assist.initial_probability_tenths
        - int(env.common_step_counter / env.cfg.jump_assist.decay_interval_steps),
        0,
    )
    random_draw = torch.randint(0, 10, (env.num_envs,), device=env.device)
    selected = candidates & (random_draw < probability_tenths)
    if torch.any(selected):
        velocity_delta = torch_rand_float(
            env.cfg.jump_assist.vertical_velocity_range[0],
            env.cfg.jump_assist.vertical_velocity_range[1],
            (env.num_envs, 1),
            device=env.device,
        ).squeeze(1)
        env.root_states[selected, 9] += velocity_delta[selected]
        env.gym.set_actor_root_state_tensor(
            env.sim, gymtorch.unwrap_tensor(env.root_states)
        )
    return selected, probability_tenths
