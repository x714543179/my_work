"""Observation terms for the Nezha spring jump task."""

import torch


def actor_commands(env):
    return env.commands * env.commands_scale


def actor_imu(
    env,
    latency_enabled=True,
    randomize_latency=True,
    latency_range=(1, 3),
):
    if latency_enabled:
        return env.obs_imu_latency_buffer[
            env._env_indices, :, env.obs_imu_latency_simstep.long()
        ]
    return torch.cat(
        (
            env.base_ang_vel * env.obs_scales.ang_vel,
            env.base_euler_xyz * env.obs_scales.quat,
        ),
        dim=1,
    )


def actor_motor(
    env,
    latency_enabled=True,
    randomize_latency=True,
    latency_range=(1, 3),
):
    dof_error = env.dof_pos - env.default_dof_pos
    dof_error = dof_error.clone()
    dof_error[:, env.wheel_indices] = 0.0
    if latency_enabled:
        return env.obs_motor_latency_buffer[
            env._env_indices, :, env.obs_motor_latency_simstep.long()
        ]
    return torch.cat(
        (
            dof_error * env.obs_scales.dof_pos,
            env.dof_vel * env.obs_scales.dof_vel,
        ),
        dim=1,
    )


def actor_actions(env):
    return env.actions


def critic_commands(env):
    return env.commands * env.commands_scale


def critic_dof_pos_error(env):
    dof_error = env.dof_pos - env.default_dof_pos
    dof_error = dof_error.clone()
    dof_error[:, env.wheel_indices] = 0.0
    return dof_error * env.obs_scales.dof_pos


def critic_dof_pos(env):
    dof_position = env.dof_pos.clone()
    dof_position[:, env.wheel_indices] = 0.0
    return dof_position * env.obs_scales.dof_pos


def critic_dof_vel(env):
    return env.dof_vel * env.obs_scales.dof_vel


def critic_actions(env):
    return env.actions


def critic_base_lin_vel(env):
    return env.base_lin_vel * env.obs_scales.lin_vel


def critic_base_ang_vel(env):
    return env.base_ang_vel * env.obs_scales.ang_vel


def critic_base_euler_xyz(env):
    return env.base_euler_xyz * env.obs_scales.quat


def critic_contact_mask(env):
    return (env.contact_forces[:, env.feet_indices, 2] > 5.0).float()


def critic_has_jumped(env):
    return env.has_jumped.unsqueeze(1).float()


def actor_imu_noise(env, term_value):
    noise = env.cfg.noise.noise_scales
    noise_level = env.cfg.noise.noise_level
    ang_vel = torch.full(
        (3,),
        noise.ang_vel * env.obs_scales.ang_vel * noise_level,
        device=env.device,
        dtype=term_value.dtype,
    )
    euler_xyz = torch.full(
        (3,),
        noise.quat * noise_level,
        device=env.device,
        dtype=term_value.dtype,
    )
    return torch.cat((ang_vel, euler_xyz))


def actor_motor_noise(env, term_value):
    noise = env.cfg.noise.noise_scales
    noise_level = env.cfg.noise.noise_level
    dof_pos = torch.full(
        (env.num_actions,),
        noise.dof_pos * env.obs_scales.dof_pos * noise_level,
        device=env.device,
        dtype=term_value.dtype,
    )
    dof_vel = torch.full(
        (env.num_actions,),
        noise.dof_vel * env.obs_scales.dof_vel * noise_level,
        device=env.device,
        dtype=term_value.dtype,
    )
    return torch.cat((dof_pos, dof_vel))
