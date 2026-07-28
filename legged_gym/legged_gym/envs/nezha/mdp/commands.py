"""Landing-target and approach-velocity commands for Nezha jumps."""

import math

import torch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.utils.math import wrap_to_pi


TARGET_X_INDEX = 0
TARGET_Y_INDEX = 1
APPROACH_VX_INDEX = 2
APPROACH_YAW_RATE_INDEX = 3
JUMP_SIGNAL_INDEX = 4


def _sample_target_angles(
    env,
    count,
    lateral_target_probability,
    forward_angle_range,
    lateral_angle_jitter_range,
):
    forward_angles = torch_rand_float(
        forward_angle_range[0],
        forward_angle_range[1],
        (count, 1),
        device=env.device,
    ).squeeze(1)
    lateral_jitter = torch_rand_float(
        lateral_angle_jitter_range[0],
        lateral_angle_jitter_range[1],
        (count, 1),
        device=env.device,
    ).squeeze(1)
    lateral_sign = torch.where(
        torch.rand(count, device=env.device) < 0.5,
        -torch.ones(count, device=env.device),
        torch.ones(count, device=env.device),
    )
    lateral_angles = lateral_sign * (0.5 * math.pi) + lateral_jitter
    lateral = torch.rand(count, device=env.device) < lateral_target_probability
    return torch.where(lateral, lateral_angles, forward_angles), lateral


def resample_jump_commands(
    env,
    env_ids,
    forward_target_distance_range=(0.8, 1.8),
    lateral_target_distance_range=(0.45, 1.2),
    forward_angle_range=(-0.5, 0.5),
    lateral_angle_jitter_range=(-0.3, 0.3),
    lateral_target_probability=0.4,
    stationary_probability=0.5,
    approach_forward_velocity_range=(0.35, 0.8),
    approach_yaw_rate_range=(-0.3, 0.3),
    jump_probability=0.9,
    command_frame_range=(60, 90),
):
    """Sample stationary/running jumps with forward and lateral targets.

    Target displacement and approach velocity are expressed in the robot body
    frame. The target is converted to a fixed world position when the jump
    signal is issued.
    """
    if len(env_ids) == 0:
        return

    count = len(env_ids)
    forward_distance = torch_rand_float(
        forward_target_distance_range[0],
        forward_target_distance_range[1],
        (count, 1),
        device=env.device,
    ).squeeze(1)
    angle, lateral = _sample_target_angles(
        env,
        count,
        lateral_target_probability,
        forward_angle_range,
        lateral_angle_jitter_range,
    )
    lateral_distance = torch_rand_float(
        lateral_target_distance_range[0],
        lateral_target_distance_range[1],
        (count, 1),
        device=env.device,
    ).squeeze(1)
    distance = torch.where(lateral, lateral_distance, forward_distance)
    env.commands[env_ids, TARGET_X_INDEX] = distance * torch.cos(angle)
    env.commands[env_ids, TARGET_Y_INDEX] = distance * torch.sin(angle)

    stationary = torch.rand(count, device=env.device) < stationary_probability
    approach_vx = torch_rand_float(
        approach_forward_velocity_range[0],
        approach_forward_velocity_range[1],
        (count, 1),
        device=env.device,
    ).squeeze(1)
    approach_yaw_rate = torch_rand_float(
        approach_yaw_rate_range[0],
        approach_yaw_rate_range[1],
        (count, 1),
        device=env.device,
    ).squeeze(1)
    env.commands[env_ids, APPROACH_VX_INDEX] = torch.where(
        stationary, torch.zeros_like(approach_vx), approach_vx
    )
    env.commands[env_ids, APPROACH_YAW_RATE_INDEX] = torch.where(
        stationary, torch.zeros_like(approach_yaw_rate), approach_yaw_rate
    )
    env.commands[env_ids, JUMP_SIGNAL_INDEX] = 0.0

    env.stationary_jump_command[env_ids] = stationary
    env.lateral_jump_command[env_ids] = lateral
    env.enable_jump_cmd[env_ids] = (
        torch.rand(count, device=env.device) < jump_probability
    )
    env.command_frame[env_ids] = torch.randint(
        int(command_frame_range[0]),
        int(command_frame_range[1]),
        (count,),
        device=env.device,
    )


def body_target_to_world(env, env_ids):
    """Return commanded body-frame target displacements in world axes."""
    yaw = env.base_euler_xyz[env_ids, 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    target_x = env.commands[env_ids, TARGET_X_INDEX]
    target_y = env.commands[env_ids, TARGET_Y_INDEX]
    return torch.stack(
        (
            cos_yaw * target_x - sin_yaw * target_y,
            sin_yaw * target_x + cos_yaw * target_y,
        ),
        dim=1,
    )


def jump_heading_error(env):
    """Return error to the commanded heading trajectory during a jump.

    The heading at signal onset is the trajectory origin. Running jumps keep
    following their sampled approach yaw rate, while stationary jumps hold the
    takeoff heading because their commanded yaw rate is zero.
    """
    elapsed = (
        env.episode_length_buf - env.jump_signal_step
    ).float().clamp(min=0.0) * env.dt
    desired_heading = (
        env.jump_start_yaw
        + env.commands[:, APPROACH_YAW_RATE_INDEX] * elapsed
    )
    error = wrap_to_pi(desired_heading - env.base_euler_xyz[:, 2])
    active = env.jump_signal_issued & (~env.has_jumped)
    return torch.where(active, error, torch.zeros_like(error))


def activate_jump_signal(env):
    """Issue a jump once the approach speed is ready, with a bounded delay."""
    waiting = (
        env.enable_jump_cmd
        & (~env.jump_signal_issued)
        & (env.episode_length_buf >= env.command_frame)
    )
    if not torch.any(waiting):
        return waiting

    forward_error = torch.abs(
        env.base_lin_vel[:, 0] - env.commands[:, APPROACH_VX_INDEX]
    )
    lateral_speed = torch.abs(env.base_lin_vel[:, 1])
    yaw_rate_error = torch.abs(
        env.base_ang_vel[:, 2] - env.commands[:, APPROACH_YAW_RATE_INDEX]
    )
    ready = (
        (forward_error <= env.cfg.commands.readiness_velocity_tolerance)
        & (lateral_speed <= env.cfg.commands.readiness_lateral_velocity_tolerance)
        & (yaw_rate_error <= env.cfg.commands.readiness_yaw_rate_tolerance)
    )
    max_delay_steps = int(
        round(env.cfg.commands.max_trigger_delay_s / env.dt)
    )
    deadline = env.episode_length_buf >= env.command_frame + max_delay_steps
    trigger = waiting & (ready | deadline)
    if not torch.any(trigger):
        return trigger

    trigger_ids = trigger.nonzero(as_tuple=False).flatten()
    env.commands[trigger, JUMP_SIGNAL_INDEX] = 1.0
    env.jump_signal_issued[trigger] = True
    env.jump_signal_step[trigger] = env.episode_length_buf[trigger]
    env.jump_start_yaw[trigger] = env.base_euler_xyz[trigger, 2]
    env.jump_origins[trigger] = env.root_states[trigger, :2]
    env.landing_targets[trigger] = (
        env.jump_origins[trigger] + body_target_to_world(env, trigger_ids)
    )
    env.pre_jump_displacement[trigger] = torch.linalg.norm(
        env.jump_origins[trigger] - env.init_state[trigger, :2], dim=1
    )
    return trigger
