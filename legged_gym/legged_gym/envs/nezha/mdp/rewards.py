"""Phase-aware rewards for stationary, running, and lateral Nezha jumps."""

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from .commands import (
    APPROACH_VX_INDEX,
    APPROACH_YAW_RATE_INDEX,
    JUMP_SIGNAL_INDEX,
    jump_heading_error,
)


def _leg_positions(env):
    position = env.dof_pos.clone()
    position[:, env.wheel_indices] = 0.0
    return position


def _pre_jump(env):
    return (
        (env.commands[:, JUMP_SIGNAL_INDEX] == 0.0)
        & (~env.jump_signal_issued)
        & (~env.has_jumped)
    )


def before_setting(env):
    error = torch.sum(torch.abs(_leg_positions(env) - env.default_dof_pos), dim=1)
    return torch.exp(-0.5 * error) * _pre_jump(env).float()


def approach_velocity(env, tracking_sigma=0.25):
    forward_error = torch.square(
        env.base_lin_vel[:, 0] - env.commands[:, APPROACH_VX_INDEX]
    )
    lateral_slip = torch.square(env.base_lin_vel[:, 1])
    yaw_rate_error = torch.square(
        env.base_ang_vel[:, 2] - env.commands[:, APPROACH_YAW_RATE_INDEX]
    )
    error = forward_error + lateral_slip + 0.5 * yaw_rate_error
    return torch.exp(-error / tracking_sigma) * _pre_jump(env).float()


def stationary_displacement(env):
    displacement = torch.sum(
        torch.square(env.root_states[:, :2] - env.init_state[:, :2]), dim=1
    )
    active = _pre_jump(env) & env.stationary_jump_command
    return displacement * active.float()


def line_z(env, takeoff_window_s=0.4):
    target_delta = env.landing_targets - env.jump_origins
    command_direction = target_delta / torch.norm(
        target_delta, dim=1, keepdim=True
    ).clamp(min=1.0e-6)
    velocity_along_command = torch.sum(
        env.root_states[:, 7:9] * command_direction, dim=1
    ).clamp(min=0.0)
    reward = env.root_states[:, 9] * torch.clamp(
        velocity_along_command / (torch.abs(env.root_states[:, 9]) + 1.0e-6),
        max=1.0,
    )
    time_since_signal = (
        env.episode_length_buf - env.jump_signal_step
    ).float() * env.dt
    in_takeoff_window = (
        (time_since_signal >= 0.0) & (time_since_signal < takeoff_window_s)
    )
    active = (
        (env.root_states[:, 9] > 0.0)
        & (~env.has_jumped)
        & (~env.was_in_flight)
        & (env.commands[:, JUMP_SIGNAL_INDEX] == 1.0)
        & in_takeoff_window
    )
    return reward * active.float()


def land_pos(env):
    landing_error = env.landing_targets - env.landing_poses
    stable = torch.linalg.norm(env.base_euler_xyz[:, :2], dim=1) < 0.6
    high_enough = env.max_height > 0.65
    return (
        torch.exp(-torch.sum(torch.abs(landing_error), dim=1))
        * env.has_jumped.float()
        * stable.float()
        * high_enough.float()
    )


def base_height_flight(env):
    reward = torch.exp(-5.0 * torch.abs(env.root_states[:, 2] - 0.75)) * 6.0
    return reward * env.was_in_flight.float() * (~env.has_jumped).float()


def base_height_stance(env):
    preparing = _pre_jump(env)
    return (
        torch.abs(env.root_states[:, 2] - 0.58) * env.has_jumped.float()
        + 0.2
        * torch.abs(env.root_states[:, 2] - 0.38)
        * preparing.float()
    )


def dof_pos(env):
    return torch.sum(torch.abs(_leg_positions(env) - env.default_dof_pos), dim=1)


def dof_hip_pos(env):
    error = torch.abs(_leg_positions(env) - env.default_dof_pos)
    return torch.sum(error[:, [0, 4, 8, 12]], dim=1)


def hip_splay(env, tolerance=0.20, flight_multiplier=2.0):
    """Discourage excessive hip abduction without locking lateral push-off."""
    hip_position = env.dof_pos[:, [0, 4, 8, 12]]
    excess = (torch.abs(hip_position) - tolerance).clamp(min=0.0)
    jump_phase = env.jump_signal_issued & (~env.has_jumped)
    phase_multiplier = torch.where(
        env.was_in_flight,
        torch.full_like(excess[:, 0], flight_multiplier),
        torch.ones_like(excess[:, 0]),
    )
    return (
        torch.sum(torch.square(excess), dim=1)
        * phase_multiplier
        * jump_phase.float()
    )


def orientation(env):
    return torch.exp(-torch.sum(torch.abs(env.base_euler_xyz[:, :2]), dim=1))


def ang_vel_xy(env):
    return torch.sum(torch.abs(env.base_ang_vel[:, :2]), dim=1)


def jump_yaw_tracking(
    env,
    heading_sigma=0.25,
    yaw_rate_sigma=0.25,
):
    heading_error = jump_heading_error(env)
    yaw_rate_error = (
        env.base_ang_vel[:, 2]
        - env.commands[:, APPROACH_YAW_RATE_INDEX]
    )
    error = (
        torch.square(heading_error) / heading_sigma
        + torch.square(yaw_rate_error) / yaw_rate_sigma
    )
    active = env.jump_signal_issued & (~env.has_jumped)
    return torch.exp(-error) * active.float()


def torques(env):
    return torch.sum(torch.abs(env.torques), dim=1)


def action_rate(env):
    return torch.sum(torch.square(env.actions - env.last_actions), dim=1)


def collision(env):
    return torch.sum(
        (
            torch.linalg.norm(
                env.contact_forces[:, env.penalised_contact_indices, :], dim=-1
            )
            > 0.1
        ).float(),
        dim=1,
    )


def dof_pos_limits(env):
    outside = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.0)
    outside += (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(outside, dim=1)


def dof_vel_limits(env):
    return torch.sum(
        (torch.abs(env.dof_vel) - env.dof_vel_limits).clip(min=0.0), dim=1
    )


def feet_contact_forces(env):
    return torch.sum(
        (
            env.contact_forces[:, env.feet_indices, 2]
            - env.cfg.rewards.max_contact_force
        ).clip(min=0.0),
        dim=1,
    )


def dof_vel(env):
    return torch.sum(torch.square(env.dof_vel), dim=1)


def dof_vel_flight(env):
    return (
        torch.sum(torch.square(env.dof_vel), dim=1)
        * env.was_in_flight.float()
        * (~env.has_jumped).float()
    )


def dof_pos_flight(env):
    return (
        torch.sum(torch.abs(_leg_positions(env) - env.default_dof_pos), dim=1)
        * env.was_in_flight.float()
        * (~env.has_jumped).float()
    )


def wheel_speed_takeoff(env):
    wheel_speed = torch.sum(
        torch.square(env.dof_vel[:, env.wheel_indices]), dim=1
    )
    flight_phase = env.was_in_flight & (~env.has_jumped)
    stationary_stance = env.stationary_jump_command & (~flight_phase)
    return wheel_speed * (flight_phase | stationary_stance).float()


def front_rear_symmetry(env):
    position = env.dof_pos
    error = torch.sum(
        torch.square(
            position[:, [0, 1, 2]]
            - position[:, [8, 9, 10]]
        ),
        dim=1,
    )
    error += torch.sum(
        torch.square(
            position[:, [4, 5, 6]]
            - position[:, [12, 13, 14]]
        ),
        dim=1,
    )
    jump_phase = (
        (env.commands[:, JUMP_SIGNAL_INDEX] == 1.0) & (~env.has_jumped)
    )
    return error * jump_phase.float()


def flight(env):
    return env.was_in_flight.float() * (~env.has_jumped).float()


def tracking_lin_vel(env, expected_flight_time=0.55, tracking_sigma=0.5):
    target_velocity = (
        env.landing_targets - env.jump_origins
    ) / expected_flight_time
    error = torch.sum(
        torch.square(env.root_states[:, 7:9] - target_velocity), dim=1
    )
    return (
        torch.exp(-error / tracking_sigma)
        * env.was_in_flight.float()
        * (~env.has_jumped).float()
        * 5.0
    )


def post_landing_velocity(env, tracking_sigma=0.25):
    forward_error = torch.square(
        env.base_lin_vel[:, 0] - env.commands[:, APPROACH_VX_INDEX]
    )
    lateral_slip = torch.square(env.base_lin_vel[:, 1])
    yaw_rate_error = torch.square(
        env.base_ang_vel[:, 2] - env.commands[:, APPROACH_YAW_RATE_INDEX]
    )
    error = forward_error + lateral_slip + 0.5 * yaw_rate_error
    return torch.exp(-error / tracking_sigma) * env.has_jumped.float()


def foot_clearance(env):
    translated = env.feet_pos - env.root_states[:, :3].unsqueeze(1)
    base_quat = env.base_quat.unsqueeze(1).expand(-1, len(env.feet_indices), -1)
    foot_body = quat_rotate_inverse(
        base_quat.reshape(-1, 4), translated.reshape(-1, 3)
    ).view(env.num_envs, len(env.feet_indices), 3)
    height_error = torch.abs(foot_body[:, :, 2] + 0.25)
    return (
        torch.sum(height_error, dim=1)
        * env.was_in_flight.float()
        * (~env.has_jumped).float()
        * 6.0
    )
