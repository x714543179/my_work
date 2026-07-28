"""Reward terms used by the manager-based Omni-Jump reproduction."""

from __future__ import annotations

import torch


def tracking_planar_velocity(env):
    error = torch.sum(
        torch.square(env.jump_target_commands[:, :2] - env.base_lin_vel[:, :2]), dim=1
    )
    flight = env.jump_active & env.was_in_flight
    return torch.exp(-error / env.cfg.rewards.tracking_sigma) * flight.to(error.dtype)


def tracking_yaw_rate(env):
    error = torch.square(env.jump_target_commands[:, 2] - env.base_ang_vel[:, 2])
    flight = env.jump_active & env.was_in_flight
    return torch.exp(-error / env.cfg.rewards.tracking_sigma) * flight.to(error.dtype)


def orientation_l2(env):
    return torch.sum(torch.square(env.base_quat[:, :2]), dim=1)


def joint_torque_l2(env):
    return torch.sum(torch.square(env.torques), dim=1)


def joint_acceleration_l2(env):
    return torch.sum(torch.square((env.last_dof_vel - env.dof_vel) / env.dt), dim=1)


def action_rate_l2(env):
    return torch.sum(torch.square(env.actions - env.last_actions), dim=1)


def height_tracking(env):
    error = torch.square(env.task_peak_height - env.jump_target_commands[:, 3])
    flight = env.jump_active & env.was_in_flight
    return torch.exp(-error / env.cfg.rewards.height_tracking_sigma) * flight.to(error.dtype)


def takeoff_vertical_velocity(env):
    rising = torch.clamp(env.root_states[:, 9], min=0.0, max=3.0)
    takeoff_phase = env.jump_active & ~env.prelanding_phase
    return rising * takeoff_phase.to(rising.dtype)


def flight_bonus(env):
    return (env.jump_active & env.was_in_flight & env.airborne).to(env.root_states.dtype)


def no_jump_timeout(env):
    return env.just_timed_out.to(env.root_states.dtype)


def jump_success(env):
    return (env.just_landed & env.jump_succeeded).to(env.root_states.dtype)


def stance_joint_pose(env):
    error = torch.sum(torch.abs(env.dof_pos - env.landing_pose_dof_pos), dim=1)
    waiting = ~env.jump_triggered
    return error * waiting.to(error.dtype)


def collision(env):
    return torch.sum(
        (torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1).float(),
        dim=1,
    )


def aerial_joint_pose(env):
    error = torch.sum(torch.abs(env.dof_pos - env.air_pose_dof_pos), dim=1)
    return error * env.aerial_phase.to(error.dtype)


def prelanding_joint_pose(env):
    error = torch.sum(torch.abs(env.dof_pos - env.prelanding_pose_dof_pos), dim=1)
    return error * env.prelanding_phase.to(error.dtype)


def landing_joint_pose(env):
    error = torch.sum(torch.abs(env.dof_pos - env.landing_pose_dof_pos), dim=1)
    return error * env.landing_phase.to(error.dtype)
