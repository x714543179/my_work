"""Thin Isaac Gym task shell for manager-based Go1 Omni-Jump training."""

from __future__ import annotations

import torch
from isaacgym import gymtorch

from legged_gym.envs.go2w.go2w_dreamwaq.go2w_robot import Go2w, get_euler_xyz_tensor

from .go1_omni_jump_config import Go1OmniJumpCfg


class Go1OmniJump(Go2w):
    """Go1 simulation plumbing with jump-cycle state owned by the task."""

    cfg: Go1OmniJumpCfg

    def _create_envs(self):
        super()._create_envs()
        canonical_feet = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        self.feet_indices = torch.tensor(
            [
                self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
                for name in canonical_feet
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _init_buffers(self):
        super()._init_buffers()

        rigid_body_states = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)
        self.foot_positions = rigid_body_states[:, self.feet_indices, :3].clone()
        self.foot_velocities = rigid_body_states[:, self.feet_indices, 7:10].clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.jump_last_contacts = torch.zeros_like(self.contacts)
        self.airborne = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grounded_before_jump = torch.zeros_like(self.airborne)
        self.jump_triggered = torch.zeros_like(self.airborne)
        self.jump_active = torch.zeros_like(self.airborne)
        self.was_in_flight = torch.zeros_like(self.airborne)
        self.jump_completed = torch.zeros_like(self.airborne)
        self.jump_failed = torch.zeros_like(self.airborne)
        self.jump_succeeded = torch.zeros_like(self.airborne)
        self.just_landed = torch.zeros_like(self.airborne)
        self.just_timed_out = torch.zeros_like(self.airborne)
        self.aerial_phase = torch.zeros_like(self.airborne)
        self.prelanding_phase = torch.zeros_like(self.airborne)
        self.landing_phase = torch.zeros_like(self.airborne)
        self.jump_target_commands = torch.zeros_like(self.commands)
        self.jump_trigger_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.takeoff_steps = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.jump_active_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.jump_airborne_steps = torch.zeros_like(self.jump_active_steps)
        self.landing_phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.landing_pose_steps = max(1, int(round(self.cfg.rewards.landing_pose_duration_s / self.dt)))
        trigger_delay = self.cfg.commands.trigger_delay_s
        self.min_trigger_steps = max(1, int(round(trigger_delay[0] / self.dt)))
        self.max_trigger_steps = max(self.min_trigger_steps + 1, int(round(trigger_delay[1] / self.dt)) + 1)
        self.takeoff_timeout_steps = max(
            1, int(round(self.cfg.commands.takeoff_timeout_s / self.dt))
        )
        self.task_peak_height = torch.full(
            (self.num_envs,), self.cfg.init_state.pos[2], dtype=torch.float, device=self.device
        )
        self.action_delay_fraction = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.control_substep = 0

        self.air_pose_dof_pos = self._joint_pose_tensor(self.cfg.init_state.air_pose_joint_angles)
        self.prelanding_pose_dof_pos = self._joint_pose_tensor(
            self.cfg.init_state.prelanding_pose_joint_angles
        )
        self.landing_pose_dof_pos = self._joint_pose_tensor(
            self.cfg.init_state.landing_pose_joint_angles
        )

    def _joint_pose_tensor(self, pose):
        values = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        for index, name in enumerate(self.dof_names):
            values[index] = pose[name]
        return values.unsqueeze(0)

    def _refresh_task_tensors(self):
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        rigid_body_states = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)
        self.foot_positions[:] = rigid_body_states[:, self.feet_indices, :3]
        self.foot_velocities[:] = rigid_body_states[:, self.feet_indices, 7:10]

        self.contacts[:] = self.contact_forces[:, self.feet_indices, 2] > 1.0
        filtered_contacts = self.contacts | self.jump_last_contacts
        self.airborne[:] = torch.all(~filtered_contacts, dim=1)
        grounded = torch.any(filtered_contacts, dim=1)
        waiting = ~self.jump_triggered
        self.grounded_before_jump |= waiting & grounded

        trigger_now = waiting & (self.episode_length_buf >= self.jump_trigger_steps)
        if torch.any(trigger_now):
            self.jump_triggered[trigger_now] = True

        self.jump_active[:] = self.jump_triggered & ~self.jump_completed & ~self.jump_failed
        self.commands[:, :3] = 0.0
        self.commands[:, 3] = self.cfg.rewards.stance_height
        self.commands[self.jump_active] = self.jump_target_commands[self.jump_active]

        takeoff_now = (
            self.jump_active
            & self.airborne
            & self.grounded_before_jump
            & ~self.was_in_flight
        )
        self.was_in_flight |= takeoff_now
        self.takeoff_steps[takeoff_now] = self.episode_length_buf[takeoff_now]

        self.just_landed[:] = self.jump_active & self.was_in_flight & grounded
        elapsed_since_trigger = self.episode_length_buf - self.jump_trigger_steps
        self.just_timed_out[:] = (
            self.jump_active
            & ~self.was_in_flight
            & (elapsed_since_trigger >= self.takeoff_timeout_steps)
        )

        self.task_peak_height[self.jump_active] = torch.maximum(
            self.task_peak_height[self.jump_active], self.root_states[self.jump_active, 2]
        )
        if torch.any(self.just_landed):
            height_error = torch.abs(
                self.task_peak_height[self.just_landed]
                - self.jump_target_commands[self.just_landed, 3]
            )
            landing_upright = torch.all(
                torch.abs(self.base_euler_xyz[self.just_landed, :2])
                <= self.cfg.rewards.landing_orientation_tolerance,
                dim=1,
            )
            landing_collision = torch.sum(
                torch.norm(
                    self.contact_forces[self.just_landed][
                        :, self.termination_contact_indices, :
                    ],
                    dim=-1,
                ),
                dim=1,
            ) > self.cfg.rewards.landing_contact_force_threshold
            self.jump_succeeded[self.just_landed] = (
                height_error <= self.cfg.rewards.jump_height_tolerance
            ) & landing_upright & ~landing_collision
            self.jump_completed[self.just_landed] = True
            self.landing_phase_steps[self.just_landed] = self.landing_pose_steps
        self.jump_failed |= self.just_timed_out

        self.jump_active[:] = self.jump_triggered & ~self.jump_completed & ~self.jump_failed
        inactive = ~self.jump_active
        self.commands[inactive, :3] = 0.0
        self.commands[inactive, 3] = self.cfg.rewards.stance_height

        active_flight = self.jump_active & self.airborne & self.was_in_flight
        self.prelanding_phase[:] = active_flight & (
            self.root_states[:, 9] < self.cfg.rewards.prelanding_vertical_velocity
        )
        self.aerial_phase[:] = active_flight & ~self.prelanding_phase
        self.landing_phase[:] = self.landing_phase_steps > 0
        self.jump_active_steps += self.jump_active.to(torch.long)
        self.jump_airborne_steps += active_flight.to(torch.long)
        self.jump_last_contacts[:] = self.contacts

    def _update_post_step_buffers(self):
        super()._update_post_step_buffers()
        self.landing_phase_steps[:] = torch.clamp_min(self.landing_phase_steps - 1, 0)
        self.just_landed[:] = False
        self.just_timed_out[:] = False

    def _actions_for_torque(self):
        if self.control_substep != 0:
            return self.actions
        delay = self.action_delay_fraction.unsqueeze(1)
        return delay * self.last_actions + (1.0 - delay) * self.actions

    def _post_decimation_step(self):
        self.control_substep = (self.control_substep + 1) % self.cfg.control.decimation

    def _reset_task_buffers(self, env_ids):
        super()._reset_task_buffers(env_ids)
        self.contacts[env_ids] = False
        self.jump_last_contacts[env_ids] = False
        self.airborne[env_ids] = False
        self.grounded_before_jump[env_ids] = False
        self.jump_triggered[env_ids] = False
        self.jump_active[env_ids] = False
        self.was_in_flight[env_ids] = False
        self.jump_completed[env_ids] = False
        self.jump_failed[env_ids] = False
        self.jump_succeeded[env_ids] = False
        self.just_landed[env_ids] = False
        self.just_timed_out[env_ids] = False
        self.aerial_phase[env_ids] = False
        self.prelanding_phase[env_ids] = False
        self.landing_phase[env_ids] = False
        self.jump_trigger_steps[env_ids] = torch.randint(
            self.min_trigger_steps,
            self.max_trigger_steps,
            (len(env_ids),),
            device=self.device,
        )
        self.takeoff_steps[env_ids] = -1
        self.jump_active_steps[env_ids] = 0
        self.jump_airborne_steps[env_ids] = 0
        self.landing_phase_steps[env_ids] = 0
        self.task_peak_height[env_ids] = self.root_states[env_ids, 2]
        self.obs_hist_buf[env_ids] = 0.0

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        attempted = self.jump_triggered[env_ids]
        took_off = self.was_in_flight[env_ids]
        active_steps = self.jump_active_steps[env_ids].clamp_min(1).to(torch.float)
        attempted_count = attempted.sum().clamp_min(1).to(torch.float)
        takeoff_count = took_off.sum().clamp_min(1).to(torch.float)
        peak_height_error = torch.abs(
            self.task_peak_height[env_ids] - self.jump_target_commands[env_ids, 3]
        )
        takeoff_latency = (
            self.takeoff_steps[env_ids] - self.jump_trigger_steps[env_ids]
        ).clamp_min(0).to(torch.float) * self.dt
        illegal_contact = torch.any(
            torch.norm(
                self.contact_forces[env_ids][:, self.termination_contact_indices, :], dim=-1
            )
            > self.cfg.terminations.illegal_contact.params["force_threshold"],
            dim=1,
        )
        excessive_roll = (
            torch.abs(self.base_euler_xyz[env_ids, 0])
            > self.cfg.terminations.excessive_roll.params["roll_limit"]
        )
        if torch.is_tensor(self.measured_heights):
            ground_height = torch.mean(self.measured_heights[env_ids], dim=1)
        else:
            ground_height = torch.zeros(
                len(env_ids), device=self.device, dtype=self.root_states.dtype
            )
        below_terrain = (
            self.root_states[env_ids, 2] - ground_height
            < self.cfg.terminations.below_terrain.params["clearance"]
        )
        jump_cycle_complete = self.jump_failed[env_ids] | (
            self.jump_completed[env_ids] & (self.landing_phase_steps[env_ids] <= 0)
        )
        completed_episode = self.episode_length_buf[env_ids] > 0

        diagnostics = {
            "jump_attempt_rate": attempted.float().mean(),
            "jump_takeoff_rate": took_off.float().mean(),
            "jump_landing_rate": self.jump_completed[env_ids].float().mean(),
            "jump_success_rate": self.jump_succeeded[env_ids].float().mean(),
            "jump_peak_height_error": (
                peak_height_error * attempted.float()
            ).sum() / attempted_count,
            "jump_airborne_fraction": (
                self.jump_airborne_steps[env_ids].to(torch.float) / active_steps
            ).sum() / attempted_count,
            "jump_takeoff_latency": (
                takeoff_latency * took_off.float()
            ).sum() / takeoff_count,
            "reset_illegal_contact_rate": (
                illegal_contact & completed_episode
            ).float().mean(),
            "reset_excessive_roll_rate": (
                excessive_roll & completed_episode
            ).float().mean(),
            "reset_below_terrain_rate": (
                below_terrain & completed_episode
            ).float().mean(),
            "reset_jump_cycle_rate": (
                jump_cycle_complete & completed_episode
            ).float().mean(),
            "reset_timeout_rate": (
                self.time_out_buf[env_ids] & completed_episode
            ).float().mean(),
        }

        super().reset_idx(env_ids)
        self.extras["episode"].update(diagnostics)

    def _reset_dofs(self, env_ids):
        joint_noise = torch.empty(
            len(env_ids), self.num_dof, dtype=self.dof_pos.dtype, device=self.device
        )
        for index, name in enumerate(self.dof_names):
            noise_range = (
                self.cfg.init_state.hip_joint_position_noise_range
                if "hip" in name
                else self.cfg.init_state.joint_position_noise_range
            )
            joint_noise[:, index].uniform_(*noise_range)
        self.dof_pos[env_ids] = torch.clamp(
            self.init_dof_pos + joint_noise,
            min=self.dof_pos_limits[:, 0],
            max=self.dof_pos_limits[:, 1],
        )
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """Reset upright with bounded linear and angular velocity noise."""
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
