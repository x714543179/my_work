"""Manager-based Nezha spring jump environment."""

import os

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    get_axis_params,
    get_euler_xyz,
    quat_rotate_inverse,
    to_torch,
    torch_rand_float,
)

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.manager_based_task import ManagerBasedTask
from legged_gym.envs.nezha import mdp
from legged_gym.terrains import TerrainImporter, get_importer_cfg
from legged_gym.utils.helpers import class_to_dict

from .nezha_jump_config import NezhaJumpCfg


def get_euler_xyz_tensor(quat):
    roll, pitch, yaw = get_euler_xyz(quat)
    euler_xyz = torch.stack((roll, pitch, yaw), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2.0 * np.pi
    return euler_xyz


class NezhaJump(ManagerBasedTask):
    """Nezha wheel-legged landing-target jump task."""

    cfg: NezhaJumpCfg

    @staticmethod
    def get_euler_xyz_tensor(quat):
        return get_euler_xyz_tensor(quat)

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        if self.sim is None:
            raise RuntimeError("Failed to create the Isaac Gym simulation.")

        importer_cfg = get_importer_cfg(self.cfg.terrain)
        self.terrain_importer = TerrainImporter(
            importer_cfg,
            self.cfg.terrain,
            self.num_envs,
            self.device,
        )
        self.terrain_importer.build()
        self.terrain_importer.add_to_sim(self.gym, self.sim)
        self.terrain_importer.attach_to_env(self)
        self._create_envs()

    def set_camera(self, position, lookat):
        camera_position = gymapi.Vec3(*position)
        camera_target = gymapi.Vec3(*lookat)
        self.gym.viewer_camera_look_at(
            self.viewer, None, camera_position, camera_target
        )

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR
        )
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        if self.robot_asset is None:
            raise RuntimeError(f"Failed to load Nezha asset: {asset_path}")

        body_names = list(
            self.gym.get_asset_rigid_body_names(self.robot_asset)
        )
        self.dof_names = list(self.gym.get_asset_dof_names(self.robot_asset))
        self.num_bodies = len(body_names)
        self.num_dof = len(self.dof_names)
        self.num_dofs = self.num_dof
        if self.num_dof != self.num_actions:
            raise RuntimeError(
                f"Nezha has {self.num_dof} DOFs, expected {self.num_actions}."
            )

        target_feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        missing_feet = [name for name in target_feet if name not in body_names]
        if missing_feet:
            raise RuntimeError(f"Nezha asset is missing feet: {missing_feet}.")
        if "trunk" not in body_names:
            raise RuntimeError("Nezha asset does not contain the 'trunk' body.")

        penalized_contact_names = self._matching_names(
            body_names, self.cfg.asset.penalize_contacts_on
        )
        termination_contact_names = self._matching_names(
            body_names, self.cfg.asset.terminate_after_contacts_on
        )
        wheel_names = self._matching_names(
            self.dof_names, self.cfg.asset.wheel_name
        )
        joint_names = self._matching_names(
            self.dof_names, self.cfg.asset.joint_name
        )
        if len(wheel_names) != 4:
            raise RuntimeError(
                f"Expected four Nezha wheel joints, got {wheel_names}."
            )

        self.motor_zero_offsets = torch.zeros(
            self.num_envs, self.num_actions, device=self.device
        )
        self.p_gains_multiplier = torch.ones_like(self.motor_zero_offsets)
        self.d_gains_multiplier = torch.ones_like(self.motor_zero_offsets)

        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(
            self.robot_asset
        )
        initial_state = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            initial_state, device=self.device, requires_grad=False
        )

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        start_pose = gymapi.Transform()
        start_pose.r = gymapi.Quat(*self.cfg.init_state.rot)
        envs_per_row = max(1, int(np.sqrt(self.num_envs)))
        self.actor_handles = []
        self.envs = []

        for env_id in range(self.num_envs):
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, envs_per_row
            )
            start_position = (
                self.env_origins[env_id] + self.base_init_state[:3]
            )
            start_pose.p = gymapi.Vec3(*start_position)

            shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, env_id
            )
            self.gym.set_asset_rigid_shape_properties(
                self.robot_asset, shape_props
            )
            actor_handle = self.gym.create_actor(
                env_handle,
                self.robot_asset,
                start_pose,
                self.cfg.asset.name,
                env_id,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, env_id)
            self.gym.set_actor_dof_properties(
                env_handle, actor_handle, dof_props
            )
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, env_id)
            self.gym.set_actor_rigid_body_properties(
                env_handle,
                actor_handle,
                body_props,
                recomputeInertia=True,
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = self._body_indices(target_feet)
        self.wheel_body_names = target_feet
        self.wheel_body_indices = self.feet_indices.clone()
        self.penalised_contact_indices = self._body_indices(
            penalized_contact_names
        )
        self.termination_contact_indices = self._body_indices(
            termination_contact_names
        )
        self.wheel_indices = self._dof_indices(wheel_names)
        self.joint_indices = self._dof_indices(joint_names)
        self.base_body_index = self.gym.find_asset_rigid_body_index(
            self.robot_asset, "trunk"
        )

    @staticmethod
    def _matching_names(names, patterns):
        return [
            name
            for pattern in patterns
            for name in names
            if pattern in name
        ]

    def _body_indices(self, names):
        return torch.tensor(
            [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name
                )
                for name in names
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _dof_indices(self, names):
        return torch.tensor(
            [
                self.gym.find_actor_dof_handle(
                    self.envs[0], self.actor_handles[0], name
                )
                for name in names
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _process_rigid_shape_props(self, props, env_id):
        return self._call_event_term(
            "friction", props, env_id, default=props
        )

    def _process_dof_props(self, props, env_id):
        props = self._call_event_term(
            "dof_props", props, env_id, default=props
        )
        self._call_event_term("motor_zero_offset", env_id=env_id)
        self._call_event_term("pd_gains", env_id=env_id)
        return props

    def _process_rigid_body_props(self, props, env_id):
        return self._call_event_term(
            "rigid_body_props",
            props,
            env_id,
            default=props,
        )

    def _call_event_term(self, name, *args, default=None, **kwargs):
        term = getattr(self.cfg.events, name, None)
        if term is None or not term.enabled:
            return default
        func = term.func
        if isinstance(func, str):
            func = getattr(self, func)
        params = dict(term.params)
        params.update(kwargs)
        if term.env_arg:
            return func(self, *args, **params)
        return func(*args, **params)

    def _get_env_origins(self):
        self.terrain_importer.configure_env_origins(self)

    def _init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state)
        dof_state_view = self.dof_state.view(self.num_envs, self.num_dof, 2)
        self.dof_pos = dof_state_view[..., 0]
        self.dof_vel = dof_state_view[..., 1]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[
            : self.num_envs * self.num_bodies
        ]
        self.rigid_body_states = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )
        self.contact_forces = gymtorch.wrap_tensor(contact_forces).view(
            self.num_envs, self.num_bodies, 3
        )

        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat(self.num_envs, 1)
        self.forward_vec = to_torch(
            [1.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)
        self.projected_gravity = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )

        self.common_step_counter = 0
        self.extras = {}
        self.torques = torch.zeros(
            self.num_envs, self.num_actions, device=self.device
        )
        self.p_gains = torch.zeros(self.num_actions, device=self.device)
        self.d_gains = torch.zeros(self.num_actions, device=self.device)
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, device=self.device
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            device=self.device,
        )
        self.commands_scale = torch.tensor(
            [
                1.0,
                1.0,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                1.0,
            ],
            device=self.device,
        )
        self._env_indices = torch.arange(self.num_envs, device=self.device)
        self.feet_air_time = torch.zeros(
            self.num_envs, len(self.feet_indices), device=self.device
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
        )

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self._init_latency_buffers()
        self._init_control_buffers()
        self._init_jump_buffers()
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    def _init_latency_buffers(self):
        action_latency_range = self.cfg.actions.command_latency.params[
            "latency_range"
        ]
        motor_latency_range = self.cfg.observations.actor.motor.params[
            "latency_range"
        ]
        imu_latency_range = self.cfg.observations.actor.imu.params[
            "latency_range"
        ]
        action_buffer_length = int(action_latency_range[1]) + 1
        motor_buffer_length = int(motor_latency_range[1]) + 1
        imu_buffer_length = int(imu_latency_range[1]) + 1
        self.cmd_action_latency_buffer = torch.zeros(
            self.num_envs,
            self.num_actions,
            action_buffer_length,
            device=self.device,
        )
        self.obs_motor_latency_buffer = torch.zeros(
            self.num_envs,
            self.num_actions * 2,
            motor_buffer_length,
            device=self.device,
        )
        self.obs_imu_latency_buffer = torch.zeros(
            self.num_envs, 6, imu_buffer_length, device=self.device
        )
        self.cmd_action_latency_simstep = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.obs_motor_latency_simstep = torch.zeros_like(
            self.cmd_action_latency_simstep
        )
        self.obs_imu_latency_simstep = torch.zeros_like(
            self.cmd_action_latency_simstep
        )
        mdp.reset_latency_buffers(
            self,
            torch.arange(self.num_envs, device=self.device),
        )

    def _init_control_buffers(self):
        self.default_dof_pos = torch.zeros(
            self.num_dof, device=self.device
        )
        self.init_dof_pos = torch.zeros_like(self.default_dof_pos)
        for dof_id, name in enumerate(self.dof_names):
            self.default_dof_pos[dof_id] = (
                self.cfg.init_state.default_joint_angles[name]
            )
            self.init_dof_pos[dof_id] = (
                self.cfg.init_state.init_joint_angles[name]
            )
            for pattern, stiffness in self.cfg.control.stiffness.items():
                if pattern in name:
                    self.p_gains[dof_id] = stiffness
                    self.d_gains[dof_id] = self.cfg.control.damping[pattern]
                    break
            else:
                if self.cfg.control.control_type in ("P", "V"):
                    raise KeyError(f"No PD gains configured for joint {name!r}.")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.init_dof_pos = self.init_dof_pos.unsqueeze(0)

    def _init_jump_buffers(self):
        self.init_state = torch.zeros_like(self.root_states)
        self.enable_jump_cmd = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.command_frame = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.jump_signal_step = torch.zeros_like(self.command_frame)
        self.jump_start_yaw = torch.zeros(
            self.num_envs, device=self.device
        )
        self.jump_signal_issued = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.stationary_jump_command = torch.ones_like(
            self.jump_signal_issued
        )
        self.lateral_jump_command = torch.zeros_like(
            self.jump_signal_issued
        )
        self.max_height = torch.zeros(self.num_envs, device=self.device)
        self.contact_filt = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
        )
        self.was_in_flight = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.has_jumped = torch.zeros_like(self.was_in_flight)
        self.landing_poses = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self.jump_origins = torch.zeros_like(self.landing_poses)
        self.landing_targets = torch.zeros_like(self.landing_poses)
        self.pre_jump_displacement = torch.zeros(
            self.num_envs, device=self.device
        )
        self.pre_jump_velocity_error_sum = torch.zeros_like(
            self.pre_jump_displacement
        )
        self.pre_jump_velocity_steps = torch.zeros_like(
            self.command_frame
        )
        self.takeoff_velocity = torch.zeros_like(
            self.pre_jump_displacement
        )
        self.jump_heading_error_sum = torch.zeros_like(
            self.pre_jump_displacement
        )
        self.jump_heading_error_steps = torch.zeros_like(
            self.command_frame
        )
        self.max_hip_deviation = torch.zeros_like(
            self.pre_jump_displacement
        )
        self.max_hip_takeoff_deviation = torch.zeros_like(
            self.pre_jump_displacement
        )
        self.max_hip_flight_deviation = torch.zeros_like(
            self.pre_jump_displacement
        )
        self.max_hip_takeoff_splay = torch.zeros_like(
            self.pre_jump_displacement
        )
        self.max_hip_flight_splay = torch.zeros_like(
            self.pre_jump_displacement
        )
        self.not_pushed_up = torch.zeros_like(self.was_in_flight)
        self.assist_probability_tenths = 0
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, :3]

    def _get_noise_scale_vec(self, cfg):
        # Per-term corruption is applied before group history is stacked.
        self.add_noise = False
        return torch.zeros_like(self.obs_buf[0])

    def _actions_for_torque(self):
        return self.action_manager.apply("decimation", self.actions)

    def _refresh_task_tensors(self):
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, :3]
        self.max_height[:] = torch.maximum(
            self.max_height, self.root_states[:, 2]
        )
        before_signal = (~self.jump_signal_issued) & (~self.has_jumped)
        approach_error = torch.sqrt(
            torch.square(
                self.base_lin_vel[:, 0]
                - self.commands[:, mdp.APPROACH_VX_INDEX]
            )
            + torch.square(self.base_lin_vel[:, 1])
            + 0.5
            * torch.square(
                self.base_ang_vel[:, 2]
                - self.commands[:, mdp.APPROACH_YAW_RATE_INDEX]
            )
        )
        self.pre_jump_velocity_error_sum += (
            approach_error * before_signal.float()
        )
        self.pre_jump_velocity_steps += before_signal.long()
        mdp.activate_jump_signal(self)

        jump_phase = self.jump_signal_issued & (~self.has_jumped)
        heading_error = torch.abs(mdp.jump_heading_error(self))
        self.jump_heading_error_sum += heading_error * jump_phase.float()
        self.jump_heading_error_steps += jump_phase.long()
        hip_deviation = torch.max(
            torch.abs(self.dof_pos[:, [0, 4, 8, 12]]), dim=1
        ).values
        self.max_hip_deviation[:] = torch.maximum(
            self.max_hip_deviation,
            hip_deviation * jump_phase.float(),
        )
        hip_deviation_from_default, _, splay_mode = mdp.hip_modes(self)
        hip_max = torch.max(
            torch.abs(hip_deviation_from_default), dim=1
        ).values
        splay_abs = torch.abs(splay_mode)
        takeoff_phase = (
            self.jump_signal_issued
            & (~self.was_in_flight)
            & (~self.has_jumped)
        )
        flight_phase = self.was_in_flight & (~self.has_jumped)
        self.max_hip_takeoff_deviation[:] = torch.maximum(
            self.max_hip_takeoff_deviation,
            hip_max * takeoff_phase.float(),
        )
        self.max_hip_flight_deviation[:] = torch.maximum(
            self.max_hip_flight_deviation,
            hip_max * flight_phase.float(),
        )
        self.max_hip_takeoff_splay[:] = torch.maximum(
            self.max_hip_takeoff_splay,
            splay_abs * takeoff_phase.float(),
        )
        self.max_hip_flight_splay[:] = torch.maximum(
            self.max_hip_flight_splay,
            splay_abs * flight_phase.float(),
        )

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        self.contact_filt = contact | self.last_contacts
        self.last_contacts[:] = contact
        all_air = torch.all(~self.contact_filt, dim=1)
        jump_signal = self.commands[:, mdp.JUMP_SIGNAL_INDEX] > 0.0
        first_takeoff = all_air & jump_signal & (~self.was_in_flight)
        self.takeoff_velocity[first_takeoff] = torch.linalg.norm(
            self.root_states[first_takeoff, 7:9], dim=1
        )
        self.was_in_flight |= all_air & jump_signal

        landed = torch.any(self.contact_filt, dim=1) & self.was_in_flight
        first_landing = landed & (~self.has_jumped)
        self.landing_poses[first_landing] = self.root_states[
            first_landing, :2
        ]
        self.has_jumped |= landed
        self.commands[first_landing, mdp.JUMP_SIGNAL_INDEX] = 0.0

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled = actions_scaled.clone()
        actions_scaled[:, self.wheel_indices] = 0.0
        velocity_target = torch.zeros_like(actions)
        velocity_target[:, self.wheel_indices] = (
            actions[:, self.wheel_indices] * self.cfg.control.vel_scale
        )
        p_gains = self.p_gains * self.p_gains_multiplier
        d_gains = self.d_gains * self.d_gains_multiplier
        torques = p_gains * (
            actions_scaled
            + self.default_dof_pos
            - self.dof_pos
            + self.motor_zero_offsets
        ) + d_gains * (velocity_target - self.dof_vel)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def reset_idx(self, env_ids):
        episode_metrics = self._jump_episode_metrics(env_ids)
        if self.cfg.commands.curriculum:
            self.update_command_curriculum(env_ids)
        super().reset_idx(env_ids)
        if episode_metrics:
            self.extras["episode"].update(episode_metrics)

    def update_command_curriculum(self, env_ids):
        mdp.jump_distance_levels(self, env_ids)

    def _command_curriculum_metrics(self):
        return {
            "forward_curriculum_level": float(
                self.forward_curriculum_level
            ),
            "forward_curriculum_frontier_success": float(
                self.forward_curriculum_frontier_success
            ),
            "forward_curriculum_frontier_samples": float(
                self.forward_curriculum_frontier_samples
            ),
            "forward_curriculum_max_distance": float(
                self.command_ranges["forward_target_distance"][1]
            ),
            "lateral_curriculum_level": float(
                self.lateral_curriculum_level
            ),
            "lateral_curriculum_frontier_success": float(
                self.lateral_curriculum_frontier_success
            ),
            "lateral_curriculum_frontier_samples": float(
                self.lateral_curriculum_frontier_samples
            ),
            "lateral_curriculum_max_distance": float(
                self.command_ranges["lateral_target_distance"][1]
            ),
        }

    def _jump_episode_metrics(self, env_ids):
        if len(env_ids) == 0:
            return {}
        valid = self.episode_length_buf[env_ids] > 0
        commanded = self.jump_signal_issued[env_ids]
        metric_ids = env_ids[valid & commanded]
        if len(metric_ids) == 0:
            return {}

        target_xy = self.landing_targets[metric_ids]
        landing_error = torch.linalg.norm(
            target_xy - self.landing_poses[metric_ids], dim=1
        )
        no_landing_error = torch.linalg.norm(
            target_xy - self.jump_origins[metric_ids], dim=1
        )
        landed = self.has_jumped[metric_ids]
        reported_landing_error = torch.where(
            landed, landing_error, no_landing_error
        )
        reported_error_vector = torch.where(
            landed.unsqueeze(1),
            target_xy - self.landing_poses[metric_ids],
            target_xy - self.jump_origins[metric_ids],
        )
        successful = (
            landed
            & (
                self.max_height[metric_ids]
                >= self.cfg.jump_metrics.success_min_height
            )
            & (
                landing_error
                <= self.cfg.jump_metrics.success_max_landing_error
            )
        )
        velocity_steps = self.pre_jump_velocity_steps[metric_ids].clamp(min=1)
        approach_error = (
            self.pre_jump_velocity_error_sum[metric_ids]
            / velocity_steps.float()
        )
        heading_steps = self.jump_heading_error_steps[metric_ids].clamp(min=1)
        heading_error = (
            self.jump_heading_error_sum[metric_ids]
            / heading_steps.float()
        )
        metrics = {
            "jump_success_rate": successful.float().mean(),
            "max_height": self.max_height[metric_ids].mean(),
            "landing_error": reported_landing_error.mean(),
            "pre_jump_velocity_error": approach_error.mean(),
            "takeoff_velocity": self.takeoff_velocity[metric_ids].mean(),
            "jump_heading_error": heading_error.mean(),
            "max_hip_deviation": self.max_hip_deviation[metric_ids].mean(),
        }
        stationary = self.stationary_jump_command[metric_ids]
        running = ~stationary
        lateral = self.lateral_jump_command[metric_ids]
        if torch.any(stationary):
            metrics.update(
                {
                    "stationary_jump_success_rate": successful[
                        stationary
                    ].float().mean(),
                    "stationary_landing_error": reported_landing_error[
                        stationary
                    ].mean(),
                    "stationary_pre_jump_displacement": (
                        self.pre_jump_displacement[metric_ids][stationary].mean()
                    ),
                }
            )
        if torch.any(running):
            metrics.update(
                {
                    "running_jump_success_rate": successful[
                        running
                    ].float().mean(),
                    "running_landing_error": reported_landing_error[
                        running
                    ].mean(),
                    "running_pre_jump_velocity_error": approach_error[
                        running
                    ].mean(),
                }
            )
        if torch.any(lateral):
            target_delta = (
                target_xy - self.jump_origins[metric_ids]
            )
            target_distance = torch.linalg.norm(target_delta, dim=1)
            target_direction = target_delta / target_distance.unsqueeze(
                1
            ).clamp(min=1.0e-6)
            cross_direction = torch.stack(
                (-target_direction[:, 1], target_direction[:, 0]), dim=1
            )
            distance_error = torch.abs(
                torch.sum(reported_error_vector * target_direction, dim=1)
            )
            cross_track_error = torch.abs(
                torch.sum(reported_error_vector * cross_direction, dim=1)
            )
            metrics.update(
                {
                    "lateral_jump_success_rate": successful[
                        lateral
                    ].float().mean(),
                    "lateral_landing_error": reported_landing_error[
                        lateral
                    ].mean(),
                    "lateral_distance_error": distance_error[
                        lateral
                    ].mean(),
                    "lateral_cross_track_error": cross_track_error[
                        lateral
                    ].mean(),
                    "lateral_jump_heading_error": heading_error[
                        lateral
                    ].mean(),
                    "lateral_max_hip_deviation": self.max_hip_deviation[
                        metric_ids
                    ][lateral].mean(),
                    "lateral_takeoff_max_hip_deviation": (
                        self.max_hip_takeoff_deviation[metric_ids][
                            lateral
                        ].mean()
                    ),
                    "lateral_takeoff_splay": self.max_hip_takeoff_splay[
                        metric_ids
                    ][lateral].mean(),
                }
            )
            lateral_flight = lateral & self.was_in_flight[metric_ids]
            if torch.any(lateral_flight):
                metrics.update(
                    {
                        "lateral_flight_max_hip_deviation": (
                            self.max_hip_flight_deviation[metric_ids][
                                lateral_flight
                            ].mean()
                        ),
                        "lateral_flight_splay": self.max_hip_flight_splay[
                            metric_ids
                        ][lateral_flight].mean(),
                    }
                )

            near_edge, far_edge = (
                self.cfg.jump_metrics.lateral_distance_bin_edges
            )
            distance_bins = {
                "near": target_distance < near_edge,
                "mid": (target_distance >= near_edge)
                & (target_distance < far_edge),
                "far": target_distance >= far_edge,
            }
            for bin_name, distance_mask in distance_bins.items():
                bin_mask = lateral & distance_mask
                if not torch.any(bin_mask):
                    continue
                metrics.update(
                    {
                        f"lateral_{bin_name}_jump_success_rate": (
                            successful[bin_mask].float().mean()
                        ),
                        f"lateral_{bin_name}_landing_error": (
                            reported_landing_error[bin_mask].mean()
                        ),
                        f"lateral_{bin_name}_max_hip_deviation": (
                            self.max_hip_deviation[metric_ids][
                                bin_mask
                            ].mean()
                        ),
                    }
                )
        return metrics

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(
            -0.1,
            0.1,
            (len(env_ids), self.num_dof),
            device=self.device,
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
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.init_state[env_ids] = self.root_states[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_task_buffers(self, env_ids):
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.base_lin_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 7:10]
        )
        self.base_ang_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 10:13]
        )
        self.last_last_actions[env_ids] = 0.0
        self.was_in_flight[env_ids] = False
        self.has_jumped[env_ids] = False
        self.jump_signal_issued[env_ids] = False
        self.jump_signal_step[env_ids] = 0
        self.jump_start_yaw[env_ids] = self.base_euler_xyz[env_ids, 2]
        self.jump_origins[env_ids] = self.init_state[env_ids, :2]
        self.landing_targets[env_ids] = (
            self.jump_origins[env_ids]
            + mdp.body_target_to_world(self, env_ids)
        )
        self.landing_poses[env_ids] = self.init_state[env_ids, :2]
        self.max_height[env_ids] = 0.0
        self.pre_jump_displacement[env_ids] = 0.0
        self.pre_jump_velocity_error_sum[env_ids] = 0.0
        self.pre_jump_velocity_steps[env_ids] = 0
        self.takeoff_velocity[env_ids] = 0.0
        self.jump_heading_error_sum[env_ids] = 0.0
        self.jump_heading_error_steps[env_ids] = 0
        self.max_hip_deviation[env_ids] = 0.0
        self.max_hip_takeoff_deviation[env_ids] = 0.0
        self.max_hip_flight_deviation[env_ids] = 0.0
        self.max_hip_takeoff_splay[env_ids] = 0.0
        self.max_hip_flight_splay[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.contact_filt[env_ids] = False
        self.not_pushed_up[env_ids] = True

    def _update_post_step_buffers(self):
        self.last_last_actions[:] = self.last_actions
        super()._update_post_step_buffers()

        candidates = (
            self.not_pushed_up
            & (~self.has_jumped)
            & (self.commands[:, mdp.JUMP_SIGNAL_INDEX] == 1.0)
        )
        _, self.assist_probability_tenths = mdp.assist_jump(
            self, candidates
        )
        self.not_pushed_up[candidates] = False

    def _parse_cfg(self, cfg):
        self.dt = cfg.control.decimation * self.sim_params.dt
        self.obs_scales = cfg.normalization.obs_scales
        self.command_ranges = class_to_dict(cfg.commands.ranges)
        if cfg.commands.curriculum:
            mdp.initialize_jump_distance_curriculum(self)
        if cfg.terrain.mesh_type not in ("heightfield", "trimesh"):
            cfg.terrain.curriculum = False
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = int(
            np.ceil(self.max_episode_length_s / self.dt)
        )

    def _init_height_points(self):
        return mdp.init_height_points(self)

    def _get_heights(self, env_ids=None):
        return mdp.get_heights(self, env_ids=env_ids)

    def _draw_debug_vis(self):
        return None
