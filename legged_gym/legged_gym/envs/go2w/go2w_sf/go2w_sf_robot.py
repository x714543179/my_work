from __future__ import annotations

from legged_gym.envs.go2w.go2w_dreamwaq.go2w_robot import Go2w
from legged_gym.envs.go2w import mdp

from .go2w_sf_config import GO2WSfCfg


class Go2wSf(Go2w):
    """GO2W environment variant for SF-TIM training."""

    cfg: GO2WSfCfg

    @staticmethod
    def get_euler_xyz_tensor(quat):
        return get_euler_xyz_tensor(quat)

    def _actions_for_torque(self):
        return self.action_manager.apply("decimation", self.actions)

    def _refresh_task_tensors(self):
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor)[:self.num_envs * self.num_bodies, :]
        self.rigid_body_states = self.rigid_body_state.view(self.num_envs, -1, 13)
        if hasattr(self, "robot_asset"):

            self.base_handles = self.gym.find_asset_rigid_body_index(self.robot_asset, "base_link")
            self.base_pos = self.rigid_body_states[:, self.base_handles][:, 0:3]


    def _reset_task_buffers(self, env_ids):
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])


    def _check_termination_impl(self):
        self.reset_buf = self._termination_illegal_contact()
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.base_contact_buf = mdp.base_height_contact(self)
        self.reset_buf |= self.base_contact_buf
       


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly 表示在z轴的重力作用
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        importer_cfg = get_importer_cfg(self.cfg.terrain)
        self.terrain_importer = TerrainImporter(importer_cfg, self.cfg.terrain, self.num_envs, self.device)
        self.terrain_importer.build()
        self.terrain_importer.add_to_sim(self.gym, self.sim)
        self.terrain_importer.attach_to_env(self)
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        return mdp.call_cfg_term(self, "events", "friction", props, env_id, default=props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        props = mdp.call_cfg_term(self, "events", "dof_props", props, env_id, default=props)
        mdp.call_cfg_term(self, "events", "motor_zero_offset", env_id=env_id)
        mdp.call_cfg_term(self, "events", "pd_gains", env_id=env_id)
        return props

    def _process_rigid_body_props(self, props, env_id):
        return mdp.call_cfg_term(self, "events", "rigid_body_props", props, env_id, default=props)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """  
        # 输出力矩用
        # self.log_dir = "./logs"  # 日志文件夹路径
        # if not os.path.exists(self.log_dir):  
        #     os.makedirs(self.log_dir)
        # self.log_file = os.path.join(self.log_dir, "torques.log")
        #pd controller
        dof_err = self.default_dof_pos - self.dof_pos # 各DOF默认位置 - 目前各DOF位置
        dof_err[:,self.wheel_indices] =  0 # 轮子的误差是0
        actions_scaled = actions * self.cfg.control.action_scale # action * 0.25
        actions_scaled[:, self.wheel_indices] = 0 # 轮子使用速度控制，角度增量为0
        vel_ref = torch.zeros_like(actions_scaled)
        vel_tmp = actions * self.cfg.control.vel_scale # action提供期望速度
        vel_ref[:, self.wheel_indices] = vel_tmp[:, self.wheel_indices] # 只有轮子使用速度控制
        control_type = self.cfg.control.control_type # 其实就是P

        p_gains = self.p_gains * self.p_gains_multiplier
        d_gains = self.d_gains * self.d_gains_multiplier

        if control_type=="P":
            torques = p_gains * (
                actions_scaled + dof_err + self.motor_zero_offsets
            ) + d_gains * (vel_ref - self.dof_vel)
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        # 输出力矩用
        # with open(self.log_file, "a") as f:  # 追加模式写入
        #     f.write(f"{torques.tolist()}\n")  # 将tensor转换为list再写入
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.init_dof_pos #* torch_rand_float(1, 1, (len(env_ids), self.num_dof), device=self.device)
        
        self.dof_vel[env_ids] = 0.
        #print("_reset_dofs:",env_ids)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        #print("_reset_root_states:",env_ids)
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)
                                  #torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _update_terrain_curriculum(self, env_ids):
        params = getattr(self.cfg.terrain, "sf_tim_curriculum", None)
        if params is not None:
            return mdp.sf_tim_terrain_levels(self, env_ids, **params)
        params = getattr(self.cfg.terrain, "mgdp_parkour_curriculum", None)
        if params is not None:
            return mdp.mgdp_parkour_terrain_levels(self, env_ids, **params)
        return super()._update_terrain_curriculum(env_ids)
