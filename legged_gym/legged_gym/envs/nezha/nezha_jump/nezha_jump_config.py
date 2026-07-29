"""Manager configuration for mixed stationary and running Nezha jumps."""

from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.nezha import mdp
from legged_gym.managers import ManagerTermCfg, ObsGroup


_SINGLE_OBS_PERMUTATION = [
    0, -1, 2, -3, 4,
    -5, 6, -7, -8, 9, -10,
    -15, 16, 17, 18, -11, 12, 13, 14,
    -23, 24, 25, 26, -19, 20, 21, 22,
    -31, 32, 33, 34, -27, 28, 29, 30,
    -39, 40, 41, 42, -35, 36, 37, 38,
    -47, 48, 49, 50, -43, 44, 45, 46,
    -55, 56, 57, 58, -51, 52, 53, 54,
]
_ACTION_PERMUTATION = [
    -4, 5, 6, 7, -0.0001, 1, 2, 3,
    -12, 13, 14, 15, -8, 9, 10, 11,
]


def _stack_permutation(permutation, frame_count):
    stacked = []
    width = len(permutation)
    for frame in range(frame_count):
        offset = frame * width
        for value in permutation:
            sign = -1.0 if value < 0 else 1.0
            stacked.append(sign * (abs(value) + offset))
    return stacked


class NezhaJumpCfg(BaseConfig):
    task_name = "NEZHA_Spring_Jump"

    class env:
        num_envs = 4096
        num_actions = 16
        frame_stack = 10
        critic_frame_stack = 3
        num_single_obs = 59
        single_num_privileged_obs = 83
        num_observations = frame_stack * num_single_obs
        num_obs_hist = 1
        num_privileged_obs = critic_frame_stack * single_num_privileged_obs
        env_spacing = 3.0
        episode_length_s = 5.0
        send_timeouts = True
        reset_height = 0.2
        test = False

    class commands:
        curriculum = True
        # [target_dx_body, target_dy_body, approach_forward_velocity,
        #  approach_yaw_rate, jump_signal]
        num_commands = 5
        resampling_time = 1000.0
        heading_command = False
        jump_probability = 0.9
        stationary_probability = 0.5
        lateral_target_probability = 0.4
        command_frame_range = [60, 90]
        readiness_velocity_tolerance = 0.2
        readiness_lateral_velocity_tolerance = 0.2
        readiness_yaw_rate_tolerance = 0.2
        max_trigger_delay_s = 0.6

        class distance_curriculum:
            # Ranges expand only after the farthest 35% of the current level
            # is reliable. 60000 environment steps are about 2500 PPO updates.
            forward_levels = [
                [0.8, 1.10],
                [0.8, 1.35],
                [0.8, 1.60],
                [0.8, 1.80],
            ]
            lateral_levels = [
                [0.45, 0.70],
                [0.45, 0.80],
                [0.45, 0.95],
                [0.45, 1.075],
                [0.45, 1.20],
            ]
            initial_forward_level = 0
            initial_lateral_level = 0
            frontier_fraction = 0.35
            forward_success_threshold = 0.70
            lateral_success_threshold = 0.55
            min_samples = 64
            min_level_duration_steps = 30000

        resample = ManagerTermCfg(
            func=mdp.resample_jump_commands,
            mode="resample",
            env_arg=True,
            params={
                "forward_target_distance_range": [0.8, 1.8],
                "lateral_target_distance_range": [0.45, 1.2],
                "forward_angle_range": [-0.5, 0.5],
                "lateral_angle_jitter_range": [-0.3, 0.3],
                "lateral_target_probability": 0.4,
                "stationary_probability": 0.5,
                "approach_forward_velocity_range": [0.35, 0.8],
                "approach_yaw_rate_range": [-0.3, 0.3],
                "jump_probability": 0.9,
                "command_frame_range": [60, 90],
            },
        )

        class ranges:
            forward_target_distance = [0.8, 1.8]
            lateral_target_distance = [0.45, 1.2]
            target_angle = [-1.8708, 1.8708]
            approach_vx = [0.0, 0.8]
            approach_yaw_rate = [-0.3, 0.3]
            jump = [0.0, 1.0]

    class terrain:
        mesh_type = "plane"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25.0
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = False
        measured_points_x = [round(-0.8 + 0.1 * index, 1) for index in range(17)]
        measured_points_y = [round(-0.5 + 0.1 * index, 1) for index in range(11)]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]
        slope_treshold = 0.75

    class init_state:
        pos = [0.0, 0.0, 0.57]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.925,
            "FL_calf_joint": -1.85,
            "FL_foot_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.925,
            "FR_calf_joint": -1.85,
            "FR_foot_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.925,
            "RL_calf_joint": -1.85,
            "RL_foot_joint": 0.0,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.925,
            "RR_calf_joint": -1.85,
            "RR_foot_joint": 0.0,
        }
        init_joint_angles = default_joint_angles
        lie_joint_angles = default_joint_angles

    class control:
        control_type = "P"
        stiffness = {
            "hip_joint": 200.0,
            "thigh_joint": 200.0,
            "calf_joint": 200.0,
            "foot_joint": 0.0,
        }
        damping = {
            "hip_joint": 4.0,
            "thigh_joint": 4.0,
            "calf_joint": 4.0,
            "foot_joint": 1.0,
        }
        action_scale = 0.25
        vel_scale = 10.0
        decimation = 4

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/nezha/urdf/nezha.urdf"
        name = "nezha"
        foot_name = "foot"
        wheel_name = ["foot"]
        joint_name = ["hip", "thigh", "calf"]
        penalize_contacts_on = ["thigh", "calf", "trunk"]
        terminate_after_contacts_on = ["baseee"]
        self_collisions = 0
        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        default_dof_drive_mode = 3
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class actions:
        command_latency = ManagerTermCfg(
            func=mdp.command_latency,
            mode="decimation",
            env_arg=True,
            params={
                "enabled": True,
                "randomize": True,
                "latency_range": [1, 3],
            },
        )

    class observations:
        class actor(ObsGroup):
            history_length = 10
            enable_corruption = True

            command = ManagerTermCfg(func=mdp.actor_commands, env_arg=True)
            imu = ManagerTermCfg(
                func=mdp.actor_imu,
                env_arg=True,
                noise=mdp.actor_imu_noise,
                params={
                    "latency_enabled": True,
                    "randomize_latency": True,
                    "latency_range": [1, 3],
                },
            )
            motor = ManagerTermCfg(
                func=mdp.actor_motor,
                env_arg=True,
                noise=mdp.actor_motor_noise,
                params={
                    "latency_enabled": True,
                    "randomize_latency": True,
                    "latency_range": [1, 3],
                },
            )
            action = ManagerTermCfg(func=mdp.actor_actions, env_arg=True)

        class critic(ObsGroup):
            history_length = 3

            command = ManagerTermCfg(func=mdp.critic_commands, env_arg=True)
            dof_pos_error = ManagerTermCfg(func=mdp.critic_dof_pos_error, env_arg=True)
            dof_pos = ManagerTermCfg(func=mdp.critic_dof_pos, env_arg=True)
            dof_vel = ManagerTermCfg(func=mdp.critic_dof_vel, env_arg=True)
            action = ManagerTermCfg(func=mdp.critic_actions, env_arg=True)
            base_lin_vel = ManagerTermCfg(func=mdp.critic_base_lin_vel, env_arg=True)
            base_ang_vel = ManagerTermCfg(func=mdp.critic_base_ang_vel, env_arg=True)
            base_euler_xyz = ManagerTermCfg(func=mdp.critic_base_euler_xyz, env_arg=True)
            contact_mask = ManagerTermCfg(func=mdp.critic_contact_mask, env_arg=True)
            has_jumped = ManagerTermCfg(func=mdp.critic_has_jumped, env_arg=True)

    class rewards:
        only_positive_rewards = False
        reward_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        max_contact_force = 900.0

    class rewards_manager:
        before_setting = ManagerTermCfg(func=mdp.before_setting, scale=8.0, env_arg=True)
        approach_velocity = ManagerTermCfg(
            func=mdp.approach_velocity,
            scale=6.0,
            env_arg=True,
            params={"tracking_sigma": 0.25},
        )
        stationary_displacement = ManagerTermCfg(
            func=mdp.stationary_displacement,
            scale=-25.0,
            env_arg=True,
        )
        line_z = ManagerTermCfg(
            func=mdp.line_z,
            scale=16.0,
            env_arg=True,
            params={"takeoff_window_s": 0.4},
        )
        flight = ManagerTermCfg(func=mdp.flight, scale=2.0, env_arg=True)
        base_height_flight = ManagerTermCfg(func=mdp.base_height_flight, scale=3.0, env_arg=True)
        base_height_stance = ManagerTermCfg(func=mdp.base_height_stance, scale=-10.0, env_arg=True)
        orientation = ManagerTermCfg(func=mdp.orientation, scale=2.0, env_arg=True)
        dof_pos = ManagerTermCfg(func=mdp.dof_pos, scale=-0.1, env_arg=True)
        dof_hip_pos = ManagerTermCfg(func=mdp.dof_hip_pos, scale=-0.5, env_arg=True)
        hip_splay_takeoff = ManagerTermCfg(
            func=mdp.hip_splay_takeoff,
            scale=-8.0,
            env_arg=True,
            params={"tolerance": 0.20},
        )
        hip_tuck_flight = ManagerTermCfg(
            func=mdp.hip_tuck_flight,
            scale=-8.0,
            env_arg=True,
        )
        ang_vel_xy = ManagerTermCfg(func=mdp.ang_vel_xy, scale=-0.2, env_arg=True)
        jump_yaw_tracking = ManagerTermCfg(
            func=mdp.jump_yaw_tracking,
            scale=2.0,
            env_arg=True,
            params={"heading_sigma": 0.25, "yaw_rate_sigma": 0.25},
        )
        torques = ManagerTermCfg(func=mdp.torques, scale=-1.0e-4, env_arg=True)
        dof_pos_limits = ManagerTermCfg(func=mdp.dof_pos_limits, scale=-10.0, env_arg=True)
        dof_vel_limits = ManagerTermCfg(func=mdp.dof_vel_limits, scale=-1.0, env_arg=True)
        dof_vel = ManagerTermCfg(func=mdp.dof_vel, scale=-1.0e-3, env_arg=True)
        dof_vel_flight = ManagerTermCfg(func=mdp.dof_vel_flight, scale=-3.0e-3, env_arg=True)
        dof_pos_flight = ManagerTermCfg(func=mdp.dof_pos_flight, scale=-0.1, env_arg=True)
        collision = ManagerTermCfg(func=mdp.collision, scale=-5.0, env_arg=True)
        action_rate = ManagerTermCfg(func=mdp.action_rate, scale=-0.15, env_arg=True)
        feet_contact_forces = ManagerTermCfg(func=mdp.feet_contact_forces, scale=-0.1, env_arg=True)
        land_pos = ManagerTermCfg(func=mdp.land_pos, scale=30.0, env_arg=True)
        tracking_lin_vel = ManagerTermCfg(
            func=mdp.tracking_lin_vel,
            scale=10.0,
            env_arg=True,
            params={"expected_flight_time": 0.55, "tracking_sigma": 0.5},
        )
        post_landing_velocity = ManagerTermCfg(
            func=mdp.post_landing_velocity,
            scale=3.0,
            env_arg=True,
            params={"tracking_sigma": 0.25},
        )
        foot_clearance = ManagerTermCfg(func=mdp.foot_clearance, scale=-3.0, env_arg=True)
        wheel_speed_takeoff = ManagerTermCfg(func=mdp.wheel_speed_takeoff, scale=-7.0e-3, env_arg=True)
        sys_front_real = ManagerTermCfg(func=mdp.front_rear_symmetry, scale=-5.0e-3, env_arg=True)

    class terminations:
        base_height = ManagerTermCfg(
            func=mdp.base_height_below,
            env_arg=True,
            params={"minimum_height": 0.2},
        )

    class events:
        latency_update = ManagerTermCfg(
            func=mdp.update_observation_latency,
            mode="decimation",
            env_arg=True,
        )
        latency_reset = ManagerTermCfg(
            func=mdp.reset_latency_buffers,
            mode="reset",
            env_arg=True,
        )
        friction = ManagerTermCfg(
            func=mdp.randomize_friction,
            mode="asset_init",
            env_arg=True,
            params={"enabled": True, "friction_range": [0.3, 1.0]},
        )
        rigid_body_props = ManagerTermCfg(
            func=mdp.randomize_rigid_body_properties,
            mode="asset_init",
            env_arg=True,
            params={
                "added_base_mass_range": [-1.0, 1.0],
                "link_mass_scale_range": [0.9, 1.1],
                "base_com_range": [-0.03, 0.03],
                "base_body_name": "trunk",
            },
        )
        dof_props = ManagerTermCfg(
            func=mdp.init_dof_properties,
            mode="asset_init",
            env_arg=True,
        )
        motor_zero_offset = ManagerTermCfg(
            func=mdp.randomize_motor_zero_offset,
            mode="asset_init",
            env_arg=True,
            params={"enabled": True, "offset_range": [-0.035, 0.035]},
        )
        pd_gains = ManagerTermCfg(
            func=mdp.randomize_pd_gains,
            mode="asset_init",
            env_arg=True,
            params={
                "enabled": True,
                "stiffness_multiplier_range": [0.9, 1.1],
                "damping_multiplier_range": [0.9, 1.1],
            },
        )

    class jump_assist:
        enabled = False
        initial_probability_tenths = 8
        decay_interval_steps = 24 * 50
        vertical_velocity_range = [1.5, 2.2]

    class jump_metrics:
        success_min_height = 0.65
        success_max_landing_error = 0.30
        lateral_distance_bin_edges = [0.70, 0.95]

    class normalization:
        contact_force_range = [0.0, 100.0]

        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            quat = 1.0

        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            quat = 0.1
            height_measurements = 0.1

    class viewer:
        pos = [10.0, 0.0, 6.0]
        lookat = [11.0, 5.0, 3.0]
        follow_robot = False

    class plots:
        enabled = False

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class NezhaJumpCfgPPO(BaseConfig):
    seed = 1
    obs_groups = {
        "actor": ["actor"],
        "critic": ["critic"],
    }

    actor = {
        "class_name": "ActorModel",
        "backbone": {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": False,
        },
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    }

    critic = {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
    }

    class algorithm:
        class_name = "PPO"
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-5
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        plugins = [
            {
                "class_name": "rsl_rl.algorithms.plugins:SymmetryLossPlugin",
                "obs_permutation": _stack_permutation(_SINGLE_OBS_PERMUTATION, 10),
                "act_permutation": _ACTION_PERMUTATION,
                "frame_stack": 1,
                "sym_coef": 1.0,
            }
        ]

    class runner:
        runner_class_name = "rsl_rl.runners:OnPolicyRunner"
        logger = "wandb"
        wandb_project = "nezha_spring_jump"
        wandb_mode = "online"
        wandb_group = "manager_reproduction"
        wandb_tags = ["nezha", "spring-jump", "manager"]
        save_interval = 500
        run_name = "nezha_spring_jump_manager"
        experiment_name = "nezha_spring_jump"
        num_steps_per_env = 24
        max_iterations = 50000
        load_run = -1
        checkpoint = -1
        resume = False
        resume_path = None
