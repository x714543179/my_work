"""Configuration for manager-based reproduction of the Go1 Omni-Jump task."""

from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.go1 import mdp
from legged_gym.managers import ManagerTermCfg, ObsGroup


class Go1OmniJumpCfg(BaseConfig):
    task_name = "Omni_Jump_go1"

    class env:
        num_envs = 4096
        num_actions = 12
        # Paper: 46-D proprioception, 20 frames, 10-D estimate, 243-D critic.
        num_observations = 46
        num_obs_hist = 20
        num_privileged_obs = 243
        env_spacing = 3.0
        send_timeouts = True
        # One randomly triggered jump attempt per episode.
        episode_length_s = 4.0

    class terrain:
        # Omni-Jump's training height field is flat; a plane avoids allocating a
        # large all-zero trimesh while preserving its 187 height samples.
        mesh_type = "plane"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 0.2
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = True
        measured_points_x = [round(-0.8 + 0.1 * index, 1) for index in range(17)]
        measured_points_y = [round(-0.5 + 0.1 * index, 1) for index in range(11)]
        max_init_terrain_level = 0
        terrain_length = 14.0
        terrain_width = 14.0
        num_rows = 1
        num_cols = 1
        slope_treshold = 0.75

        class importer:
            terrain_type = "generator"
            mesh_type = "plane"
            max_init_terrain_level = 0
            use_terrain_origins = False

    class commands:
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4
        # Targets are sampled on reset and held for the complete jump cycle.
        resampling_time = 100.0
        heading_command = False
        height_observation_scale = 2.0
        small_command_threshold = 0.20
        trigger_delay_s = [0.5, 1.0]
        takeoff_timeout_s = 1.5

        resample = ManagerTermCfg(func=mdp.resample_omni_jump_commands, mode="resample", env_arg=True)

        class ranges:
            lin_vel_x = [-0.2, 1.0]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-0.8, 0.8]
            # The paper demonstrates 0.52 m and 0.66 m commands but does not
            # publish a training range. This continuous interval brackets both.
            height_z = [0.50, 0.68]
            heading = [-1.0, 1.0]

    class init_state:
        pos = [0.0, 0.0, 0.34]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

        # Hip noise is kept smaller because left/right abduction asymmetry
        # creates a strong roll impulse immediately after reset.
        hip_joint_position_noise_range = [-0.03, 0.03]
        joint_position_noise_range = [-0.10, 0.10]
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.7220,
            "FL_calf_joint": -1.4441,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.7220,
            "FR_calf_joint": -1.4441,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.7220,
            "RL_calf_joint": -1.4441,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.7220,
            "RR_calf_joint": -1.4441,
        }
        init_joint_angles = default_joint_angles
        # Analytical-IK aerial target used by the paper's q_air reward.
        air_pose_joint_angles = {
            "FL_hip_joint": 0.04,
            "FL_thigh_joint": 1.16565784,
            "FL_calf_joint": -2.43440387,
            "FR_hip_joint": -0.04,
            "FR_thigh_joint": 1.16565784,
            "FR_calf_joint": -2.43440387,
            "RL_hip_joint": 0.04,
            "RL_thigh_joint": 1.68571010,
            "RL_calf_joint": -2.31866118,
            "RR_hip_joint": -0.04,
            "RR_thigh_joint": 1.68571010,
            "RR_calf_joint": -2.31866118,
        }
        # The paper does not publish its Go1 q_pre target. This intermediate
        # analytical-IK pose extends the legs between q_air and q_ground.
        prelanding_pose_joint_angles = {
            "FL_hip_joint": 0.02,
            "FL_thigh_joint": 0.90,
            "FL_calf_joint": -1.84,
            "FR_hip_joint": -0.02,
            "FR_thigh_joint": 0.90,
            "FR_calf_joint": -1.84,
            "RL_hip_joint": 0.02,
            "RL_thigh_joint": 1.11,
            "RL_calf_joint": -1.79,
            "RR_hip_joint": -0.02,
            "RR_thigh_joint": 1.11,
            "RR_calf_joint": -1.79,
        }
        landing_pose_joint_angles = default_joint_angles

    class control:
        control_type = "P"
        stiffness = {"joint": 40.0}
        damping = {"joint": 1.2}
        action_scale = 0.25
        vel_scale = 0.0
        decimation = 4

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf"
        name = "go1"
        foot_name = "foot"
        wheel_name = []
        joint_name = ["hip", "thigh", "calf"]
        penalize_contacts_on = ["thigh", "calf"]
        # Match the local Nezha task: leg contacts are penalized, while only a
        # base collision terminates the episode.
        terminate_after_contacts_on = ["base"]
        self_collisions = 1
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        thickness = 0.01

    class actions:
        joint_position = ManagerTermCfg(func=mdp.joint_position_action, mode="policy", env_arg=True)

    class observations:
        class actor(ObsGroup):
            base_ang_vel = ManagerTermCfg(func=mdp.base_angular_velocity, env_arg=True)
            projected_gravity = ManagerTermCfg(func=mdp.projected_gravity, env_arg=True)
            velocity_commands = ManagerTermCfg(func=mdp.velocity_commands, env_arg=True)
            jump_height = ManagerTermCfg(func=mdp.jump_height_command, env_arg=True)
            joint_position = ManagerTermCfg(func=mdp.joint_position, env_arg=True)
            joint_velocity = ManagerTermCfg(func=mdp.joint_velocity, env_arg=True)
            previous_action = ManagerTermCfg(func=mdp.previous_action, env_arg=True)

        class critic(ObsGroup):
            state_target = ManagerTermCfg(func=mdp.estimator_target, env_arg=True)
            policy = ManagerTermCfg(func=mdp.policy_observation, env_arg=True)
            environment = ManagerTermCfg(func=mdp.environment_context, env_arg=True)

    class rewards:
        only_positive_rewards = False
        tracking_sigma = 0.25
        height_tracking_sigma = 0.05
        soft_dof_pos_limit = 1.0
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.9
        stance_height = 0.34
        settled_height = 0.34
        jump_height_tolerance = 0.04
        landing_orientation_tolerance = 0.5
        landing_contact_force_threshold = 1.0
        prelanding_vertical_velocity = -0.05
        landing_pose_duration_s = 0.15
        max_contact_force = 500.0

    class rewards_manager:
        height_tracking = ManagerTermCfg(func=mdp.height_tracking, scale=4.0, env_arg=True)
        takeoff_vertical_velocity = ManagerTermCfg(
            func=mdp.takeoff_vertical_velocity, scale=2.0, env_arg=True
        )
        flight_bonus = ManagerTermCfg(func=mdp.flight_bonus, scale=1.0, env_arg=True)
        jump_success = ManagerTermCfg(
            func=mdp.jump_success, scale=20.0, use_dt=False, env_arg=True
        )
        no_jump_timeout = ManagerTermCfg(
            func=mdp.no_jump_timeout, scale=-5.0, use_dt=False, env_arg=True
        )
        tracking_planar_velocity = ManagerTermCfg(func=mdp.tracking_planar_velocity, scale=1.5, env_arg=True)
        tracking_yaw_rate = ManagerTermCfg(func=mdp.tracking_yaw_rate, scale=0.6, env_arg=True)
        orientation_l2 = ManagerTermCfg(func=mdp.orientation_l2, scale=-0.8, env_arg=True)
        stance_joint_pose = ManagerTermCfg(func=mdp.stance_joint_pose, scale=-0.1, env_arg=True)
        aerial_joint_pose = ManagerTermCfg(func=mdp.aerial_joint_pose, scale=-0.4, env_arg=True)
        prelanding_joint_pose = ManagerTermCfg(func=mdp.prelanding_joint_pose, scale=-0.6, env_arg=True)
        landing_joint_pose = ManagerTermCfg(func=mdp.landing_joint_pose, scale=-0.12, env_arg=True)
        collision = ManagerTermCfg(func=mdp.collision, scale=-1.0, env_arg=True)
        joint_torque_l2 = ManagerTermCfg(func=mdp.joint_torque_l2, scale=-1.0e-5, env_arg=True)
        action_rate_l2 = ManagerTermCfg(func=mdp.action_rate_l2, scale=-0.01, env_arg=True)
        joint_acceleration_l2 = ManagerTermCfg(func=mdp.joint_acceleration_l2, scale=-2.5e-7, env_arg=True)

    class terminations:
        illegal_contact = ManagerTermCfg(
            func=mdp.illegal_contact, env_arg=True, params={"force_threshold": 1.0}
        )
        excessive_roll = ManagerTermCfg(func=mdp.excessive_roll, env_arg=True, params={"roll_limit": 2.4})
        below_terrain = ManagerTermCfg(
            func=mdp.fallen_below_terrain, env_arg=True, params={"clearance": -0.5}
        )
        jump_cycle_complete = ManagerTermCfg(func=mdp.jump_cycle_complete, env_arg=True)

    class events:
        height_measurement = ManagerTermCfg(func=mdp.update_height_measurements, mode="step", env_arg=True)
        system_delay = ManagerTermCfg(
            func=mdp.randomize_system_delay,
            mode="reset",
            env_arg=True,
            params={"delay_range_ms": [0.0, 4.0]},
        )
        friction = ManagerTermCfg(
            func=mdp.randomize_friction,
            mode="asset_init",
            env_arg=True,
            params={"friction_range": [0.2, 1.25]},
        )
        rigid_body_props = ManagerTermCfg(
            func=mdp.randomize_rigid_body_properties,
            mode="asset_init",
            env_arg=True,
            params={"added_base_mass_range": [-2.0, 2.0], "base_com_range": [-0.05, 0.05]},
        )
        dof_props = ManagerTermCfg(func=mdp.initialize_dof_properties, mode="asset_init", env_arg=True)
        pd_gains = ManagerTermCfg(
            func=mdp.randomize_motor_strength,
            mode="asset_init",
            env_arg=True,
            params={"strength_range": [0.9, 1.1]},
        )

    class normalization:
        contact_force_range = [0.0, 500.0]

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
        add_noise = False
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.0
            quat = 0.0
            height_measurements = 0.1

    class viewer:
        pos = [-3.0, -3.0, 2.0]
        lookat = [0.0, 0.0, 0.4]
        follow_robot = True

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


class Go1OmniJumpCfgPPO(BaseConfig):
    seed = 1

    actor = {
        "class_name": "ActorModel",
        "backbone": {
            "class_name": "rsl_rl.models:OmniJumpActorBackbone",
            "actor_hidden_dims": [512, 256, 128],
            "estimator_hidden_dims": [258, 128],
            "state_dim": 10,
            "state_target_group": "critic_state_target",
            "activation": "elu",
            "estimator_loss_coef": 1.0,
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
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        runner_class_name = "rsl_rl.runners:OnPolicyRunner"
        logger = "wandb"
        wandb_project = "omni_jump_go1"
        wandb_mode = "online"
        wandb_group = "one_shot_jump"
        wandb_tags = ["go1", "omninet", "paper", "one-shot-trigger"]
        save_interval = 400
        run_name = "paper_h20_est10_one_shot"
        experiment_name = "omni_jump_go1_one_shot"
        num_steps_per_env = 24
        max_iterations = 5000
        load_run = -1
        checkpoint = -1
        resume = False
        resume_path = None
