from __future__ import annotations

from legged_gym.managers import ManagerTermCfg, ObsGroup
from legged_gym.envs.go2w import mdp
from legged_gym.envs.go2w.go2w_dreamwaq.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO


_HEIGHT_POINTS_X = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_HEIGHT_POINTS_Y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
_MGDP_HEIGHT_POINTS_X = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
_MGDP_HEIGHT_POINTS_Y = _HEIGHT_POINTS_Y
_BASE_ACTOR_DIM = GO2WRoughCfg.env.num_observations
_ELEVATION_DIM = len(_HEIGHT_POINTS_X) * len(_HEIGHT_POINTS_Y)


class GO2WSfCfg(GO2WRoughCfg):
    task_name = "SF_TIM_go2w"

    class env(GO2WRoughCfg.env):
        num_envs = 4096
        num_actions = 16
        num_observations = _BASE_ACTOR_DIM + _ELEVATION_DIM
        num_obs_hist = 6
        num_privileged_obs = GO2WRoughCfg.env.num_privileged_obs + _ELEVATION_DIM
        episode_length_s = 20

    class commands(GO2WRoughCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 10.0
        heading_command = False

        class ranges:
            lin_vel_x = [0.0, 1.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

        resample = ManagerTermCfg(
            func=mdp.mgdp_mixed_resample_commands,
            env_arg=True,
            params={
                "easy_terrain_types": [],
                "first_stage_lin_vel_x": [-1.2, 1.2],
                "first_stage_lin_vel_y": [-1.0, 1.0],
                "first_stage_ang_vel_yaw": [-1.0, 1.0],
                "first_stage_heading": [-3.14, 3.14],
                "parkour_lin_vel_x": [0.0, 1.5],
            },
        )

    class terrain(GO2WRoughCfg.terrain):
        mesh_type = "trimesh"
        horizontal_scale = 0.05
        vertical_scale = 0.005
        border_size = 5
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.5
        measure_heights = True
        measured_points_x = _MGDP_HEIGHT_POINTS_X
        measured_points_y = _MGDP_HEIGHT_POINTS_Y
        selected = None
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 10.0
        terrain_width = 4.0
        num_goals = 10
        num_rows = 10
        num_cols = 10
        terrain_proportions = [1.0] * 10
        slope_treshold = 0.75
        add_roughness = True
        height = [0.01, 0.04]
        downsampled_scale = 0.5
        add_air_beam = True
        add_air_stone = True
        mgdp_parkour_curriculum = {
            "success_distance_ratio": 0.5,
            "log_metrics": True,
        }

        class importer:
            terrain_type = "generator"
            mesh_type = "trimesh"
            max_init_terrain_level = 5
            use_terrain_origins = True
            generator = {
                "class_name": "legged_gym.terrains.generators.gap_parkour:GapParkourTerrainGenerator",
                "difficulty_range": [0.0, 1.0],
                "num_goals": 10,
                "sub_terrains": {
                    "single_gap": {
                        "class_name": "legged_gym.terrains.generators.mgdp:ParkourGapTerrain",
                        "proportion": 2.0,
                        "terrain_type": 3,
                    },
                    "step_stone": {
                        "class_name": "legged_gym.terrains.generators.mgdp:ParkourStepTerrain",
                        "proportion": 1.0,
                        "terrain_type": 4,
                    },
                    "stones_2rows": {
                        "class_name": "legged_gym.terrains.generators.mgdp:ParkourStepTerrain",
                        "proportion": 1.0,
                        "num_stones": 2,
                        "terrain_type": 5,
                    },
                    "stones_1row": {
                        "class_name": "legged_gym.terrains.generators.mgdp:ParkourStepTerrain",
                        "proportion": 1.0,
                        "num_stones": 1,
                        "terrain_type": 7,
                    },

                    "air_beams": {
                        "class_name": "legged_gym.terrains.generators.mgdp:BeamTerrain",
                        "proportion": 1.0,
                        "terrain_type": 13,
                    },
                    "air_stone": {
                        "class_name": "legged_gym.terrains.generators.mgdp:AirStoneTerrain",
                        "proportion": 2.0,
                        "terrain_type": 14,
                    },
                    "slope_replaces_hurdle": {
                        "class_name": "legged_gym.terrains.generators.rough:PyramidSlopeTerrain",
                        "proportion": 1.0,
                        "terrain_type": 15,
                    },
                    "ramp": {
                        "class_name": "legged_gym.terrains.generators.mgdp:RampTerrain",
                        "proportion": 1.0,
                        "terrain_type": 16,
                    },

                },
            }

    class init_state(GO2WRoughCfg.init_state):
        pos = [-4.0, 0.0, 0.5]

    class observations:
        class actor(ObsGroup):
            imu = ManagerTermCfg(
                func=mdp.imu,
                env_arg=True,
                noise=mdp.imu_noise,
                params={
                    "latency_enabled": True,
                    "randomize_latency": True,
                    "latency_range": [1, 3],
                },
            )
            command = ManagerTermCfg(func="_obs_commands")
            motor = ManagerTermCfg(
                func=mdp.motor,
                env_arg=True,
                noise=mdp.motor_noise,
                params={
                    "latency_enabled": True,
                    "randomize_latency": True,
                    "latency_range": [1, 3],
                },
            )
            dof_pos = ManagerTermCfg(func=mdp.dof_pos, env_arg=True, noise=mdp.dof_pos_noise)
            action = ManagerTermCfg(func="_obs_actions")
            terrain = ManagerTermCfg(
                func=mdp.elevation_map,
                env_arg=True,
                params={
                    "base_height_offset": 0.5,
                    "clip": [-1.0, 1.0],
                    "add_noise": True,
                    "noise_ratio_range": [0.0, 0.1],
                    "noise_magnitude_range": [-1.0, 2.0],
                },
            )

        class critic(ObsGroup):
            policy = ManagerTermCfg(func="_obs_policy")
            base_lin_vel = ManagerTermCfg(func="_obs_base_lin_vel")
            contact_forces = ManagerTermCfg(func="_obs_contact_forces")
            heights = ManagerTermCfg(func="_obs_height_measurements")

    class rewards_manager(GO2WRoughCfg.rewards_manager):
        pass

    class noise(GO2WRoughCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(GO2WRoughCfg.noise.noise_scales):
            height_measurements = 0.1


class GO2WSfTrotCfg(GO2WSfCfg):
    task_name = "SF_TIM_go2w_trot"

    class commands(GO2WRoughCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 10.0
        heading_command = False

        class ranges:
            lin_vel_x = [-1.2, 1.2]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

        resample = ManagerTermCfg(func=mdp.resample_commands, env_arg=True)

    class terrain(GO2WSfCfg.terrain):
        mesh_type = "trimesh"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.5
        measure_heights = True
        measured_points_x = _HEIGHT_POINTS_X
        measured_points_y = _HEIGHT_POINTS_Y
        curriculum = True
        selected = None
        terrain_kwargs = None
        max_init_terrain_level = 4
        terrain_length = 8.0
        terrain_width = 8.0
        num_goals = None
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        slope_treshold = 0.75
        mgdp_parkour_curriculum = None

        class importer:
            terrain_type = "generator"
            mesh_type = "trimesh"
            max_init_terrain_level = 4
            use_terrain_origins = True
            generator = {
                "class_name": "legged_gym.terrains.generators.rough:RoughTerrainGenerator",
                "difficulty_range": [0.0, 1.0],
                "sub_terrains": {
                    "slope": {
                        "class_name": "legged_gym.terrains.generators.rough:PyramidSlopeTerrain",
                        "proportion": 0.1,
                        "terrain_type": 0,
                    },
                    "rough_slope": {
                        "class_name": "legged_gym.terrains.generators.rough:RandomRoughSlopeTerrain",
                        "proportion": 0.1,
                        "terrain_type": 1,
                    },
                    "stairs_down": {
                        "class_name": "legged_gym.terrains.generators.rough:PyramidStairsTerrain",
                        "proportion": 0.35,
                        "inverted": True,
                        "terrain_type": 2,
                    },
                    "stairs_up": {
                        "class_name": "legged_gym.terrains.generators.rough:PyramidStairsTerrain",
                        "proportion": 0.25,
                        "inverted": False,
                        "terrain_type": 3,
                    },
                    "discrete": {
                        "class_name": "legged_gym.terrains.generators.rough:DiscreteObstaclesTerrain",
                        "proportion": 0.2,
                        "terrain_type": 4,
                    },
                },
            }

    class init_state(GO2WSfCfg.init_state):
        pos = [0.0, 0.0, 0.5]


class GO2WSfCfgPPO(GO2WRoughCfgPPO):
    seed = 5
    obs_groups = {
        "actor": ["actor_imu", "actor_command", "actor_motor", "actor_dof_pos", "actor_action"],
        "critic": ["critic_policy", "critic_base_lin_vel", "critic_contact_forces", "critic_heights"],
    }

    actor = {
        "class_name": "ActorModel",
        "backbone": {
            "class_name": "rsl_rl.models:SFTIMActorBackbone",
            "actor_hidden_dims": [512, 256, 128],
            "cenet_hidden_dims": [128, 64],
            "cenet_decoder_hidden_dims": [64, 128],
            "terrain_encoder_hidden_dims": [128, 64],
            "terrain_decoder_hidden_dims": [64, 128],
            "latent_dim": 19,
            "velocity_dim": 3,
            "terrain_latent_dim": 16,
            "terrain_group": "actor_terrain",
            "proprio_groups": ["actor_imu", "actor_command", "actor_motor", "actor_dof_pos", "actor_action"],
            "velocity_target_group": "prev_critic_base_lin_vel",
            "terrain_target_group": "actor_terrain",
            "activation": "elu",
            "aux_loss_coef": 0.25,
            "velocity_loss_coef": 1.0,
            "reconstruction_loss_coef": 1.0,
            "kl_loss_coef": 1.0,
            "terrain_loss_coef": 1.0,
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

    class algorithm(GO2WRoughCfgPPO.algorithm):
        class_name = "PPO"
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        entropy_coef = 0.003
        plugins = []

    class runner(GO2WRoughCfgPPO.runner):
        runner_class_name = "rsl_rl.runners:OnPolicyRunner"
        logger = "wandb"
        wandb_project = "sf_tim_go2w"
        wandb_mode = "online"
        save_interval = 500
        run_name = "sf_tim_all_terrain"
        experiment_name = "sf_tim_go2w"
        num_steps_per_env = 24
        max_iterations = 10000
        load_run = -1
        checkpoint = -1
        resume = False
        resume_path = None


class GO2WSfTrotCfgPPO(GO2WSfCfgPPO):
    class runner(GO2WSfCfgPPO.runner):
        run_name = "sf_tim_stage1_trot"
        experiment_name = "sf_tim_go2w"
        max_iterations = 10000
