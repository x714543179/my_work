"""Nezha jumping with ParkourFormer-inspired future-conditioned actions."""

from copy import deepcopy
from legged_gym.managers import ManagerTermCfg, ObsGroup
from legged_gym.envs.nezha.nezha_jump.nezha_jump_config import NezhaJumpCfg, NezhaJumpCfgPPO
from . import mdp


class NezhaJumpFormerCfg(NezhaJumpCfg):
    task_name = "NEZHA_JumpFormer"

    class env(NezhaJumpCfg.env):
        # Smaller rollout batches for the attention backbone on a 16 GB GPU.
        num_envs = 1024

    class observations(NezhaJumpCfg.observations):
        class future_state(ObsGroup):
            # Supervision only: not included in the actor or critic input sets.
            state = ManagerTermCfg(func=mdp.prediction_state, env_arg=True)


class NezhaJumpFormerCfgPPO(NezhaJumpCfgPPO):
    obs_groups = deepcopy(NezhaJumpCfgPPO.obs_groups)
    actor = deepcopy(NezhaJumpCfgPPO.actor)
    actor["backbone"] = {
        "class_name": "rsl_rl.models.future_attention_model:FutureAttentionModel",
        "frame_dim": 59,
        "history_length": 10,
        "future_state_dim": 41,
        "future_steps": 2,
        "token_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        # FP32 short-sequence path; faster than auto SDPA for Q=2/K=10
        # on the measured RTX 4090 Laptop / PyTorch 2.4.1 workload.
        # Switch to "auto" to let PyTorch select a kernel on another setup.
        "attention_backend": "math",
        "ffn_dim": 256,
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
    }
    critic = deepcopy(NezhaJumpCfgPPO.critic)

    class algorithm(NezhaJumpCfgPPO.algorithm):
        plugins = [
            {
                "class_name": "rsl_rl.algorithms.plugins.future_prediction_plugin:FuturePredictionPlugin",
                "state_group": "future_state",
                "state_layout": "nezha-gravity-qoffset-dq-vlin-vang-v1",
                "loss_coef": 1.0,
                # Paper specifies higher weight for negative advantage; 2.0
                # is our exposed adaptation, not a reported paper constant.
                "negative_advantage_weight": 2.0,
            },
        ] + deepcopy(NezhaJumpCfgPPO.algorithm.plugins)

    class runner(NezhaJumpCfgPPO.runner):
        experiment_name = "nezha_jumpformer"
        run_name = "nezha_jumpformer"
        wandb_project = "nezha_jumpformer"
        wandb_group = "future_prediction"
        wandb_tags = ["nezha", "jumpformer", "manager"]
