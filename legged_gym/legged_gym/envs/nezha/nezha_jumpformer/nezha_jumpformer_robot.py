"""JumpFormer task reuses Nezha's manager-owned physics and jumping MDP."""

from legged_gym.envs.nezha.nezha_jump.nezha_jump_robot import NezhaJump
from .nezha_jumpformer_config import NezhaJumpFormerCfg


class NezhaJumpFormer(NezhaJump):
    """Add clean supervision through ObservationManager, not the action path."""

    cfg: NezhaJumpFormerCfg
