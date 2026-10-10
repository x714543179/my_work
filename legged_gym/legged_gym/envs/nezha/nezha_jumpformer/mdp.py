"""Clean training-only state for short-horizon Nezha prediction."""

import torch
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs.nezha.mdp import observations


def prediction_state(env):
    """41-D [gravity, joint offset, joint velocity, linear/angular velocity].

    Motor order and scaling match nezha_jump. Continuous wheel angles are
    zeroed, as in its actor/critic; wheel velocities remain observable.
    This is the CURRENT clean state. The rollout plugin builds future labels.
    """
    return torch.cat((
        quat_rotate_inverse(env.base_quat, env.gravity_vec),
        observations.critic_dof_pos_error(env),
        observations.critic_dof_vel(env),
        observations.critic_base_lin_vel(env),
        observations.critic_base_ang_vel(env),
    ), dim=-1)
