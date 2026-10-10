"""Regressions for positive flight rewards after non-foot ground contact."""

import unittest
from types import SimpleNamespace

import isaacgym  # Must precede torch in Isaac Gym installations.
import torch

from legged_gym.envs.nezha import mdp
from legged_gym.envs.nezha.nezha_jump import NezhaJump


def flight_env(num_envs=1):
    root_states = torch.zeros(num_envs, 13)
    root_states[:, 2] = 0.75
    root_states[:, 7] = 1.0 / 0.55
    return SimpleNamespace(
        num_envs=num_envs,
        dt=0.02,
        was_in_flight=torch.ones(num_envs, dtype=torch.bool),
        has_jumped=torch.zeros(num_envs, dtype=torch.bool),
        flight_nonfoot_contact=torch.zeros(num_envs, dtype=torch.bool),
        jump_signal_issued=torch.ones(num_envs, dtype=torch.bool),
        contact_forces=torch.zeros(num_envs, 8, 3),
        non_foot_indices=torch.tensor([4, 5, 6, 7]),
        penalised_contact_indices=torch.tensor([4, 5, 6, 7]),
        base_euler_xyz=torch.zeros(num_envs, 3),
        base_ang_vel=torch.zeros(num_envs, 3),
        commands=torch.zeros(num_envs, 5),
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        jump_signal_step=torch.zeros(num_envs, dtype=torch.long),
        jump_start_yaw=torch.zeros(num_envs),
        root_states=root_states,
        landing_targets=torch.tensor([[1.0, 0.0]]).repeat(num_envs, 1),
        jump_origins=torch.zeros(num_envs, 2),
        dof_vel=torch.ones(num_envs, 16),
        dof_pos=torch.ones(num_envs, 16),
        default_dof_pos=torch.zeros(1, 16),
        wheel_indices=torch.tensor([3, 7, 11, 15]),
    )


class FlightRewards(unittest.TestCase):
    def test_nonfoot_impacts_disable_all_positive_flight_terms_per_environment(self):
        env = flight_env(6)
        env.contact_forces[0, 4, 2] = 10.0  # Calf support.
        env.contact_forces[1, 5, 0] = 2.0  # Also detect horizontal trunk contact.
        env.contact_forces[2, 6, 1] = 2.0  # Other non-foot links are covered.
        env.contact_forces[3, 0, 2] = 100.0  # Foot contact is not an invalid impact.
        env.contact_forces[4, 7, 2] = 0.5  # Below the contact threshold.
        env.contact_forces[5, 4, 2] = 10.0
        env.was_in_flight[5] = False  # Pre-takeoff contact does not poison flight.
        NezhaJump._update_flight_contact(env)
        self.assertEqual(env.flight_nonfoot_contact.tolist(), [True, True, True, False, False, False])
        flight_mask = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        torch.testing.assert_close(mdp.flight(env), flight_mask)
        torch.testing.assert_close(mdp.base_height_flight(env), 6.0 * flight_mask)
        torch.testing.assert_close(mdp.tracking_lin_vel(env), 5.0 * flight_mask)
        torch.testing.assert_close(
            mdp.jump_yaw_tracking(env), torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        )
        # All feet can be airborne again after a bounce; rewards must stay off.
        env.contact_forces.zero_()
        NezhaJump._update_flight_contact(env)
        torch.testing.assert_close(mdp.tracking_lin_vel(env), 5.0 * flight_mask)
        env.was_in_flight[5] = True
        NezhaJump._update_flight_contact(env)
        self.assertEqual(mdp.flight(env)[5].item(), 1.0)

    def test_airborne_penalties_remain_active_after_invalid_contact(self):
        env = flight_env()
        env.contact_forces[0, 4, 2] = 10.0
        penalties = (mdp.hip_tuck_flight, mdp.dof_vel_flight, mdp.dof_pos_flight)
        before = [term(env).clone() for term in penalties]
        NezhaJump._update_flight_contact(env)
        for term, expected in zip(penalties, before):
            torch.testing.assert_close(term(env), expected)
            self.assertGreater(term(env).item(), 0.0)
        self.assertGreater(mdp.collision(env).item(), 0.0)
        self.assertEqual(mdp.flight(env).item(), 0.0)

    def test_normal_touchdown_ends_flight_rewards(self):
        env = flight_env()
        env.has_jumped[:] = True
        NezhaJump._update_flight_contact(env)
        self.assertFalse(env.flight_nonfoot_contact.item())
        for term in (mdp.flight, mdp.base_height_flight, mdp.tracking_lin_vel, mdp.jump_yaw_tracking):
            self.assertEqual(term(env).item(), 0.0)


if __name__ == '__main__':
    unittest.main()
