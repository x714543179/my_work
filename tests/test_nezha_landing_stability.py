"""CPU regressions for landing stability and touchdown reward settlement."""

import math
import unittest
from types import SimpleNamespace

import isaacgym  # Must precede torch in Isaac Gym installations.
import torch

from legged_gym.envs.nezha.nezha_jump import NezhaJump, NezhaJumpCfg
from legged_gym.envs.nezha.nezha_jumpformer import NezhaJumpFormerCfg
from legged_gym.managers import RewardManager


def landing_env(num_envs=1, cfg_type=NezhaJumpFormerCfg):
    cfg = cfg_type()
    dt = cfg.sim.dt * cfg.control.decimation
    return SimpleNamespace(
        cfg=cfg,
        dt=dt,
        num_envs=num_envs,
        device="cpu",
        has_jumped=torch.ones(num_envs, dtype=torch.bool),
        just_landed=torch.ones(num_envs, dtype=torch.bool),
        jump_succeeded=torch.zeros(num_envs, dtype=torch.bool),
        jump_success_this_step=torch.zeros(num_envs, dtype=torch.bool),
        contact_filt=torch.ones(num_envs, 4, dtype=torch.bool),
        base_euler_xyz=torch.zeros(num_envs, 3),
        base_ang_vel=torch.zeros(num_envs, 3),
        landing_pitch=torch.zeros(num_envs),
        landing_pitch_rate=torch.zeros(num_envs),
        max_height=torch.full((num_envs,), 0.75),
        landing_targets=torch.tensor([[0.04, -0.03]]).repeat(num_envs, 1),
        landing_poses=torch.zeros(num_envs, 2),
        landing_stability_steps=torch.zeros(num_envs, dtype=torch.long),
        landing_stability_steps_required=math.ceil(
            cfg.jump_metrics.success_stability_duration_s / dt
        ),
        root_states=torch.zeros(num_envs, 13),
        rew_buf=torch.zeros(num_envs),
        extras={},
    )


class LandingStability(unittest.TestCase):
    def test_roll_and_roll_rate_reject_both_directions_and_allow_limits(self):
        env = landing_env(7)
        env.base_euler_xyz[:, 0] = torch.tensor(
            [0.0, 0.21, -0.21, 0.0, 0.0, 0.20, -0.20]
        )
        env.base_ang_vel[:, 0] = torch.tensor(
            [0.0, 0.0, 0.0, 1.01, -1.01, 1.0, -1.0]
        )
        for _ in range(env.landing_stability_steps_required):
            NezhaJump._update_landing_stability(env)
        self.assertEqual(
            env.jump_succeeded.tolist(),
            [True, False, False, False, False, True, True],
        )

    def test_roll_instability_restarts_the_continuous_stability_window(self):
        for field, value in (("base_euler_xyz", -0.21), ("base_ang_vel", 1.01)):
            with self.subTest(field=field):
                env = landing_env()
                for _ in range(env.landing_stability_steps_required - 1):
                    NezhaJump._update_landing_stability(env)
                getattr(env, field)[0, 0] = value
                NezhaJump._update_landing_stability(env)
                self.assertEqual(env.landing_stability_steps.item(), 0)
                self.assertFalse(env.jump_succeeded.item())
                getattr(env, field)[0, 0] = 0.0
                for _ in range(env.landing_stability_steps_required - 1):
                    NezhaJump._update_landing_stability(env)
                    self.assertFalse(env.jump_success_this_step.item())
                NezhaJump._update_landing_stability(env)
                self.assertTrue(env.jump_success_this_step.item())

    def test_touchdown_reward_is_immediate_and_success_still_waits_for_stability(self):
        for cfg_type in (NezhaJumpCfg, NezhaJumpFormerCfg):
            with self.subTest(task=cfg_type.__name__):
                env = landing_env(cfg_type=cfg_type)
                self.assertEqual(env.cfg.jump_metrics.success_stability_duration_s, 0.15)
                self.assertEqual(env.landing_stability_steps_required, 8)
                manager = RewardManager(env, {
                    "land_pos": env.cfg.rewards_manager.land_pos,
                    "jump_success": env.cfg.rewards_manager.jump_success,
                })
                expected_land_pos = 30.0 * math.exp(-0.07)
                NezhaJump._update_landing_stability(env)
                manager.compute()
                self.assertAlmostEqual(env.rew_buf.item(), expected_land_pos, places=4)
                self.assertFalse(env.jump_success_this_step.item())
                env.just_landed.zero_()
                for _ in range(6):
                    NezhaJump._update_landing_stability(env)
                    manager.compute()
                    self.assertEqual(env.rew_buf.item(), 0.0)
                NezhaJump._update_landing_stability(env)
                manager.compute()
                self.assertEqual(env.rew_buf.item(), 20.0)
                for _ in range(10):
                    NezhaJump._update_landing_stability(env)
                    manager.compute()
                    self.assertEqual(env.rew_buf.item(), 0.0)
                self.assertAlmostEqual(
                    manager.episode_sums["land_pos"].item(), expected_land_pos, places=4
                )
                self.assertEqual(manager.episode_sums["jump_success"].item(), 20.0)

    def test_landing_accuracy_is_continuous_beyond_ten_centimeters(self):
        env = landing_env(3)
        env.landing_targets = torch.tensor([[0.05, 0.0], [0.15, 0.0], [0.30, 0.0]])
        manager = RewardManager(env, {
            "land_pos": env.cfg.rewards_manager.land_pos,
            "jump_success": env.cfg.rewards_manager.jump_success,
        })
        NezhaJump._update_landing_stability(env)
        manager.compute()
        torch.testing.assert_close(
            env.rew_buf, 30.0 * torch.exp(-torch.tensor([0.05, 0.15, 0.30]))
        )
        env.just_landed.zero_()
        for _ in range(7):
            NezhaJump._update_landing_stability(env)
        manager.compute()
        torch.testing.assert_close(env.rew_buf, torch.tensor([20.0, 0.0, 0.0]))

    def test_touchdown_still_requires_minimum_height_but_not_stable_orientation(self):
        env = landing_env(2)
        env.max_height[0] = 0.60
        env.base_euler_xyz[1, 0] = 0.30
        manager = RewardManager(env, {
            "land_pos": env.cfg.rewards_manager.land_pos,
            "jump_success": env.cfg.rewards_manager.jump_success,
        })
        NezhaJump._update_landing_stability(env)
        manager.compute()
        torch.testing.assert_close(
            env.rew_buf, torch.tensor([0.0, 30.0 * math.exp(-0.07)])
        )
        self.assertFalse(env.jump_succeeded.any().item())


if __name__ == "__main__":
    unittest.main()
