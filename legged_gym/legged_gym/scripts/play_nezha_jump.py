"""Play a trained Nezha landing-target jump policy."""

import os

import isaacgym  # noqa: F401
import torch
from isaacgym import gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.nezha import mdp
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import disable_manager_randomization, get_load_path


TASK_NAME = "nezha_jump"
PLAY_EPISODES = 10


def _configure_play_env(env_cfg, args):
    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False

    # Evaluate the learned jump instead of the training-time vertical velocity assist.
    env_cfg.jump_assist.enabled = False
    env_cfg.commands.jump_probability = 1.0
    env_cfg.commands.resample.params["jump_probability"] = 1.0

    disable_manager_randomization(env_cfg)
    env_cfg.events.rigid_body_props.enabled = False

    env_cfg.viewer.pos = [2.5, -4.0, 1.8]
    env_cfg.viewer.lookat = [0.8, 0.0, 0.5]


class JumpTargetVisualizer:
    def __init__(self, env):
        self.env = env
        self.env_id = torch.zeros(1, dtype=torch.long, device=env.device)
        self.start_marker = gymutil.WireframeSphereGeometry(
            0.035, 8, 8, None, color=(0.0, 1.0, 0.0)
        )
        self.target_marker = gymutil.WireframeSphereGeometry(
            0.055, 8, 8, None, color=(1.0, 1.0, 0.0)
        )

    def draw(self):
        env = self.env
        if env.viewer is None:
            return

        env.gym.clear_lines(env.viewer)
        if bool(env.jump_signal_issued[0].item()):
            start_xy_tensor = env.jump_origins[0]
            target_xy_tensor = env.landing_targets[0]
        else:
            start_xy_tensor = env.root_states[0, :2]
            target_xy_tensor = (
                start_xy_tensor
                + mdp.body_target_to_world(env, self.env_id)[0]
            )
        start_xy = start_xy_tensor.detach().cpu().numpy()
        target_xy = target_xy_tensor.detach().cpu().numpy()

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(float(start_xy[0]), float(start_xy[1]), 0.035)
        target_pose = gymapi.Transform()
        target_pose.p = gymapi.Vec3(float(target_xy[0]), float(target_xy[1]), 0.055)

        env_handle = env.envs[0]
        gymutil.draw_lines(
            self.start_marker, env.gym, env.viewer, env_handle, start_pose
        )
        gymutil.draw_lines(
            self.target_marker, env.gym, env.viewer, env_handle, target_pose
        )


def _target_command(env):
    return env.commands[0, :2].detach().cpu().numpy()


def _print_episode_start(env, episode):
    command = _target_command(env)
    trigger_time = int(env.command_frame[0].item()) * env.dt
    approach_forward = env.commands[0, mdp.APPROACH_VX_INDEX].item()
    approach_yaw_rate = env.commands[
        0, mdp.APPROACH_YAW_RATE_INDEX
    ].item()
    jump_mode = (
        "stationary" if bool(env.stationary_jump_command[0].item()) else "running"
    )
    target_mode = (
        "lateral" if bool(env.lateral_jump_command[0].item()) else "forward"
    )
    print(
        f"[episode {episode:02d}] mode={jump_mode}/{target_mode}, "
        f"body target=({command[0]:+.3f}, {command[1]:+.3f}) m, "
        f"approach_vx={approach_forward:+.3f} m/s, "
        f"approach_yaw_rate={approach_yaw_rate:+.3f} rad/s, "
        f"jump eligible at {trigger_time:.2f} s"
    )


def _landing_metrics(env):
    target_xy = env.landing_targets[0]
    landing_xy = env.landing_poses[0]
    error = torch.linalg.norm(target_xy - landing_xy).item()
    max_height = env.max_height[0].item()
    landing = landing_xy.detach().cpu().numpy()
    return landing, error, max_height


def play(args):
    if args.task not in ("a1", TASK_NAME):
        raise ValueError(f"This script only supports --task={TASK_NAME!s}.")
    args.task = TASK_NAME

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    _configure_play_env(env_cfg, args)

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)

    # Load explicitly so CUDA checkpoints can also be replayed on CPU.
    args.resume = False
    train_cfg.runner.resume = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=TASK_NAME,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    checkpoint_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
    )
    checkpoint_path = get_load_path(
        checkpoint_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint,
    )
    print(f"Loading model from: {checkpoint_path}")
    ppo_runner.load(checkpoint_path, map_location=env.device)
    policy = ppo_runner.get_inference_policy(device=env.device)
    obs = ppo_runner.env.get_observations().to(env.device)

    visualizer = JumpTargetVisualizer(env)
    completed_episodes = 0
    episode = 1
    jump_signal_active = False
    landing_reported = False
    _print_episode_start(env, episode)

    with torch.inference_mode():
        while completed_episodes < PLAY_EPISODES:
            actions = policy(obs)["actions"]
            obs, _, dones, _ = ppo_runner.env.step(actions)
            obs = obs.to(env.device)
            visualizer.draw()

            jump_signal = bool(
                env.commands[0, mdp.JUMP_SIGNAL_INDEX].item() > 0.5
            )
            if jump_signal and not jump_signal_active:
                target = env.landing_targets[0].detach().cpu().numpy()
                print(
                    f"[episode {episode:02d}] jump signal active, "
                    f"world target=({target[0]:+.3f}, {target[1]:+.3f}) m"
                )
            jump_signal_active = jump_signal

            has_landed = bool(env.has_jumped[0].item())
            if has_landed and not landing_reported:
                landing, error, max_height = _landing_metrics(env)
                print(
                    f"[episode {episode:02d}] landed=({landing[0]:+.3f}, "
                    f"{landing[1]:+.3f}) m, error={error:.3f} m, "
                    f"max_height={max_height:.3f} m"
                )
                landing_reported = True

            if bool(dones[0].item()):
                completed_episodes += 1
                if completed_episodes >= PLAY_EPISODES:
                    break
                episode += 1
                jump_signal_active = False
                landing_reported = False
                _print_episode_start(env, episode)


if __name__ == "__main__":
    play(get_args())
