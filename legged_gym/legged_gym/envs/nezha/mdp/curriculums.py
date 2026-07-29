"""Distance curricula for forward and lateral Nezha jumps."""

import torch


def _apply_distance_level(env, command_type):
    cfg = env.cfg.commands.distance_curriculum
    levels = getattr(cfg, f"{command_type}_levels")
    level = getattr(env, f"{command_type}_curriculum_level")
    command_range = list(levels[level])
    env.command_ranges[f"{command_type}_target_distance"] = command_range


def initialize_jump_distance_curriculum(env):
    """Initialize independent forward and lateral command ranges."""
    cfg = env.cfg.commands.distance_curriculum
    for command_type in ("forward", "lateral"):
        levels = getattr(cfg, f"{command_type}_levels")
        initial_level = int(getattr(cfg, f"initial_{command_type}_level"))
        initial_level = min(max(initial_level, 0), len(levels) - 1)
        setattr(env, f"{command_type}_curriculum_level", initial_level)
        setattr(env, f"{command_type}_curriculum_last_change_step", 0)
        setattr(env, f"{command_type}_curriculum_frontier_success", 0.0)
        setattr(env, f"{command_type}_curriculum_frontier_samples", 0)
        _apply_distance_level(env, command_type)
    env._jump_curriculum_last_eval_step = -1


def _update_distance_level(env, env_ids, command_type):
    cfg = env.cfg.commands.distance_curriculum
    levels = getattr(cfg, f"{command_type}_levels")
    level_attr = f"{command_type}_curriculum_level"
    level = getattr(env, level_attr)
    current_range = levels[level]

    valid = (
        (env.episode_length_buf[env_ids] > 0)
        & env.jump_signal_issued[env_ids]
    )
    lateral = env.lateral_jump_command[env_ids]
    type_mask = lateral if command_type == "lateral" else ~lateral
    target_distance = torch.linalg.norm(
        env.landing_targets[env_ids] - env.jump_origins[env_ids], dim=1
    )
    frontier_start = current_range[1] - cfg.frontier_fraction * (
        current_range[1] - current_range[0]
    )
    frontier = valid & type_mask & (target_distance >= frontier_start)
    sample_count = int(torch.sum(frontier).item())
    setattr(
        env,
        f"{command_type}_curriculum_frontier_samples",
        sample_count,
    )
    if sample_count < cfg.min_samples:
        return

    landing_error = torch.linalg.norm(
        env.landing_targets[env_ids] - env.landing_poses[env_ids], dim=1
    )
    successful = (
        env.has_jumped[env_ids]
        & (env.max_height[env_ids] >= env.cfg.jump_metrics.success_min_height)
        & (
            landing_error
            <= env.cfg.jump_metrics.success_max_landing_error
        )
    )
    success_rate = float(successful[frontier].float().mean().item())
    setattr(
        env,
        f"{command_type}_curriculum_frontier_success",
        success_rate,
    )

    current_step = int(env.common_step_counter)
    last_change_step = getattr(
        env, f"{command_type}_curriculum_last_change_step"
    )
    threshold = getattr(cfg, f"{command_type}_success_threshold")
    ready_to_advance = (
        current_step - last_change_step >= cfg.min_level_duration_steps
    )
    if (
        level < len(levels) - 1
        and ready_to_advance
        and success_rate >= threshold
    ):
        setattr(env, level_attr, level + 1)
        setattr(
            env,
            f"{command_type}_curriculum_last_change_step",
            current_step,
        )
        _apply_distance_level(env, command_type)


def jump_distance_levels(env, env_ids):
    """Advance each distance range from frontier-bin jump success."""
    if len(env_ids) == 0 or not env.cfg.commands.curriculum:
        return
    current_step = int(env.common_step_counter)
    if env._jump_curriculum_last_eval_step == current_step:
        return
    env._jump_curriculum_last_eval_step = current_step
    _update_distance_level(env, env_ids, "forward")
    _update_distance_level(env, env_ids, "lateral")
