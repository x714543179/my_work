# Nezha Spring Jump

This manager-based task is based on
`/home/lt/Code/My_unitree_go2_gym/legged_gym/envs/NEZHA_Jump/NEZHA_Spring_Jump`.
`NezhaJump` derives directly from `ManagerBasedTask` and owns its robot asset,
state-buffer, and control setup without inheriting the Go2W environment.
It extends the reference task with mixed stationary/running commands and
two-dimensional forward/lateral landing targets.
It uses the `resources/robots/nezha/urdf/nezha.urdf` asset, body-frame
landing targets, approach-velocity commands, a timed jump trigger, and
flight/landing state tracking.

Each episode lasts 5 seconds. The five-dimensional command is
`[target_dx_body, target_dy_body, approach_forward_velocity, approach_yaw_rate,
jump_signal]`. Half of the samples command a zero approach velocity for a
stationary jump; the other half command `0.35-0.8 m/s` forward motion for a
running jump, with a yaw-rate command in `[-0.3, 0.3] rad/s`. Forty percent of
targets are sampled around the left or right lateral direction and the rest
are forward or diagonal. Forward/diagonal target distance is `0.80-1.80 m`;
lateral target distance is `0.45-1.20 m`. Lateral ground
velocity is treated as wheel slip rather than commanded, so sideways motion is
produced by the takeoff impulse instead of impossible sideways wheel rolling.

In 90% of episodes, a jump becomes eligible at policy step 60-89. Running
jumps wait until the approach-velocity error is at most `0.2 m/s`, with a
bounded extra delay of `0.6 s`. On the rising edge, the body-frame target is
converted into a fixed world-frame landing target relative to the robot's
current position. The signal remains enabled through the first landing.
`jump_assist` is disabled by default.

The actor receives ten stacked 59-D frames (590 values). The critic receives
three stacked 83-D privileged frames (249 values). The five-dimensional
command remains the only command-related input; jump-heading error is used for
rewarding and diagnostics but is not added to either observation. Both
networks use the source three-layer MLP sizes and PPO hyperparameters.

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task nezha_jump \
  --headless
```

Replay the latest checkpoint with one environment:

```bash
python legged_gym/legged_gym/scripts/play_nezha_jump.py
```

Select a specific run and checkpoint when needed:

```bash
python legged_gym/legged_gym/scripts/play_nezha_jump.py \
  --load_run Jul24_11-09-31_nezha_spring_jump_manager \
  --checkpoint 7500
```

Playback forces a jump command in every episode and disables the training-time
vertical velocity assist. The viewer marks the start in green and the target in
yellow; landing error and maximum height are printed after each landing.

Before the signal, the policy tracks the commanded approach velocity.
Stationary samples additionally penalize displacement from the reset position
and wheel speed; running samples are free to roll. During flight, horizontal
velocity is tracked in both world x and y from the two-dimensional landing
target and an expected `0.55 s` flight time. After landing, stationary samples
track zero velocity while running samples resume their approach velocity.
During takeoff and flight, heading and yaw rate follow the heading trajectory
rather than turning the body toward the landing target. Lateral distance and
cross-track errors are logged as diagnostics but do not have a dedicated
landing reward.

Hip joints use a `-0.5` default-position weight. A
separate jump-phase penalty permits up to `0.20 rad` hip motion and penalizes
only the excess with weight `-4.0`, and doubles that penalty in flight. This
leaves useful lateral push-off authority while discouraging large symmetric
leg splay on long jumps.

The `line_z` reward is active for only 0.4 seconds after the actual jump signal
and stops as soon as flight begins. The jump signal returns to zero on the
first landing. TensorBoard records configured weights under `RewardWeight/*`,
their per-step values after multiplying by `env.dt` under `RewardScale/*`, and
these commanded-jump episode metrics:

- `Episode/jump_success_rate`: flight and landing with height at least 0.65 m
  and landing error at most 0.30 m.
- `Episode/max_height`: maximum base height.
- `Episode/landing_error`: planar distance from the target; a failed landing is
  measured from the signal-time jump origin.
- `Episode/stationary_jump_success_rate` and
  `Episode/running_jump_success_rate`: success split by approach mode.
- `Episode/lateral_jump_success_rate`: success for left/right lateral targets.
- `Episode/stationary_pre_jump_displacement`: reset-to-signal displacement for
  stationary commands.
- `Episode/pre_jump_velocity_error` and
  `Episode/running_pre_jump_velocity_error`: approach-command tracking error.
- `Episode/takeoff_velocity`: horizontal velocity at first flight detection.
- `Episode/jump_heading_error`: mean absolute heading-trajectory error during
  the jump phase.
- `Episode/lateral_distance_error` and
  `Episode/lateral_cross_track_error`: lateral landing error resolved along
  and perpendicular to the commanded jump direction.
- `Episode/max_hip_deviation` and
  `Episode/lateral_max_hip_deviation`: largest absolute hip angle during the
  jump phase.

WandB logging defaults to project `nezha_spring_jump` in online mode.
