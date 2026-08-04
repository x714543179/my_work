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
The local `v_z` ablation uses `post_landing_vertical_velocity=-1.0`: world-frame
vertical motion is penalized quadratically below `0.5 m/s` and linearly above
that threshold after first landing. It is a soft reward only and is not part of
the one-step jump-success conditions.
During takeoff and flight, heading and yaw rate follow the heading trajectory
rather than turning the body toward the landing target. Lateral distance and
cross-track errors are logged as diagnostics but do not have a dedicated
landing reward.

Hip joints use a `-0.5` default-position weight. The `hip_splay_takeoff=-4.0`
term penalizes only left/right symmetric splay beyond `0.20 rad` before flight,
without penalizing collective lateral hip motion. During flight,
`hip_tuck_flight=-4.0` pulls all four hips back toward their default positions.
This preserves lateral push-off authority while discouraging takeoff splay and
an untucked flight posture separately.

The `line_z` reward is active for only 0.4 seconds after the actual jump signal
and stops as soon as flight begins. The jump signal returns to zero on the
first landing. `land_pos=30.0` is a smooth landing-accuracy reward issued only
on that first landing. `jump_success=20.0` is issued once after all four wheels
remain in contact for about 0.15 s while maximum height is at least 0.65 m,
landing error is at most 0.10 m, absolute pitch is at most 0.20 rad, and
absolute pitch rate is at most 1.0 rad/s. Both one-step rewards use
`use_dt=False`, so they are not attenuated by the control timestep. TensorBoard
records configured weights under
`RewardWeight/*`, their effective scales under `RewardScale/*`, and these
commanded-jump episode metrics:

- `Episode/jump_success_rate`: flight followed by the stable landing conditions
  used by the one-step `jump_success` reward.
- `Episode/max_height`: maximum base height.
- `Episode/landing_error`: planar distance from the target; a failed landing is
  measured from the signal-time jump origin.
- `Episode/rear_calf_takeoff_contact_rate`: fraction of commanded jumps whose
  left or right rear calf contacts the ground after the jump signal and before
  first takeoff.
- `Episode/landing_pitch` and `Episode/landing_pitch_rate`: maximum absolute
  pitch and body-frame pitch rate after first landing, measured only over
  episodes that landed.
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
- `Episode/lateral_takeoff_max_hip_deviation` and
  `Episode/lateral_flight_max_hip_deviation`: maximum hip deviation from the
  default pose in the takeoff and flight phases of lateral jumps. The flight
  metric includes only episodes that actually entered flight.
- `Episode/lateral_takeoff_splay` and `Episode/lateral_flight_splay`: maximum
  magnitude of the left/right splay mode in each phase.
- `Episode/lateral_{near,mid,far}_jump_success_rate`,
  `Episode/lateral_{near,mid,far}_landing_error`, and
  `Episode/lateral_{near,mid,far}_max_hip_deviation`: lateral performance split
  into `[0.45, 0.70) m`, `[0.70, 0.95) m`, and `[0.95, 1.20] m` command bins.

WandB logging defaults to project `nezha_spring_jump` in online mode.

## Distance curriculum

Training starts with forward targets in `[0.8, 1.10] m` and lateral targets in
`[0.45, 0.70] m`. Forward targets expand through maximum distances `1.35`,
`1.60`, and `1.80 m`; lateral targets expand through `0.80`, `0.95`, `1.075`,
and `1.20 m`. The lower distance bound remains fixed so every level retains
easy commands.

Each level is held for at least 30000 environment steps, approximately 1250 PPO
iterations with 24 steps per rollout. Advancement uses success within the
farthest 35% of the active range, requiring 70% for forward jumps and 55% for
lateral jumps. A non-empty reset batch is sufficient for evaluation; the sample
count remains logged as a diagnostic. TensorBoard records
`Episode/{forward,lateral}_curriculum_level`,
`Episode/{forward,lateral}_curriculum_frontier_success`,
`Episode/{forward,lateral}_curriculum_frontier_samples`, and
`Episode/{forward,lateral}_curriculum_max_distance`. Play disables the
curriculum and samples the full configured distance ranges.
