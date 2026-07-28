# Go1 OmniNet Jump

Manager-based Go1 adaptation of *OmniNet: Omnidirectional Jumping Neural
Network With Height-Awareness for Quadrupedal Robots* and its public
[`arclab-hku/Omni-Jump`](https://github.com/arclab-hku/Omni-Jump) code.

## Training

From `legged_gym/`:

```bash
wandb login
python legged_gym/scripts/train.py \
  --task=go1_omni_jump \
  --num_envs=4096 \
  --group_name=one_shot_jump \
  --headless
```

WandB logging is enabled by default with project `omni_jump_go1`, group
`one_shot_jump`, and an online run. Local checkpoints are stored under
`logs/omni_jump_go1_one_shot/`.

Run a checkpoint with:

```bash
python legged_gym/scripts/play.py --task=go1_omni_jump --num_envs=8
```

## Paper Alignment

- The policy receives the paper's 46-D proprioceptive observation.
- A 20-frame, 920-D history is encoded into the paper's 10-D target:
  CoM `[z, x, y]`, four foot heights, and body-frame linear velocity.
- Actor and critic inputs are 56-D and 243-D respectively.
- The critic adds the paper's 187-point egocentric height map.
- The paper rewards are retained, while height, planar, and yaw tracking are
  gated to the active flight. Dense vertical-velocity and flight rewards plus
  a no-takeoff timeout penalty follow the local `Legged_Jump/go2w_jump`
  training strategy.
- Commands use continuous jump heights in `[0.50, 0.68] m`; `vx`, `vy`, and
  yaw rate jointly define forward, lateral, diagonal, and turning jumps.
- Each four-second episode contains one jump. Before a randomized 0.5--1.0 s
  trigger, the visible command is `[0, 0, 0, 0.34]`; at the trigger it switches
  to the sampled velocity, yaw, and height target. The command and jump rewards
  switch off after landing or a 1.5 s no-takeoff timeout.
- As in the local Nezha task, only base contact terminates an episode. Hip,
  thigh, and calf contacts remain penalized without causing immediate resets.
  Initial hip positions receive independent `[-0.03, 0.03] rad` noise and
  thigh/calf positions receive `[-0.1, 0.1] rad` noise, clipped to the URDF
  joint limits. The base resets upright with each linear and angular velocity
  component sampled independently from `[-0.1, 0.1]`.
- Friction, center of mass, motor strength, payload, and a `[0, 4] ms` system
  delay are randomized. The sub-step delay is approximated by interpolating
  the previous and current action over the first 5 ms simulation step.

This task trains the paper's jumping policy only. The paper's walking demo
trains a separate trotting policy and selects between the two policies in the
deployment control loop; it is not a single mixed walking/jumping policy.

## Go1 Adaptations

The paper evaluates Go2 and Aliengo, so two Go1-specific values are not
published. This task uses the repository Go1 URDF, reduces payload
randomization from `[-5, 5] kg` to `[-2, 2] kg` to keep the 4.8 kg Go1 trunk
mass positive, and uses a pre-landing IK pose interpolated between the public
Go1 aerial target and its nominal ground pose. These adaptations are explicit
in `go1_omni_jump_config.py`.

The old five-frame checkpoints are not shape-compatible with this paper
configuration. The earlier continuously commanded checkpoint is behaviorally
incompatible with the one-shot task as well, so train a new run rather than
resuming it.
