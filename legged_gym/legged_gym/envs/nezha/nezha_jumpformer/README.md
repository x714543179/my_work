# Nezha JumpFormer

任务名：`nezha_jumpformer`。基于当前 `nezha_jump`，采用
《ParkourFormer: Integrating Predictive Supervision and Sequence Modeling into
Parkour Locomotion》（用户提供的 arXiv:2605.25782v3，2026-06-12 版本）
第 3.1–3.3 节的当前状态查询历史、两步状态预测与未来条件动作机制。

这是针对 Nezha 跳跃任务的适配：不包含论文的 RGB-D 编码、视觉条件 FFN、
AMP 动作库和判别器，不能把论文中的 G1 parkour 实验结果视为本任务效果。

## 数据流

```text
ObservationManager: 10 × 59-D actor history (oldest -> newest)
    -> frame projection + temporal position encoding
    -> latest frame creates two queries
    -> two cross-attention + SwiGLU blocks over the history
    -> two 41-D future body-state predictions
    -> concat(current frame, temporal features, predicted future states)
    -> action MLP -> Gaussian ActorModel -> 16 actions

ObservationManager: clean current 41-D future_state group
    -> rollout t+1/t+2 labels + reset/timeout/tail validity masks
    -> prediction loss added to PPO by FuturePredictionPlugin
```

- 沿用十帧历史（论文使用八帧），最新帧仍在最后。原始 actor 输入仍是
  `10 × 59 = 590` 维，不需要将真实未来状态塞入环境的 actor observation。
- 预测第 1、2 个策略步后的身体状态，当前 `dt=0.02 s`，即未来 20/40 ms。
- 动作 MLP 的输入为当前 59 维观测、`2 × 128` 维时序特征和
  `2 × 41` 维预测观测，共 **397 维**。预测值真正参与动作输出，
  没有 detach；PPO 与监督损失都能更新预测头及共享注意力骨干。
- critic 保持原任务的 `3 × 83 = 249` 维特权观测和 MLP。
- 环境、动作控制、资产、跳跃命令/课程、奖励、终止和随机化继承
  `NezhaJump`；默认并行环境数为 1024，可由 `--num_envs` 覆盖。

## 预测状态及其监督

每个预测状态为 41 维，顺序固定：

| 切片 | 内容 | 缩放 |
|---|---|---|
| `[0:3]` | 机身坐标系投影重力 | 单位重力方向 |
| `[3:19]` | 16 个关节相对默认姿态的位置偏差 | `obs_scales.dof_pos` |
| `[19:35]` | 16 个关节速度，包含轮速 | `obs_scales.dof_vel` |
| `[35:38]` | 机身坐标系线速度 | `obs_scales.lin_vel` |
| `[38:41]` | 机身坐标系角速度 | `obs_scales.ang_vel` |

位置采用原生 URDF/任务关节顺序。连续轮关节的四个位置维与原任务一致置零，
避免预测没有界限的累计轮角。监督状态无观测噪声和传感延迟。这里的“未来观测”
指未来身体/运动状态，不预测外部命令、未来策略动作或随机噪声。

`future_state` 观测组只输出当前真实状态，且不在 actor/critic 的输入组中。
插件在完整 rollout 后按时间建立未来标签，标签与对应历史一起由原生
RolloutStorage 打乱成 minibatch。推理只需要 actor 历史，无需未来标签、
仿真特权状态或 Critic。

若 t 到 t+k 中任一步发生 reset，包括 timeout，该预测标签无效；不把
下一回合的初态当作上一回合的未来。用 rollout 最后的 next observation
补齐最后一步的 t+1 目标；最后一项 t+2 尚未采集，显式屏蔽。每个 rollout
独立监督，不使用旧策略缓存补齐未知未来。

默认总损失为现有 PPO + 原任务对称损失 + 预测监督：

```text
future_loss = loss_coef * sum(valid * advantage_weight * mean_D((prediction-target)^2))
                           / max(sum(valid), 1)
```

有效目标按状态维度取均方误差。负 advantage 权重默认为 2，其他为 1；
论文只说明负 advantage 权重更大，未给出这里使用的精确常数。
`loss_coef=1.0`、负优势权重 2、4 个 attention heads、2 层、FFN 宽度 256
均为可调适配参数，不声称是论文的完整超参数复现。Attention dropout 为零，
确保 PPO 采样与更新阶段相同输入对应相同策略均值。

## 文件位置

- `nezha_jumpformer_config.py`：任务配置、Actor 架构、预测插件及日志命名。
- `nezha_jumpformer_robot.py`：继承原 Nezha manager 环境。
- `mdp.py`：训练专用干净状态 observation term。
- `rsl_rl/models/future_attention_model.py`：可复用注意力骨干、预测头、动作头和导出。
- `rsl_rl/algorithms/plugins/future_prediction_plugin.py`：PPO 监督插件。

通用框架增加 ObservationManager 的可选额外观测组计算和 adapter 传递。
PPO 在额外损失 hook 期间通过 Batch.actor_extra 暴露原始 Actor 前向结果，
预测插件直接复用带梯度的 predicted_future，避免重复历史编码。hook 结束后
清空临时引用；镜像前向不会覆盖它。原插件 hook 签名和训练流程保持兼容，
ActorModel、Critic 和 OnPolicyRunner 的接口不变。
原 `nezha_jump` 仍注册原环境和 MLP 配置。

## 训练与播放

在 `/home/lt/my_work/legged-manager` 下，使用已有 Isaac Gym 环境：

```bash
conda activate my_work
export PYTHONPATH="$PWD/legged_gym:$PWD/rsl_rl-1.0.2:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python legged_gym/legged_gym/scripts/train.py \
  --task nezha_jumpformer --num_envs 1024 --headless
```

学习率、PPO epochs、rollout 长度和对称损失沿用原任务。当前学习率
`1e-5` 是原任务设定，尚未针对新注意力网络调参。Runner 沿用 WandB logger；
只需本地日志时，在新配置 `runner` 中设置 `logger = "tensorboard"`。
不要与已占满显存的其他训练进程同时启动；4096 环境的短时 GPU 基准见文末，显存峰值会随运行状态变化。

```bash
python legged_gym/legged_gym/scripts/play_nezha_jump.py \
  --task nezha_jumpformer --load_run <run_directory> --checkpoint <iteration>
```

新任务使用单独的 `logs/nezha_jumpformer/`。原 `nezha_jump` MLP checkpoint
不能直接作为该 Transformer 的完整 checkpoint 加载；请从新任务开始训练。
新 checkpoint 保存 Actor（包括预测器）、Critic、PPO optimizer，以及预测状态
布局/预测步数校验信息；加载时检查布局兼容。不保存仿真 episode 轨迹。

部署时 `runner.export_policy_to_jit(...)` 导出只接收 `[N,590]` 历史的策略。
历史缓冲由调用方按旧到新维护，每个 reset 清零；推理调用放在 `torch.inference_mode()` 中。
提供 ONNX 接口，当前验证
以 TorchScript 为准，不声称已完成实际机器人部署。

## 诊断指标

TensorBoard/WandB scalar：

- `Loss/future_prediction`：包含系数与负 advantage 加权的监督损失。
- `Loss/future_mse_step_1`、`Loss/future_mse_step_2`：各预测步的未加权、
  按状态维平均的 MSE（在训练 minibatch 上统计，不是独立验证集误差）。
- `Loss/future_valid_step_1`、`Loss/future_valid_step_2`：有效预测标签比例。
- 原跳跃成功率、落点误差、最大高度等仍保留。

预测 MSE 降低不等于跳跃成功率提高；应同时比较原任务和新任务的任务指标。

## 验证命令

```bash
CUDA_VISIBLE_DEVICES=-1 python -m unittest discover \
  -s tests -p test_nezha_jumpformer.py -v
CUDA_VISIBLE_DEVICES=-1 python tests/nezha_jumpformer_smoke.py
```

CPU 测试覆盖：未来时间对齐、reset/timeout/rollout 尾部屏蔽、真实未来信息
隔离、预测作为 Actor 输入及其梯度、原生历史顺序、局部 reset 历史清空、
小批监督拟合、原任务 MDP 配置保持、PPO/对称插件共存、checkpoint 和 JIT。
仿真 smoke 使用 4 个 CPU 环境、3 个 rollout，每 rollout 24 步，沿用
5 个 PPO epochs / 4 个 minibatches，强制一次 timeout 检查跨回合处理。
这些验证只能说明训练链路可执行，不能证明新任务已收敛或性能优于基线。

### 本次执行结果（2026-09-30）

- 16 项 CPU 回归测试通过，包含注意力后端、前向复用、损失/梯度/更新数值等价和原 MLP 任务兼容性。
- 4 个真实 CPU PhysX 环境完成 3 轮训练，24 步/轮，5 epochs/4 minibatches。
- 预测头参数实际更新；checkpoint 保存/恢复及只输入历史的 JIT 推理通过。
- 仿真报告：`/tmp/nezha_jumpformer_smoke_sep30_v3/report.json`。
- 4096 个 GPU 环境完成 8 轮后端对比；另通过 CUDA 输出/梯度等价检查。未验证长程收敛和实际机器人效果。

### 每轮耗时与前向复用（2026-09-30）

4096 环境 × 24 步 = 98,304 个 transition；4 个 minibatch 使每个批次包含
24,576 段十帧历史，5 epochs 共进行 20 次优化器更新。注意力编码、
预测头和对称正则的镜像 Actor 都需要梯度，计算和激活显存明显多于原 MLP。

预测损失现复用 PPO 原始 Actor 前向的结果（不 detach），每个 minibatch
的历史编码由 3 次减少为 2 次：原始前向和镜像前向各一次。
没有修改网络宽度、层数、学习率、PPO epochs、未来监督或对称损失系数；
模型和优化器 checkpoint 结构不变。CPU 回归核对两种插件顺序下的损失、
梯度、参数更新和动作输出，并检查 batch 不保留额外计算图。FP32 的
梯度求和顺序不同，比较使用浮点容差，不保证 bitwise 一致。

训练时使用 --headless 可关闭可视化。1024 是本任务默认环境数，
--num_envs 4096 会覆盖它；减少环境数也会减少每轮采集的数据量，
不能只用每轮秒数评价训练效率，应同时比较 Perf/total_fps。
磁盘上的代码优化需要重新启动/恢复训练才会生效。前向次数不等于耗时比例，
应以相同设置的实际基准为准，不能由少一次前向直接推算总耗时收益。

### FP32 短序列注意力优化（2026-09-30）

新任务 actor.backbone.attention_backend 默认设为 "math"。对于当前
2 个 query / 10 帧 memory，显式 bmm/softmax 路径在本机比 PyTorch
自动选择的 FP32 memory-efficient attention 更快。实现使用原
nn.MultiheadAttention 参数及公开 forward 接口；设置 need_weights=True
并关闭跨头权重平均以选择该路径。注意力权重没有成为额外训练目标。
改回 "auto" 可恢复 PyTorch 自动后端选择；其他长度、硬件或版本应重新测量。

保留 float32、128 维 token、2 层、4 heads、十帧历史、两步预测、
全部预测梯度、对称损失、24 步 rollout 和 5 epochs / 4 minibatches。
checkpoint 参数名、尺寸和优化器状态布局不变，可以继续加载已有新任务模型。
单一 actor 观测组不再使用 torch.cat 复制历史；梯度开启且输入属于
inference tensor 时仍复制为普通 tensor，保持独立调用的兼容性。

本机 RTX 4090 Laptop / PyTorch 2.4.1，4096 环境，headless，本地
TensorBoard，torch/OMP/MKL 单线程，TF32 关闭。一次连续基准运行各后端
4 轮，分别排除第一轮，以剩余 3 轮取平均：

| 指标 | auto | math |
|---|---:|---:|
| 采样（秒/轮） | 1.191 | 1.140 |
| 更新（秒/轮） | 5.842 | 4.395 |
| 合计（秒/轮） | 7.033 | 5.535 |
| PyTorch 峰值 allocated（GiB） | 6.232 | 5.044 |

整轮用时减少约 21.3%，更新用时减少约 24.8%。显存行仅计 PyTorch
分配的 tensor，**不包括 PhysX、CUDA 上下文等总显存开销**。这是短时
吞吐基准，不是长程训练或跳跃成功率验证；运行负载、温度等会影响绝对耗时。

- 完整记录：/tmp/jumpformer_full_gpu_sep30/report.json
- 各轮计时：/tmp/jumpformer_full_gpu_sep30/iterations.json
- 基准脚本：/tmp/benchmark_jumpformer_full_gpu.py
- CPU 回归日志：/tmp/jumpformer_attention_backend_tests.log
- CUDA 上另验证两种后端的动作、预测、输入和参数梯度在浮点容差内一致。
