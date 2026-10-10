"""History-query policy with supervised, action-conditioning future states.

Inspired by ParkourFormer (user-provided arXiv:2605.25782v3, Sec. 3.1-3.3).
This proprioceptive adaptation has no RGB-D conditioning or AMP discriminator.
"""

from __future__ import annotations

import copy
import math
import torch
from torch import nn
from rsl_rl.modules import MLP
from rsl_rl.utils import unpad_trajectories


class CrossAttentionBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, ffn_dim: int, attention_backend: str = "auto"):
        super().__init__()
        if attention_backend not in ("auto", "math"):
            raise ValueError("attention_backend must be 'auto' or 'math'")
        self.attention_backend = attention_backend
        # PPO rollout and update must use the same deterministic backbone.
        self.attention = nn.MultiheadAttention(token_dim, num_heads, dropout=0.0, batch_first=True)
        self.attention_norm = nn.LayerNorm(token_dim)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.gate = nn.Linear(token_dim, ffn_dim)
        self.value = nn.Linear(token_dim, ffn_dim)
        self.output = nn.Linear(ffn_dim, token_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # At Q=2, K=10, profiling on RTX 4090 Laptop / torch 2.4.1 found
        # MHA's FP32 bmm/softmax path faster than auto SDPA. Requesting
        # per-head weights selects that path without changing parameters or math.
        # Keep auto selectable for other sequence lengths/devices/torch versions.
        attended, _ = self.attention(
            query, memory, memory, need_weights=self.attention_backend == "math",
            average_attn_weights=False,
        )
        query = self.attention_norm(query + attended)
        x = self.ffn_norm(query)
        return query + self.output(torch.nn.functional.silu(self.gate(x)) * self.value(x))


class FutureAttentionCore(nn.Module):
    """Pure tensor network; only ordered actor history is accessible here."""

    def __init__(self, frame_dim, history_length, future_state_dim, future_steps,
                 output_dim, token_dim, num_heads, num_layers, ffn_dim, hidden_dims, activation,
                 attention_backend="auto"):
        super().__init__()
        if min(frame_dim, history_length, future_state_dim, future_steps, num_layers, num_heads) < 1:
            raise ValueError("History, prediction and attention dimensions must be positive")
        if token_dim % num_heads or token_dim % 2:
            raise ValueError("token_dim must be even and divisible by num_heads")
        self.frame_dim = int(frame_dim)
        self.history_length = int(history_length)
        self.future_state_dim = int(future_state_dim)
        self.future_steps = int(future_steps)
        self.token_dim = int(token_dim)
        self.input_dim = self.frame_dim * self.history_length
        self.token_projection = nn.Linear(frame_dim, token_dim)
        position = torch.arange(history_length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, token_dim, 2) * (-math.log(10000.0) / token_dim))
        encoding = torch.zeros(history_length, token_dim)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("position_encoding", encoding.unsqueeze(0))
        self.query_projection = nn.Linear(token_dim, future_steps * token_dim)
        self.query_norm = nn.LayerNorm(token_dim)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(token_dim, num_heads, ffn_dim, attention_backend) for _ in range(num_layers)
        ])
        self.prediction_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.ELU(),
                                             nn.Linear(token_dim, future_state_dim))
        # Predicted observations are actual inputs to the action MLP, not just
        # an auxiliary output. PPO gradients also flow through this connection.
        self.action_input_dim = frame_dim + future_steps * (token_dim + future_state_dim)
        self.action_head = MLP(self.action_input_dim, output_dim, hidden_dims, activation)

    def encode(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.dim() != 2 or history.size(1) != self.input_dim:
            raise ValueError("Expected flattened oldest-to-newest actor history")
        frames = history.reshape(-1, self.history_length, self.frame_dim)
        tokens = self.token_projection(frames)
        memory = tokens + self.position_encoding
        query = self.query_projection(tokens[:, -1]).reshape(-1, self.future_steps, self.token_dim)
        query = self.query_norm(query)
        for block in self.blocks:
            query = block(query, memory)
        return frames[:, -1], query

    def predict_future(self, history: torch.Tensor) -> torch.Tensor:
        _, features = self.encode(history)
        return self.prediction_head(features)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current, features = self.encode(history)
        prediction = self.prediction_head(features)
        action_input = torch.cat((current, features.flatten(1), prediction.flatten(1)), dim=-1)
        return self.action_head(action_input), prediction


class FutureActorExport(nn.Module):
    """Deterministic deployment policy accepting only proprioceptive history."""

    def __init__(self, core):
        super().__init__()
        self.core = copy.deepcopy(core).eval().requires_grad_(False)
        self.input_dim = core.input_dim
        self.input_names = ["actor_history"]
        self.output_names = ["actions"]

    def forward(self, actor_history: torch.Tensor) -> torch.Tensor:
        actions, _ = self.core(actor_history)
        return actions

    @torch.jit.ignore
    def get_dummy_inputs(self):
        return (torch.zeros(1, self.input_dim),)


class FutureAttentionModel(nn.Module):
    """ActorModel backbone using the existing TensorDict/model interface."""

    is_recurrent = False

    def __init__(self, obs, obs_groups, obs_set, output_dim, frame_dim=59,
                 history_length=10, future_state_dim=41, future_steps=2, token_dim=128,
                 num_heads=4, num_layers=2, ffn_dim=256, hidden_dims=(512, 256, 128),
                 activation="elu", attention_backend="auto"):
        super().__init__()
        self.obs_groups = list(obs_groups[obs_set])
        self.obs_dim = sum(obs[group].shape[-1] for group in self.obs_groups)
        if self.obs_dim != frame_dim * history_length:
            raise ValueError("Actor observation width does not match history_length * frame_dim")
        if any(obs[group].ndim != 2 for group in self.obs_groups):
            raise ValueError("FutureAttentionModel requires flat observation groups")
        self.future_state_dim = int(future_state_dim)
        self.future_steps = int(future_steps)
        self.core = FutureAttentionCore(frame_dim, history_length, future_state_dim, future_steps,
                                        output_dim, token_dim, num_heads, num_layers, ffn_dim,
                                        hidden_dims, activation, attention_backend)

    def _history(self, obs):
        if len(self.obs_groups) == 1:
            history = obs[self.obs_groups[0]]
            # External callers may reuse a rollout inference tensor with grad
            # enabled; autograd must receive a normal tensor in that case.
            if torch.is_grad_enabled() and torch.is_inference(history):
                return history.clone()
            return history
        return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

    def forward(self, obs, masks=None, hidden_state=None, train_mode=False):
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        actions, prediction = self.core(self._history(obs))
        return {"actions": actions, "predicted_future": prediction}

    def predict_future(self, obs):
        return self.core.predict_future(self._history(obs))

    def reset(self, dones=None, hidden_state=None):
        pass  # ObservationManager owns and resets the explicit history.

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        pass

    def update_normalization(self, obs):
        pass  # Inputs and supervised targets use the task's fixed physical scales.

    def as_jit(self):
        return FutureActorExport(self.core)

    def as_onnx(self, verbose=False):
        return FutureActorExport(self.core)
