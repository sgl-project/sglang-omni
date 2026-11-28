# SPDX-License-Identifier: Apache-2.0
"""Causal windowed transformer layers for the Audio8 codec."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, kw_only=True)
class CodecTransformerConfig:
    layer_count: int
    attention_head_count: int
    hidden_size: int
    intermediate_size: int
    key_value_head_count: int | None = None
    head_size: int = 64
    rope_base: float = 10_000
    norm_epsilon: float = 1e-5
    dropout_rate: float = 0.1


def apply_rotary(activations: Tensor, rotations: Tensor) -> Tensor:
    pairs = activations.float().reshape(*activations.shape[:-1], -1, 2)
    rotations = rotations.view(1, pairs.shape[1], 1, pairs.shape[3], 2)
    rotated = torch.stack(
        (
            pairs[..., 0] * rotations[..., 0] - pairs[..., 1] * rotations[..., 1],
            pairs[..., 1] * rotations[..., 0] + pairs[..., 0] * rotations[..., 1],
        ),
        dim=-1,
    )
    return rotated.flatten(3).to(activations.dtype)


class CodecRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.epsilon: float = epsilon
        self.weight: nn.Parameter = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(
            normalized.pow(2).mean(-1, keepdim=True) + self.epsilon
        )
        return normalized.to(hidden_states.dtype) * self.weight


class CodecLayerScale(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gamma: nn.Parameter = nn.Parameter(1e-2 * torch.ones(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states.mul_(self.gamma)


class CodecAttention(nn.Module):
    def __init__(self, config: CodecTransformerConfig) -> None:
        super().__init__()
        self.attention_head_count: int = config.attention_head_count
        self.key_value_head_count: int = (
            config.key_value_head_count
            if config.key_value_head_count is not None
            else config.attention_head_count
        )
        self.head_size: int = config.head_size
        self.dropout_rate: float = config.dropout_rate
        self.query_size: int = self.attention_head_count * self.head_size
        self.key_value_size: int = self.key_value_head_count * self.head_size
        self.wqkv: nn.Linear = nn.Linear(
            config.hidden_size, self.query_size + 2 * self.key_value_size, bias=False
        )
        self.wo: nn.Linear = nn.Linear(self.query_size, config.hidden_size, bias=False)

    def forward(
        self, hidden_states: Tensor, rotations: Tensor, attention_mask: Tensor
    ) -> Tensor:
        batch_size, token_count, _ = hidden_states.shape
        query, key, value = self.wqkv(hidden_states).split(
            (self.query_size, self.key_value_size, self.key_value_size), dim=-1
        )
        query = query.view(
            batch_size, token_count, self.attention_head_count, self.head_size
        )
        key = key.view(
            batch_size, token_count, self.key_value_head_count, self.head_size
        )
        value = value.view_as(key)
        query = apply_rotary(query, rotations).transpose(1, 2)
        key = apply_rotary(key, rotations).transpose(1, 2)
        value = value.transpose(1, 2)
        repeats = self.attention_head_count // self.key_value_head_count
        attended = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(repeats, dim=1),
            value.repeat_interleave(repeats, dim=1),
            attn_mask=attention_mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
        )
        return self.wo(
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, token_count, self.query_size)
        )


class CodecFeedForward(nn.Module):
    def __init__(self, config: CodecTransformerConfig) -> None:
        super().__init__()
        self.w1: nn.Linear = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.w3: nn.Linear = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.w2: nn.Linear = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.dropout: nn.Dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.w2(
            self.dropout(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))
        )


class CodecTransformerBlock(nn.Module):
    def __init__(self, config: CodecTransformerConfig) -> None:
        super().__init__()
        self.attention: CodecAttention = CodecAttention(config)
        self.feed_forward: CodecFeedForward = CodecFeedForward(config)
        self.ffn_norm: CodecRMSNorm = CodecRMSNorm(
            config.hidden_size, config.norm_epsilon
        )
        self.attention_norm: CodecRMSNorm = CodecRMSNorm(
            config.hidden_size, config.norm_epsilon
        )
        self.attention_layer_scale: CodecLayerScale = CodecLayerScale(
            config.hidden_size
        )
        self.ffn_layer_scale: CodecLayerScale = CodecLayerScale(config.hidden_size)

    def forward(
        self, hidden_states: Tensor, rotations: Tensor, attention_mask: Tensor
    ) -> Tensor:
        hidden_states = hidden_states + self.attention_layer_scale(
            self.attention(
                self.attention_norm(hidden_states), rotations, attention_mask
            )
        )
        return hidden_states + self.ffn_layer_scale(
            self.feed_forward(self.ffn_norm(hidden_states))
        )


class CodecWindowTransformer(nn.Module):
    def __init__(self, config: CodecTransformerConfig, window_size: int) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(
            [CodecTransformerBlock(config) for _ in range(config.layer_count)]
        )
        self.norm: CodecRMSNorm = CodecRMSNorm(config.hidden_size, config.norm_epsilon)
        self.window_size: int = window_size
        self.head_size: int = config.head_size
        self.rope_base: float = config.rope_base

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        token_count = hidden_states.shape[1]
        token_positions = torch.arange(token_count, device=hidden_states.device)
        row = token_positions[:, None]
        column = token_positions[None, :]
        mask = (column <= row) & (column >= (row - self.window_size + 1).clamp_min(0))
        frequencies = 1.0 / (
            self.rope_base
            ** (
                torch.arange(0, self.head_size, 2, device=hidden_states.device).float()
                / self.head_size
            )
        )
        phases = torch.outer(token_positions, frequencies)
        rotations = torch.polar(torch.ones_like(phases), phases)
        rotations = torch.stack((rotations.real, rotations.imag), dim=-1).to(
            torch.bfloat16
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotations, mask[None, None])
        return self.norm(hidden_states).transpose(1, 2)
