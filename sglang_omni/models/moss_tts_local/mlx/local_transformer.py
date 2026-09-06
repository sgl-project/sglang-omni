# SPDX-License-Identifier: Apache-2.0
"""The frame-local GPT-2 decoder used by MOSS-TTS Local on MLX.

Adapted from the MIT-licensed mlx-audio implementation.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from .config import GPT2Config


def _rotate_half(x: mx.array) -> mx.array:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return mx.stack([-odd, even], axis=-1).reshape(x.shape)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float) -> None:
        super().__init__()
        self.dim = dim
        self.base = base

    def __call__(self, x: mx.array) -> mx.array:
        positions = mx.arange(x.shape[1], dtype=mx.float32)
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        freqs = positions[:, None] * inv_freq[None, :]
        cos = mx.repeat(mx.cos(freqs), 2, axis=-1)[None, :, None, :].astype(x.dtype)
        sin = mx.repeat(mx.sin(freqs), 2, axis=-1)[None, :, None, :].astype(x.dtype)
        return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, config: GPT2Config, layer_index: int) -> None:
        super().__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim**-0.5
        if config.scale_attn_by_inverse_layer_idx:
            self.scale /= layer_index + 1
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_base)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, width = x.shape
        query, key, value = mx.split(self.c_attn(x), 3, axis=-1)
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self.rope(query.reshape(shape)).transpose(0, 2, 1, 3)
        key = self.rope(key.reshape(shape)).transpose(0, 2, 1, 3)
        value = value.reshape(shape).transpose(0, 2, 1, 3)
        positions = mx.arange(length)
        mask = mx.where(
            positions[:, None] >= positions[None, :],
            0.0,
            mx.finfo(x.dtype).min,
        ).astype(x.dtype)
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask=mask
        )
        return self.c_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, width))


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        inner = config.n_inner or 4 * config.n_embd
        self.fc_in = nn.Linear(config.n_embd, inner, bias=True)
        self.fc_out = nn.Linear(inner, config.n_embd, bias=True)
        self.activation = config.activation_function

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc_in(x)
        if self.activation == "silu":
            x = nn.silu(x)
        elif self.activation == "gelu_new":
            x = (
                0.5
                * x
                * (
                    1.0
                    + mx.tanh(
                        math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3))
                    )
                )
            )
        else:
            x = nn.gelu(x)
        return self.fc_out(x)


class Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_index: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(config, layer_index)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class LocalTransformer(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.h = [Block(config, index) for index in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden = inputs
        for block in self.h:
            hidden = block(hidden)
        return self.ln_f(hidden)
