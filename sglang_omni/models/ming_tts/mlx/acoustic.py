# SPDX-License-Identifier: Apache-2.0
"""MLX inference counterparts of Ming talker's DiT and Aggregator."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import AcousticConfig


def rotary_frequencies(length: int, head_dim: int) -> tuple[mx.array, mx.array]:
    inv_freq = 1.0 / (
        10000.0 ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    )
    phase = mx.arange(length, dtype=mx.float32)[:, None] * inv_freq
    return mx.cos(phase), mx.sin(phase)


def apply_rotary(x: mx.array, rope: tuple[mx.array, mx.array]) -> mx.array:
    cos, sin = rope
    even = x[..., 0::2].astype(mx.float32)
    odd = x[..., 1::2].astype(mx.float32)
    return mx.stack(
        (even * cos - odd * sin, odd * cos + even * sin), axis=-1
    ).reshape(x.shape).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = [nn.Linear(dim, dim)]

    def __call__(self, x: mx.array, rope: tuple[mx.array, mx.array]) -> mx.array:
        batch, length, dim = x.shape
        q, k, v = [
            proj(x).reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            for proj in (self.to_q, self.to_k, self.to_v)
        ]
        q, k = apply_rotary(q, rope), apply_rotary(k, rope)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5
        )
        return self.to_out[0](out.transpose(0, 2, 1, 3).reshape(batch, length, dim))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mult))
        self.fc2 = nn.Linear(int(dim * mult), dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class DiTBlock(nn.Module):
    def __init__(self, config: AcousticConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config.hidden_size, config.num_heads)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.mlp = FeedForward(config.hidden_size, config.mlp_ratio)

    def __call__(self, x: mx.array, rope: tuple[mx.array, mx.array]) -> mx.array:
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(self.norm_final(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.time_mlp = [nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim)]
        self._freqs = mx.exp(
            mx.arange(128, dtype=mx.float32) * (-math.log(10000) / 127)
        )
        # Private arrays are excluded from parameters(); finish before thread handoff.
        mx.eval(self._freqs)

    def __call__(self, t: mx.array) -> mx.array:
        phase = 1000 * t[:, None].astype(mx.float32) * self._freqs
        x = mx.concatenate((mx.sin(phase), mx.cos(phase)), axis=-1)
        x = x.astype(self.time_mlp[0].weight.dtype)
        for layer in self.time_mlp:
            x = layer(x)
        return x


class CondEmbedder(nn.Module):
    def __init__(self, llm_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.cond_embedder = nn.Linear(llm_dim, hidden_size)

    def __call__(self, c: mx.array) -> mx.array:
        return self.cond_embedder(c.astype(self.cond_embedder.weight.dtype))


class DiT(nn.Module):
    def __init__(self, config: AcousticConfig, latent_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.head_dim = config.hidden_size // config.num_heads
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.x_embedder = nn.Linear(latent_dim, config.hidden_size)
        self.c_embedder = CondEmbedder(llm_dim, config.hidden_size)
        self.blocks = [DiTBlock(config) for _ in range(config.depth)]
        self.final_layer = FinalLayer(config.hidden_size, latent_dim)

    def __call__(
        self, x: mx.array, t: mx.array, c: mx.array, latent_history: mx.array
    ) -> mx.array:
        x = mx.concatenate((latent_history, x), axis=1)
        x = x.astype(self.x_embedder.weight.dtype)
        x = self.x_embedder(x)
        cond = self.t_embedder(t)[:, None] + self.c_embedder(c)
        x = mx.concatenate((cond, x), axis=1)
        rope = rotary_frequencies(x.shape[1], self.head_dim)
        for block in self.blocks:
            x = block(x, rope)
        return self.final_layer(x)

    def forward_with_cfg(
        self, x: mx.array, t: mx.array, c: mx.array, latent_history: mx.array
    ) -> mx.array:
        patch_size = x.shape[1]
        x = mx.concatenate((x, x), axis=0)
        c = mx.concatenate((c, mx.zeros_like(c)), axis=0)
        history = mx.concatenate((latent_history, latent_history), axis=0)
        t = mx.broadcast_to(t, (x.shape[0],))
        return self(x, t, c, history)[:, -patch_size:]


class Aggregator(nn.Module):
    def __init__(self, config: AcousticConfig, latent_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.head_dim = config.hidden_size // config.num_heads
        self.word_embedder = nn.Embedding(1, config.hidden_size)
        self.x_embedder = nn.Linear(latent_dim, config.hidden_size)
        self.blocks = [DiTBlock(config) for _ in range(config.depth)]
        self.final_layer = FinalLayer(config.hidden_size, llm_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.x_embedder(x.astype(self.x_embedder.weight.dtype))
        cls = self.word_embedder(mx.zeros((x.shape[0], 1), dtype=mx.int32))
        x = mx.concatenate((cls, x), axis=1)
        rope = rotary_frequencies(x.shape[1], self.head_dim)
        for block in self.blocks:
            x = block(x, rope)
        return self.final_layer(x)[:, :1]
