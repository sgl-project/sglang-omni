# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Tencent. All rights reserved.
"""MLX attention and modulation blocks for AuK."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.inv_freq = 1 / (10000 ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

    def __call__(self, positions: mx.array) -> tuple[mx.array, mx.array]:
        positions = positions[None, :] if positions.ndim == 1 else positions
        # note (Codex): BF16 position indices repeat beyond 256; keep angles FP32.
        angles = positions.astype(mx.float32)[..., None] * self.inv_freq.astype(
            mx.float32
        )
        return mx.cos(angles)[:, None], mx.sin(angles)[:, None]


def apply_rope(x: mx.array, rope: tuple[mx.array, mx.array]) -> mx.array:
    cosine, sine = rope
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    return (
        mx.stack([even * cosine - odd * sine, odd * cosine + even * sine], axis=-1)
        .reshape(x.shape)
        .astype(x.dtype)
    )


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256) -> None:
        super().__init__()
        self.freq_embed_dim = freq_embed_dim
        self.time_mlp = [nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)]

    def __call__(self, timestep: mx.array) -> mx.array:
        half_dim = self.freq_embed_dim // 2
        frequencies = mx.exp(
            mx.arange(half_dim).astype(mx.float32) * (-math.log(10000) / (half_dim - 1))
        )
        angles = 1000 * timestep.astype(mx.float32)[:, None] * frequencies[None, :]
        hidden = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)
        hidden = hidden.astype(self.time_mlp[0].weight.dtype)
        for layer in self.time_mlp:
            hidden = layer(hidden)
        return hidden


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(
        self, x: mx.array, emb: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            self.linear(nn.silu(emb)), 6, axis=-1
        )
        norm = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return norm, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        scale, shift = mx.split(self.linear(nn.silu(emb)), 2, axis=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, mult: float) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.linear_out = nn.Linear(inner_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, value = mx.split(self.linear_in(x), 2, axis=-1)
        return self.linear_out(nn.silu(gate) * value)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, joint: bool) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.q_norm = nn.RMSNorm(dim_head, eps=mx.finfo(mx.float32).eps)
        self.k_norm = nn.RMSNorm(dim_head, eps=mx.finfo(mx.float32).eps)
        self.to_out = [nn.Linear(inner_dim, dim)]
        if joint:
            self.to_qkv_c = nn.Linear(dim, inner_dim * 3)
            self.c_q_norm = nn.RMSNorm(dim_head, eps=mx.finfo(mx.float32).eps)
            self.c_k_norm = nn.RMSNorm(dim_head, eps=mx.finfo(mx.float32).eps)
            self.to_out_c = nn.Linear(inner_dim, dim)
        else:
            pass

    def heads_from_projection(
        self, projection: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        return tuple(
            tensor.reshape(tensor.shape[0], -1, self.heads, self.dim_head).transpose(
                0, 2, 1, 3
            )
            for tensor in mx.split(projection, 3, axis=-1)
        )

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array],
        mask: mx.array | None,
        bias: mx.array | None,
        c: mx.array | None = None,
        c_rope: tuple[mx.array, mx.array] | None = None,
        c_mask: mx.array | None = None,
    ) -> mx.array | tuple[mx.array, mx.array]:
        query, key, value = self.heads_from_projection(self.to_qkv(x))
        query = apply_rope(self.q_norm(query), rope)
        key = apply_rope(self.k_norm(key), rope)
        if c is not None:
            assert c_rope is not None
            c_query, c_key, c_value = self.heads_from_projection(self.to_qkv_c(c))
            c_query = apply_rope(self.c_q_norm(c_query), c_rope)
            c_key = apply_rope(self.c_k_norm(c_key), c_rope)
            query = mx.concatenate([query, c_query], axis=2)
            key = mx.concatenate([key, c_key], axis=2)
            value = mx.concatenate([value, c_value], axis=2)
        else:
            pass

        out = (
            mx.fast.scaled_dot_product_attention(
                query, key, value, scale=self.dim_head**-0.5, mask=bias
            )
            .transpose(0, 2, 1, 3)
            .reshape(x.shape[0], -1, self.heads * self.dim_head)
        )
        x_out = self.to_out[0](out[:, : x.shape[1]])
        x_out = mx.where(mask[..., None], x_out, 0) if mask is not None else x_out
        if c is None:
            return x_out
        else:
            c_out = self.to_out_c(out[:, x.shape[1] :])
            c_out = (
                mx.where(c_mask[..., None], c_out, 0) if c_mask is not None else c_out
            )
            return x_out, c_out


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(dim, heads, dim_head, joint=False)
        self.ff_norm = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.ff = SwiGLUFeedForward(dim, ff_mult)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        rope: tuple[mx.array, mx.array],
        mask: mx.array | None,
        bias: mx.array | None,
    ) -> mx.array:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, t)
        x = x + gate_msa[:, None] * self.attn(norm, rope, mask, bias)
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp[:, None] * self.ff(norm)


class MMDiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float) -> None:
        super().__init__()
        self.attn_norm_c = AdaLayerNorm(dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(dim, heads, dim_head, joint=True)
        self.ff_norm_c = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.ff_c = SwiGLUFeedForward(dim, ff_mult)
        self.ff_norm_x = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.ff_x = SwiGLUFeedForward(dim, ff_mult)

    def __call__(
        self,
        x: mx.array,
        c: mx.array,
        t: mx.array,
        rope: tuple[mx.array, mx.array],
        c_rope: tuple[mx.array, mx.array],
        mask: mx.array | None,
        c_mask: mx.array,
        bias: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(
            c, t
        )
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(
            x, t
        )
        x_attn, c_attn = self.attn(norm_x, rope, mask, bias, norm_c, c_rope, c_mask)
        c = c + c_gate_msa[:, None] * c_attn
        norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        c = c + c_gate_mlp[:, None] * self.ff_c(norm_c)
        x = x + x_gate_msa[:, None] * x_attn
        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        return c, x + x_gate_mlp[:, None] * self.ff_x(norm_x)


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16) -> None:
        super().__init__()
        self.conv1d = [
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        ]

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        x = mx.where(mask[..., None], x, 0) if mask is not None else x
        for layer in self.conv1d:
            x = layer(x)
            if mask is not None and isinstance(layer, nn.Conv1d):
                x = mx.where(mask[..., None], x, 0)
            else:
                pass
        return x


class AudioPromptEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        x = self.linear(x)
        return x + self.conv_pos_embed(x, mask)
