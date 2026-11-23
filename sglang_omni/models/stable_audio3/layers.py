# SPDX-License-Identifier: Apache-2.0
# Adapted from Stability-AI/stable-audio-3 at commit 790793b89ddceea612be9ce42ff6830d18f6fa57.
# Copyright (c) 2026 Stability AI. The MIT license notice is retained in licenses/.
"""Transformer layers shared by the Stable Audio 3 DiT and the SAME decoder."""

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

ROTARY_BASE = 10000.0
QK_RMS_NORM_EPS = 1e-6
FOURIER_MIN_FREQUENCY = 0.5
FOURIER_MAX_FREQUENCY = 10000.0


def expo_fourier_features(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Cosine and sine features at log-spaced frequencies, computed in fp32 and returned in the input dtype."""
    frequencies = torch.exp(
        torch.linspace(0, 1, dim // 2, device=values.device, dtype=torch.float32)
        * (math.log(FOURIER_MAX_FREQUENCY) - math.log(FOURIER_MIN_FREQUENCY))
        + math.log(FOURIER_MIN_FREQUENCY)
    )
    phases = values.float().unsqueeze(-1) * frequencies * 2 * math.pi
    return torch.cat([phases.cos(), phases.sin()], dim=-1).to(values.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float, force_fp32: bool) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.force_fp32 = force_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.force_fp32:
            normed = F.rms_norm(x.float(), x.shape[-1:], self.gamma.float(), self.eps)
            return normed.to(x.dtype)
        else:
            return F.rms_norm(x, x.shape[-1:], self.gamma, self.eps)


class DynamicTanh(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class RotaryEmbedding(nn.Module):
    """Rotary angles for the first dim channels of each head."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        exponents = torch.arange(0, dim, 2).float() / dim
        self.register_buffer("inv_freq", 1.0 / ROTARY_BASE**exponents)

    def forward(self, sequence_length: int) -> torch.Tensor:
        positions = torch.arange(
            sequence_length, device=self.inv_freq.device, dtype=torch.float32
        )
        angles = torch.outer(positions, self.inv_freq.float())
        return torch.cat((angles, angles), dim=-1)


def apply_rotary(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate the leading channels of x in fp32, leaving the rest unchanged."""
    rotary_dim = angles.shape[-1]
    rotated, unrotated = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    first_half, second_half = rotated.chunk(2, dim=-1)
    rotated = rotated * angles.cos() + torch.cat((-second_half, first_half), -1) * (
        angles.sin()
    )
    return torch.cat((rotated.to(x.dtype), unrotated), dim=-1)


class DifferentialAttention(nn.Module):
    """Attention whose output is the difference of two softmax maps over the same values."""

    def __init__(
        self,
        dim: int,
        *,
        head_dim: int,
        context_dim: int | None,
        qk_norm: Literal["rms", "dyt"],
    ) -> None:
        super().__init__()
        self.num_heads = dim // head_dim
        if context_dim is None:
            self.to_qkv = nn.Linear(dim, dim * 5, bias=False)
        else:
            self.to_q = nn.Linear(dim, dim * 2, bias=False)
            self.to_kv = nn.Linear(context_dim, context_dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        if qk_norm == "rms":
            self.q_norm = RMSNorm(head_dim, eps=QK_RMS_NORM_EPS, force_fp32=False)
            self.k_norm = RMSNorm(head_dim, eps=QK_RMS_NORM_EPS, force_fp32=False)
        else:
            self.q_norm = DynamicTanh(head_dim)
            self.k_norm = DynamicTanh(head_dim)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        return x.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        rotary_angles: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context is None:
            q, k, v, q_diff, k_diff = self.to_qkv(x).chunk(5, dim=-1)
        else:
            q, q_diff = self.to_q(x).chunk(2, dim=-1)
            k, k_diff, v = self.to_kv(context).chunk(3, dim=-1)
        q = self.q_norm(torch.stack([self.split_heads(q), self.split_heads(q_diff)]))
        k = self.k_norm(torch.stack([self.split_heads(k), self.split_heads(k_diff)]))
        if rotary_angles is not None:
            q, k = apply_rotary(q, rotary_angles), apply_rotary(k, rotary_angles)
        else:
            pass
        v = self.split_heads(v)
        maps = [
            F.scaled_dot_product_attention(
                q[index], k[index], v, attn_mask=attention_mask
            )
            for index in range(2)
        ]
        return self.to_out((maps[0] - maps[1]).transpose(1, 2).flatten(2))


class GatedLinearUnit(nn.Module):
    def __init__(self, dim: int, inner_dim: int, *, sinusoidal: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2)
        self.sinusoidal = sinusoidal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, gates = self.proj(x).chunk(2, dim=-1)
        if self.sinusoidal:
            # note (ruilin): the checkpoint was trained with this truncated pi.
            return values * torch.sin(3.14159265359 * gates)
        else:
            return values * F.silu(gates)


class GatedFeedForward(nn.Module):
    def __init__(self, dim: int, *, multiplier: float, sinusoidal: bool) -> None:
        super().__init__()
        inner_dim = int(dim * multiplier)
        # note (ruilin): the identity keeps the checkpoint's ff.0 / ff.2 tensor names.
        self.ff = nn.Sequential(
            GatedLinearUnit(dim, inner_dim, sinusoidal=sinusoidal),
            nn.Identity(),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)
