# SPDX-License-Identifier: Apache-2.0
# Adapted from Stability-AI/stable-audio-3 at commit 790793b89ddceea612be9ce42ff6830d18f6fa57.
# Copyright (c) 2026 Stability AI. The MIT license notice is retained in licenses/.
"""Transformer layers shared by the Stable Audio 3 DiT and the SAME decoder."""

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from sglang_omni.platforms import current_platform

if current_platform.is_cuda():
    from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
    from sglang.srt.utils.custom_op import register_custom_op
else:
    pass

AttentionBackend = Literal["flash", "sdpa"]
ROTARY_BASE = 10000.0
QK_RMS_NORM_EPS = 1e-6
FOURIER_MIN_FREQUENCY = 0.5
FOURIER_MAX_FREQUENCY = 10000.0
# note (ruilin): the official inference uses this truncated pi, so the angles match it bit for bit.
TRUNCATED_PI = 3.14159265359
# note (ruilin): differential attention runs its two softmax maps as one call over twice the heads; self-attention projects queries, keys and values plus the second map's queries and keys, cross-attention keys and values plus the second map's keys.
DIFFERENTIAL_MAPS = 2
SELF_ATTENTION_PROJECTIONS = 5
CROSS_ATTENTION_PROJECTIONS = 3


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
        self.eps: float = eps
        self.force_fp32: bool = force_fp32

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
    """Rotary angles for the first dim channels of each head, shaped [tokens, dim]."""

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
    """Rotate the leading channels of x [batch, tokens, heads, head_dim] in fp32, leaving the rest unchanged."""
    rotary_dim = angles.shape[-1]
    angles = angles[:, None]
    rotated, unrotated = x[..., :rotary_dim].float(), x[..., rotary_dim:]
    first_half, second_half = rotated.chunk(2, dim=-1)
    rotated = rotated * angles.cos() + torch.cat((-second_half, first_half), -1) * (
        angles.sin()
    )
    return torch.cat((rotated.to(x.dtype), unrotated), dim=-1)


def flash_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_lengths: torch.Tensor | None,
    window_tokens: int,
) -> torch.Tensor:
    """FA3 over [batch, tokens, heads, head_dim]; a window_tokens of -1 is unbounded."""
    # note (ruilin): one split keeps a row's reduction order independent of the rows batched with it.
    return flash_attn_varlen_func(
        queries,
        keys,
        values,
        None,
        None,
        max_seqlen_q=queries.shape[1],
        max_seqlen_k=keys.shape[1],
        seqused_k=key_lengths,
        window_size=(window_tokens, window_tokens),
        num_splits=1,
    )


if current_platform.is_cuda():
    # note (ruilin): the compiled blocks reach FA3 through this op, so each block traces as one graph.
    flash_attention_op = register_custom_op(
        flash_attention,
        op_name="stable_audio3_flash_attention",
        mutates_args=[],
        out_shape="queries",
    )
else:
    pass


def attend(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    backend: AttentionBackend,
    key_lengths: torch.Tensor | None,
    window_tokens: int | None,
) -> torch.Tensor:
    """Attention over [batch, tokens, heads, head_dim] tensors; each row sees its first key_lengths keys, and each query the keys within window_tokens positions."""
    if backend == "flash":
        return flash_attention_op(
            queries,
            keys,
            values,
            key_lengths,
            -1 if window_tokens is None else window_tokens,
        )
    else:
        positions = torch.arange(keys.shape[1], device=keys.device)
        mask = None
        if key_lengths is not None:
            mask = (positions < key_lengths[:, None])[:, None, None, :]
        else:
            pass
        if window_tokens is not None:
            band = (positions[None, :] - positions[:, None]).abs() <= window_tokens
            mask = band if mask is None else mask & band
        else:
            pass
        return F.scaled_dot_product_attention(
            queries.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
            attn_mask=mask,
        ).transpose(1, 2)


class DifferentialAttention(nn.Module):
    """Attention whose output is the difference of two softmax maps over the same values; both maps run as one call over twice the heads."""

    def __init__(
        self,
        dim: int,
        *,
        head_dim: int,
        context_dim: int | None,
        qk_norm: Literal["rms", "dyt"],
        backend: AttentionBackend,
    ) -> None:
        super().__init__()
        self.num_heads: int = dim // head_dim
        self.backend: AttentionBackend = backend
        if context_dim is None:
            self.to_qkv = nn.Linear(dim, dim * SELF_ATTENTION_PROJECTIONS, bias=False)
        else:
            self.to_q = nn.Linear(dim, dim * DIFFERENTIAL_MAPS, bias=False)
            self.to_kv = nn.Linear(
                context_dim, context_dim * CROSS_ATTENTION_PROJECTIONS, bias=False
            )
        self.to_out = nn.Linear(dim, dim, bias=False)
        if qk_norm == "rms":
            self.q_norm = RMSNorm(head_dim, eps=QK_RMS_NORM_EPS, force_fp32=False)
            self.k_norm = RMSNorm(head_dim, eps=QK_RMS_NORM_EPS, force_fp32=False)
        else:
            self.q_norm = DynamicTanh(head_dim)
            self.k_norm = DynamicTanh(head_dim)

    def split_heads(self, x: torch.Tensor, parts: int) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        return x.view(batch_size, sequence_length, parts * self.num_heads, -1)

    def context_keys_values(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys of both maps and the values for cross-attention, fixed across sampling steps."""
        keys, values = self.split_heads(
            self.to_kv(context), CROSS_ATTENTION_PROJECTIONS
        ).split([DIFFERENTIAL_MAPS * self.num_heads, self.num_heads], dim=2)
        return self.k_norm(keys), torch.cat([values, values], dim=2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context_keys_values: tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_angles: torch.Tensor | None = None,
        key_lengths: torch.Tensor | None = None,
        window_tokens: int | None = None,
    ) -> torch.Tensor:
        if context_keys_values is None:
            queries, keys, values, query_diffs, key_diffs = self.split_heads(
                self.to_qkv(x), SELF_ATTENTION_PROJECTIONS
            ).chunk(SELF_ATTENTION_PROJECTIONS, dim=2)
            queries = self.q_norm(torch.cat([queries, query_diffs], dim=2))
            keys = self.k_norm(torch.cat([keys, key_diffs], dim=2))
            values = torch.cat([values, values], dim=2)
        else:
            queries = self.q_norm(self.split_heads(self.to_q(x), DIFFERENTIAL_MAPS))
            keys, values = context_keys_values
        if rotary_angles is not None:
            queries = apply_rotary(queries, rotary_angles)
            keys = apply_rotary(keys, rotary_angles)
        else:
            pass
        maps = attend(
            queries,
            keys,
            values,
            backend=self.backend,
            key_lengths=key_lengths,
            window_tokens=window_tokens,
        )
        difference = maps[:, :, : self.num_heads] - maps[:, :, self.num_heads :]
        return self.to_out(difference.flatten(2))


class GatedLinearUnit(nn.Module):
    def __init__(self, dim: int, inner_dim: int, *, sinusoidal: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2)
        self.sinusoidal: bool = sinusoidal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, gates = self.proj(x).chunk(2, dim=-1)
        if self.sinusoidal:
            return values * torch.sin(TRUNCATED_PI * gates)
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
