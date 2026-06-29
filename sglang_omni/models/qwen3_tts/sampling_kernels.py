# SPDX-License-Identifier: Apache-2.0
"""Optional sampling kernels for Qwen3-TTS."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rotl32(x, r: tl.constexpr) -> tl.uint32:
        x = x.to(tl.uint64)
        return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF

    @triton.jit
    def _fmix32(h: tl.uint32) -> tl.uint32:
        h ^= h >> 16
        h = (h * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xC2B2AE35) & 0xFFFFFFFF
        h ^= h >> 16
        return h

    @triton.jit
    def _murmur3_mix(h: tl.uint32, k: tl.uint32) -> tl.uint32:
        k = (k * 0xCC9E2D51) & 0xFFFFFFFF
        k = _rotl32(k, 15)
        k = (k * 0x1B873593) & 0xFFFFFFFF
        h ^= k
        h = _rotl32(h, 13)
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
        return h

    @triton.jit
    def _seeded_gumbel_sample_sorted_kernel(
        probs,
        sorted_idx,
        seeds,
        positions,
        out,
        num_cols: tl.constexpr,
        probs_stride_b: tl.constexpr,
        probs_stride_k: tl.constexpr,
        idx_stride_b: tl.constexpr,
        idx_stride_k: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < num_cols

        seed = tl.load(seeds + row).to(tl.uint64)
        pos = tl.load(positions + row).to(tl.uint32)
        col = offsets.to(tl.uint32)

        h: tl.uint32 = 0
        h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
        h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
        h = _murmur3_mix(h, pos)
        h = _murmur3_mix(h, col)
        h ^= 16
        h = _fmix32(h)

        u = h.to(tl.float64) / 4294967295.0
        u = tl.maximum(u, 2.2250738585072014e-308)
        gumbel = -tl.log(-tl.log(u))
        weights = tl.load(
            probs + row * probs_stride_b + offsets * probs_stride_k,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float64)
        scores = tl.where(mask, weights + gumbel, -float("inf"))
        max_score = tl.max(scores, axis=0)
        candidates = tl.where(scores == max_score, offsets, num_cols)
        rank = tl.min(candidates, axis=0)
        token = tl.load(sorted_idx + row * idx_stride_b + rank * idx_stride_k)
        tl.store(out + row, token)

else:
    _seeded_gumbel_sample_sorted_kernel = None


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def sample_from_sorted_probs_with_seed_small_k(
    probs: torch.Tensor,
    sorted_idx: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor | None:
    if (
        _seeded_gumbel_sample_sorted_kernel is None
        or not probs.is_cuda
        or not sorted_idx.is_cuda
        or not seeds.is_cuda
        or not positions.is_cuda
    ):
        return None
    if probs.ndim != 2 or sorted_idx.shape != probs.shape:
        return None
    if seeds.ndim != 1 or positions.ndim != 1:
        return None
    batch_size, num_cols = probs.shape
    if batch_size == 0:
        return torch.empty((0,), device=probs.device, dtype=torch.long)
    if seeds.shape[0] != batch_size or positions.shape[0] != batch_size:
        return None
    if num_cols <= 0 or num_cols > 1024:
        return None

    block_size = _next_power_of_2(num_cols)
    out = torch.empty((batch_size,), device=probs.device, dtype=torch.long)
    _seeded_gumbel_sample_sorted_kernel[(batch_size,)](
        probs,
        sorted_idx,
        seeds,
        positions,
        out,
        int(num_cols),
        probs.stride(0),
        probs.stride(1),
        sorted_idx.stride(0),
        sorted_idx.stride(1),
        block_size,
    )
    return out
