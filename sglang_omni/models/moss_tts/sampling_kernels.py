# SPDX-License-Identifier: Apache-2.0
"""Graph-friendly seeded sampling kernels owned by MOSS-TTS."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.sampling.murmur_hash import fmix32, murmur3_mix, murmur_hash32
from sglang.srt.layers.sampler import multinomial_with_seed
from triton.language.extra import libdevice

_UINT32_MAX_F64 = tl.constexpr(float(torch.iinfo(torch.uint32).max))


@triton.jit
def _seeded_gumbel_argmax_kernel(
    scores_ptr,
    seeds_ptr,
    positions_ptr,
    output_ptr,
    VOCAB_SIZE: tl.constexpr,
    SCORE_ROW_STRIDE: tl.constexpr,
    SCORE_COL_STRIDE: tl.constexpr,
    SEED_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate seeded Gumbel noise and reduce one vocabulary row in one pass."""

    row = tl.program_id(0)
    token_ids = tl.arange(0, BLOCK_SIZE)
    valid = token_ids < VOCAB_SIZE
    seed = tl.load(seeds_ptr + row * SEED_STRIDE).to(tl.uint64)
    position = tl.load(positions_ptr + row * POSITION_STRIDE).to(tl.uint32)

    hash_value: tl.uint32 = 0
    hash_value = murmur3_mix(hash_value, (seed & 0xFFFFFFFF).to(tl.uint32))
    hash_value = murmur3_mix(
        hash_value,
        ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32),
    )
    hash_value = murmur3_mix(hash_value, position)
    hash_value = murmur3_mix(hash_value, token_ids.to(tl.uint32))
    hash_value ^= 16
    hash_value = fmix32(hash_value)

    # Match multinomial_with_seed exactly, including hash_value == UINT32_MAX.
    uniform = hash_value.to(tl.float64) / _UINT32_MAX_F64
    log_uniform = libdevice.log(uniform)
    neg_log_uniform = -tl.maximum(log_uniform, -1.7976931348623157e308)
    gumbel = -libdevice.log(neg_log_uniform)
    score = tl.load(
        scores_ptr + row * SCORE_ROW_STRIDE + token_ids * SCORE_COL_STRIDE,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float64)
    value = score + gumbel

    is_nan = value != value
    nan_index = tl.min(tl.where(valid & is_nan, token_ids, 2147483647), axis=0)
    non_nan_value = tl.where(valid & ~is_nan, value, -float("inf"))
    max_value = tl.max(non_nan_value, axis=0)
    max_index = tl.min(
        tl.where(
            valid & ~is_nan & (non_nan_value == max_value),
            token_ids,
            2147483647,
        ),
        axis=0,
    )
    tl.store(
        output_ptr + row,
        tl.where(nan_index != 2147483647, nan_index, max_index),
    )


def seeded_gumbel_argmax(
    scores: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Sample rows without materializing a full-vocabulary Gumbel tensor."""

    if scores.ndim != 2:
        raise ValueError(f"seeded Gumbel scores must be rank 2, got {scores.shape}")
    rows, vocab_size = scores.shape
    if seeds.shape != (rows,) or positions.shape != (rows,):
        raise ValueError("seeded Gumbel metadata shape mismatch")
    if scores.device.type != "cuda" or not (
        seeds.device == scores.device
        and positions.device == scores.device
        and output.device == scores.device
    ):
        raise ValueError("seeded_gumbel_argmax requires one CUDA device")
    if scores.dtype != torch.float32:
        raise TypeError(f"seeded Gumbel scores must be float32, got {scores.dtype}")
    if seeds.dtype != torch.int64 or positions.dtype != torch.int64:
        raise TypeError("seeded Gumbel seeds and positions must be int64")
    if output.dtype != torch.int64 or output.ndim != 1 or output.numel() < rows:
        raise ValueError(
            "seeded Gumbel output must be a sufficiently large int64 vector"
        )
    if output.stride(0) != 1:
        raise ValueError("seeded Gumbel output must have stride 1")

    block_size = triton.next_power_of_2(vocab_size)
    if block_size > 2048:
        raise ValueError(
            f"seeded Gumbel one-pass vocabulary exceeds 2048: {vocab_size}"
        )
    result = output[:rows]
    _seeded_gumbel_argmax_kernel[(rows,)](
        scores,
        seeds,
        positions,
        result,
        VOCAB_SIZE=vocab_size,
        SCORE_ROW_STRIDE=scores.stride(0),
        SCORE_COL_STRIDE=scores.stride(1),
        SEED_STRIDE=seeds.stride(0),
        POSITION_STRIDE=positions.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return result


@torch.compile(dynamic=True)
def multinomial_with_seed_and_token_ids(
    logprobs: torch.Tensor,
    seed: torch.Tensor,
    positions: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Seeded Gumbel-max using original vocabulary ids as RNG columns."""

    seed = seed.to(torch.uint64)
    hashed = murmur_hash32(seed, positions, token_ids)
    noise = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max
    noise.log_().clamp_(min=torch.finfo(noise.dtype).min).neg_()
    noise.log_().neg_()
    noise.add_(logprobs.to(torch.float64))
    return torch.argmax(noise, dim=1)


def sample_seeded_branchless(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Seeded temperature/top-k/top-p sampling without host control flow."""

    vocab = logits.shape[-1]
    do_sample = temperature > 0
    safe_temp = torch.where(do_sample, temperature, torch.ones_like(temperature))
    scores = logits / safe_temp.unsqueeze(1)

    k_active = (top_k > 0) & (top_k < vocab)
    k_clamped = top_k.clamp(min=1, max=vocab)
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    kth = sorted_scores.gather(1, (k_clamped - 1).unsqueeze(1))
    threshold = torch.where(
        k_active.unsqueeze(1), kth, torch.full_like(kth, float("-inf"))
    )
    scores = scores.masked_fill(scores < threshold, float("-inf"))

    p_active = (top_p > 0.0) & (top_p < 1.0)
    sorted_masked = sorted_scores.masked_fill(sorted_scores < threshold, float("-inf"))
    probs_sorted = torch.softmax(sorted_masked, dim=-1)
    cumulative = torch.cumsum(probs_sorted, dim=-1)
    remove = cumulative > top_p.unsqueeze(1)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove = remove & p_active.unsqueeze(1)
    remove_scattered = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, sorted_indices, remove
    )
    scores = scores.masked_fill(remove_scattered, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    sampled = multinomial_with_seed(scores, seeds, positions).view(-1)
    fallback = (~do_sample) | (probs.sum(dim=-1) <= 0)
    return torch.where(fallback, torch.argmax(logits, dim=-1), sampled)


__all__ = [
    "multinomial_with_seed_and_token_ids",
    "sample_seeded_branchless",
    "seeded_gumbel_argmax",
]
