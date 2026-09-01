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
    # Clamp both ends like sglang's multinomial_with_seed: a uniform of exactly
    # 1 would give +inf noise and a NaN score against a -inf logprob, so the
    # top bucket is capped at the hash spacing (2 ** -32).
    log_uniform = tl.minimum(
        tl.maximum(log_uniform, -1.7976931348623157e308), -2.3283064365386963e-10
    )
    gumbel = -libdevice.log(-log_uniform)
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
    noise.log_().clamp_(min=torch.finfo(noise.dtype).min, max=-(2.0**-32)).neg_()
    noise.log_().neg_()
    noise.add_(logprobs.to(torch.float64))
    return torch.argmax(noise, dim=1)


_F64_MIN = tl.constexpr(-1.7976931348623157e308)
# Note (Jiaxin Deng): sglang caps log(uniform) at both ends, so a hash of
# UINT32_MAX no longer yields +inf gumbel; the fused draw follows the cap.
_F64_LOG_CAP = tl.constexpr(-(2.0**-32))

# Note (Jiaxin Deng): single-block cap; larger vocabularies keep the sort-based
# path because one program can no longer hold the row.
MAX_FUSED_SAMPLE_VOCAB = 2048


@triton.jit
def _fused_seeded_sample_kernel(
    logits_ptr,
    temperature_ptr,
    top_p_ptr,
    top_k_ptr,
    seeds_ptr,
    positions_ptr,
    output_ptr,
    VOCAB: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    valid = idx < VOCAB

    logits = tl.load(
        logits_ptr + row * ROW_STRIDE + idx, mask=valid, other=-float("inf")
    ).to(tl.float32)

    temp = tl.load(temperature_ptr + row).to(tl.float32)
    top_p = tl.load(top_p_ptr + row).to(tl.float32)
    top_k = tl.load(top_k_ptr + row).to(tl.int64)
    do_sample = temp > 0
    safe_temp = tl.where(do_sample, temp, 1.0)
    scores = logits / safe_temp

    # Note (Jiaxin Deng): key packs orderable score bits above a complemented
    # index so equal scores keep input order, matching cub's stable radix sort
    # (torch.sort) bit-for-bit at nucleus tie boundaries. -0.0 canonicalizes to
    # +0.0 first: the zeros compare equal numerically, so bit-distinct keys
    # would break that input-order tie rule.
    key_scores = tl.where(scores == 0.0, 0.0, scores)
    bits = key_scores.to(tl.int32, bitcast=True)
    orderable = (bits ^ ((bits >> 31) | -2147483648)).to(tl.uint32).to(tl.int64)
    biased = orderable - tl.full((), 2147483648, tl.int64)
    idx64 = idx.to(tl.int64)
    inv_idx = tl.full((), 4294967295, tl.int64) - idx64
    key = (biased << 32) + inv_idx
    key = tl.where(valid, key, tl.full(key.shape, -9223372036854775807, tl.int64))
    skey = tl.sort(key, descending=True)

    s_idx = tl.full((), 4294967295, tl.int64) - (skey - ((skey >> 32) << 32))
    s_bits_orderable = (skey >> 32) + tl.full((), 2147483648, tl.int64)
    s_bits = s_bits_orderable.to(tl.int32)
    s_bits = s_bits ^ (((~s_bits) >> 31) | -2147483648)
    s_scores = s_bits.to(tl.float32, bitcast=True)
    lane = tl.arange(0, BLOCK)
    s_valid = lane < VOCAB
    s_scores = tl.where(s_valid, s_scores, -float("inf"))

    k_active = (top_k > 0) & (top_k < VOCAB)
    k_clamped = tl.minimum(tl.maximum(top_k, 1), VOCAB)
    kth = tl.sum(tl.where(lane.to(tl.int64) == k_clamped - 1, s_scores, 0.0), axis=0)
    threshold = tl.where(k_active, kth, -float("inf"))
    masked_sorted = tl.where(s_scores < threshold, -float("inf"), s_scores)

    p_active = (top_p > 0.0) & (top_p < 1.0)
    row_max = tl.max(masked_sorted, axis=0)
    finite_max = row_max > -float("inf")
    # Note (Jiaxin Deng): masking on == -inf lets NaN and +inf lanes poison z the
    # way torch.softmax poisons the baseline row, so both fall back identically.
    exp_term = tl.where(
        masked_sorted == -float("inf"),
        0.0,
        tl.exp(masked_sorted - tl.where(finite_max, row_max, 0.0)),
    )
    z = tl.sum(exp_term, axis=0)
    probs_sorted = tl.where(z > 0, exp_term / z, 0.0)
    inclusive = tl.cumsum(probs_sorted, axis=0)
    remove = ((inclusive - probs_sorted) > top_p) & p_active
    final_sorted = tl.where(remove, -float("inf"), masked_sorted)

    # Match multinomial_with_seed exactly, including hash_value == UINT32_MAX;
    # the hash keys on original token ids so lane order is irrelevant.
    seed = tl.load(seeds_ptr + row).to(tl.uint64)
    position = tl.load(positions_ptr + row).to(tl.uint32)
    hash_value: tl.uint32 = 0
    hash_value = murmur3_mix(hash_value, (seed & 0xFFFFFFFF).to(tl.uint32))
    hash_value = murmur3_mix(hash_value, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    hash_value = murmur3_mix(hash_value, position)
    hash_value = murmur3_mix(hash_value, s_idx.to(tl.uint32))
    hash_value ^= 16
    hash_value = fmix32(hash_value)
    uniform = hash_value.to(tl.float64) / _UINT32_MAX_F64
    log_uniform = tl.minimum(tl.maximum(libdevice.log(uniform), _F64_MIN), _F64_LOG_CAP)
    gumbel = -libdevice.log(-log_uniform)
    value = final_sorted.to(tl.float64) + gumbel

    is_nan = value != value
    nan_index = tl.min(tl.where(s_valid & is_nan, s_idx, 2147483647), axis=0)
    non_nan = tl.where(s_valid & ~is_nan, value, -float("inf"))
    max_value = tl.max(non_nan, axis=0)
    max_index = tl.min(
        tl.where(s_valid & ~is_nan & (non_nan == max_value), s_idx, 2147483647),
        axis=0,
    )
    sampled = tl.where(nan_index != 2147483647, nan_index, max_index)

    # Note (Jiaxin Deng): torch.argmax ranks NaN above every number, so a NaN row
    # must select its first NaN lane; comparing against a NaN max matches none.
    logit_is_nan = logits != logits
    nan_greedy = tl.min(tl.where(valid & logit_is_nan, idx, 2147483647), axis=0)
    finite = valid & ~logit_is_nan
    max_logit = tl.max(tl.where(finite, logits, -float("inf")), axis=0)
    finite_greedy = tl.min(
        tl.where(finite & (logits == max_logit), idx, 2147483647), axis=0
    )
    greedy = tl.where(nan_greedy != 2147483647, nan_greedy, finite_greedy)
    use_fallback = (~do_sample) | (z <= 0) | (z != z)
    result = tl.where(use_fallback, greedy.to(tl.int64), sampled)
    tl.store(output_ptr + row, result)


def sample_seeded_fused(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Single-kernel seeded sampler for vocab <= 2048.

    Numerically equivalent to :func:`sample_seeded_branchless` rather than bit
    identical by construction: the in-kernel softmax and cumsum reduce in a
    different order than aten, so at an exact nucleus boundary the kept set may
    differ by the boundary token. The tie order (input order), the greedy and
    NaN fallbacks, and the seeded draw are exact.
    """

    rows, vocab = logits.shape
    if vocab > MAX_FUSED_SAMPLE_VOCAB:
        raise ValueError(
            f"fused seeded sampler supports vocab <= {MAX_FUSED_SAMPLE_VOCAB}, got {vocab}"
        )
    if rows == 0:
        return torch.empty(0, device=logits.device, dtype=torch.int64)
    if (
        top_k.dtype.is_floating_point
        or top_k.dtype.is_complex
        or top_k.dtype == torch.bool
    ):
        raise TypeError(f"top_k must be an integer tensor, got {top_k.dtype}")
    top_k = top_k.to(torch.int64)
    logits = logits.float().contiguous()
    out = torch.empty(rows, device=logits.device, dtype=torch.int64)
    block = triton.next_power_of_2(vocab)
    _fused_seeded_sample_kernel[(rows,)](
        logits,
        temperature.float().contiguous(),
        top_p.float().contiguous(),
        top_k.contiguous(),
        seeds.to(torch.int64).contiguous(),
        positions.to(torch.int64).contiguous(),
        out,
        VOCAB=vocab,
        ROW_STRIDE=logits.stride(0),
        BLOCK=block,
        num_warps=8 if block >= 1024 else 4,
    )
    return out


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
    # Note (Jiaxin Deng): an unstable sort leaves the tie order backend dependent
    # (a two lane row reverses on cu130) and that picks the token at a nucleus edge.
    sorted_scores, sorted_indices = torch.sort(
        scores, descending=True, dim=-1, stable=True
    )
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
    "MAX_FUSED_SAMPLE_VOCAB",
    "multinomial_with_seed_and_token_ids",
    "sample_seeded_branchless",
    "sample_seeded_fused",
    "seeded_gumbel_argmax",
]
