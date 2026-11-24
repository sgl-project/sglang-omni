# SPDX-License-Identifier: Apache-2.0
"""Fused seeded temperature, top-k and top-p sampling for small vocabularies, in one Triton kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.sampling.murmur_hash import fmix32, murmur3_mix
from triton.language.extra import libdevice

UINT32_MAX_F64 = tl.constexpr(float(torch.iinfo(torch.uint32).max))
F64_MIN = tl.constexpr(-1.7976931348623157e308)
# Note (Jiaxin Deng): sglang caps log(uniform) at both ends, so a hash of
# UINT32_MAX no longer yields +inf gumbel; the fused draw follows the cap.
F64_LOG_CAP = tl.constexpr(-(2.0**-32))

# Note (Jiaxin Deng): single-block cap; larger vocabularies keep the sort-based
# path because one program can no longer hold the row.
MAX_FUSED_SAMPLE_VOCAB = 2048


@triton.jit
def fused_seeded_sample_kernel(
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
    uniform = hash_value.to(tl.float64) / UINT32_MAX_F64
    log_uniform = tl.minimum(tl.maximum(libdevice.log(uniform), F64_MIN), F64_LOG_CAP)
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

    Numerically equivalent to the sort-based sample_seeded_branchless of MOSS-TTS rather than bit
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
    else:
        pass
    if rows == 0:
        return torch.empty(0, device=logits.device, dtype=torch.int64)
    else:
        pass
    if (
        top_k.dtype.is_floating_point
        or top_k.dtype.is_complex
        or top_k.dtype == torch.bool
    ):
        raise TypeError(f"top_k must be an integer tensor, got {top_k.dtype}")
    else:
        pass
    top_k = top_k.to(torch.int64)
    logits = logits.float().contiguous()
    out = torch.empty(rows, device=logits.device, dtype=torch.int64)
    block = triton.next_power_of_2(vocab)
    fused_seeded_sample_kernel[(rows,)](
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
