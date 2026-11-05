# SPDX-License-Identifier: Apache-2.0
"""Fused seeded sampling, embedding lookup and frame-local attention kernels."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.sampling.murmur_hash import fmix32, murmur3_mix
from triton.language.extra import libdevice

from sglang_omni.models.moss_tts.sampling_kernels import MAX_FUSED_SAMPLE_VOCAB

_F64_MIN = tl.constexpr(-1.7976931348623157e308)
_F64_LOG_CAP = tl.constexpr(-(2.0**-32))
_UINT32_MAX_F64 = tl.constexpr(4294967295.0)


@triton.jit
def attend_prefix(
    QKV,
    row,
    batch,
    group,
    Embedding,
    K,
    V,
    Out,
    Residual,
    H: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    CAPACITY: tl.constexpr,
    POSITION: tl.constexpr,
    EMBED: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Insert one row's KV and attend over its valid prefix in fp32."""
    h = group * GROUP + tl.arange(0, GROUP)
    d = tl.arange(0, BD)
    dmask = (h[:, None] < HEADS) & (d[None, :] < D)
    offset = row * (3 * H) + h[:, None] * D + d[None, :]
    q = tl.load(QKV + offset, dmask, 0).to(tl.float32)
    k = tl.load(QKV + offset + H, dmask, 0).to(tl.float32)
    v = tl.load(QKV + offset + 2 * H, dmask, 0).to(tl.float32)
    base = (batch * HEADS + h[:, None]) * CAPACITY * D + d[None, :]
    tl.store(K + base + POSITION * D, k, dmask)
    tl.store(V + base + POSITION * D, v, dmask)
    if POSITION == 0:
        result = v
    else:
        s = tl.arange(0, BS)
        scores = tl.full((GROUP, BS), -float("inf"), tl.float32)
        for p in tl.static_range(POSITION):
            kp = tl.load(K + base + p * D, dmask, 0).to(tl.float32)
            sp = tl.sum(kp * q, axis=1) * (D**-0.5)
            scores = tl.where(s[None, :] == p, sp[:, None], scores)
        sp = tl.sum(k * q, axis=1) * (D**-0.5)
        scores = tl.where(s[None, :] == POSITION, sp[:, None], scores)
        probs = tl.exp(scores - tl.max(scores, axis=1)[:, None])
        probs = probs / tl.sum(probs, axis=1)[:, None]
        result = (
            v * tl.sum(tl.where(s[None, :] == POSITION, probs, 0.0), axis=1)[:, None]
        )
        for p in tl.static_range(POSITION):
            vp = tl.load(V + base + p * D, dmask, 0).to(tl.float32)
            result += (
                vp * tl.sum(tl.where(s[None, :] == p, probs, 0.0), axis=1)[:, None]
            )
    tl.store(Out + batch * H + h[:, None] * D + d[None, :], result, dmask)
    if EMBED:
        residual = tl.load(Embedding + row * H + h[:, None] * D + d[None, :], dmask, 0)
        tl.store(Residual + batch * H + h[:, None] * D + d[None, :], residual, dmask)
    else:
        pass


@triton.jit
def sample_seeded_row(
    logits_ptr,
    temperature_ptr,
    top_p_ptr,
    top_k_ptr,
    seeds_ptr,
    positions_ptr,
    row,
    VOCAB: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    POSITION_OFFSET: tl.constexpr,
):
    """Preserve the standalone fused_seeded_sample_kernel's seeded arithmetic."""
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

    # note (Jiaxin Deng): Pack score and inverse index to preserve stable ties.
    # note (Jiaxin Deng): Canonicalize signed zero before sorting.
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

    # note (Dino1844): Hash original token ids so sorting preserves seeded draws.
    seed = tl.load(seeds_ptr + row).to(tl.uint64)
    position = (tl.load(positions_ptr + row) + POSITION_OFFSET).to(tl.uint32)
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

    # note (Jiaxin Deng): Preserve argmax's first-NaN tie rule.
    logit_is_nan = logits != logits
    nan_greedy = tl.min(tl.where(valid & logit_is_nan, idx, 2147483647), axis=0)
    finite = valid & ~logit_is_nan
    max_logit = tl.max(tl.where(finite, logits, -float("inf")), axis=0)
    finite_greedy = tl.min(
        tl.where(finite & (logits == max_logit), idx, 2147483647), axis=0
    )
    greedy = tl.where(nan_greedy != 2147483647, nan_greedy, finite_greedy)
    use_fallback = (~do_sample) | (z <= 0) | (z != z)
    return tl.where(use_fallback, greedy.to(tl.int64), sampled)


@triton.jit
def frame_attention_kernel(
    QKV,
    K,
    V,
    Out,
    H: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    CAPACITY: tl.constexpr,
    POSITION: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Depth-0 attention: batch-indexed QKV rows, no lookup, no residual."""
    batch, group = tl.program_id(0), tl.program_id(1)
    attend_prefix(
        QKV,
        batch,
        batch,
        group,
        QKV,  # unused Embedding pointer (EMBED=False)
        K,
        V,
        Out,
        Out,  # unused Residual pointer (EMBED=False)
        H,
        HEADS,
        D,
        CAPACITY,
        POSITION,
        False,
        BD,
        BS,
        GROUP,
    )


@triton.jit
def seeded_sample_attend_kernel(
    logits_ptr,
    temperature_ptr,
    top_p_ptr,
    top_k_ptr,
    seeds_ptr,
    base_positions_ptr,
    codes_ptr,
    QKV,
    Embedding,
    K,
    V,
    Out,
    Residual,
    CODE_STRIDE: tl.constexpr,
    CHANNEL: tl.constexpr,
    VOCAB: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    H: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    CAPACITY: tl.constexpr,
    POSITION: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
    GROUP: tl.constexpr,
    ATTEND: tl.constexpr,
):
    """Sample one codebook, gather its embedding and optionally attend to QKV."""
    batch, group = tl.program_id(0), tl.program_id(1)
    sampled = sample_seeded_row(
        logits_ptr,
        temperature_ptr,
        top_p_ptr,
        top_k_ptr,
        seeds_ptr,
        base_positions_ptr,
        batch,
        VOCAB,
        ROW_STRIDE,
        BLOCK,
        CHANNEL + 1,
    )
    if group == 0:
        tl.store(codes_ptr + batch * CODE_STRIDE + CHANNEL, sampled)
    else:
        pass
    if ATTEND:
        attend_prefix(
            QKV,
            sampled,
            batch,
            group,
            Embedding,
            K,
            V,
            Out,
            Residual,
            H,
            HEADS,
            D,
            CAPACITY,
            POSITION,
            True,
            BD,
            BS,
            GROUP,
        )
    else:
        h = group * GROUP + tl.arange(0, GROUP)
        d = tl.arange(0, BD)
        dmask = (h[:, None] < HEADS) & (d[None, :] < D)
        residual = tl.load(
            Embedding + sampled * H + h[:, None] * D + d[None, :], dmask, 0
        )
        tl.store(Residual + batch * H + h[:, None] * D + d[None, :], residual, dmask)


def frame_lookup_attention(
    qkv: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position: int,
    *,
    group: int = 8,
    num_warps: int = 4,
) -> torch.Tensor:
    """Depth-0 fused KV insertion and attention over batch QKV rows."""
    batch, heads, capacity, dim = key.shape
    hidden = heads * dim
    out = torch.empty((batch, hidden), dtype=key.dtype, device=key.device)
    frame_attention_kernel[(batch, triton.cdiv(heads, group))](
        qkv,
        key,
        value,
        out,
        hidden,
        heads,
        dim,
        capacity,
        position,
        triton.next_power_of_2(dim),
        triton.next_power_of_2(position + 1),
        group,
        num_warps=num_warps,
    )
    return out


def seeded_sample_attend(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    seeds: torch.Tensor,
    base_positions: torch.Tensor,
    codes: torch.Tensor,
    channel: int,
    qkv: torch.Tensor | None,
    embedding: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position: int,
    group: int = 32,
    num_warps: int = 8,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Sample a codebook and return optional attention plus its embedding."""
    rows, vocab = logits.shape
    if vocab > MAX_FUSED_SAMPLE_VOCAB:
        raise ValueError(
            f"fused seeded sample-attend supports vocab <= {MAX_FUSED_SAMPLE_VOCAB}, got {vocab}"
        )
    else:
        pass
    batch, heads, capacity, dim = key.shape
    hidden = heads * dim
    attend = qkv is not None
    if attend:
        out = torch.empty((rows, hidden), dtype=key.dtype, device=key.device)
    else:
        out = key  # unused pointer (ATTEND=False)
    residual = torch.empty((rows, hidden), dtype=embedding.dtype, device=logits.device)
    if codes.stride(1) != 1:
        raise ValueError("codes must have unit column stride")
    else:
        pass
    seeded_sample_attend_kernel[(rows, triton.cdiv(heads, group))](
        logits,
        temperature,
        top_p,
        top_k,
        seeds,
        base_positions,
        codes,
        qkv if attend else key,
        embedding,
        key,
        value,
        out,
        residual,
        codes.stride(0),
        channel,
        vocab,
        logits.stride(0),
        triton.next_power_of_2(vocab),
        hidden,
        heads,
        dim,
        capacity,
        position,
        triton.next_power_of_2(dim),
        triton.next_power_of_2(position + 1),
        group,
        attend,
        num_warps=num_warps,
    )
    return (out if attend else None, residual)


__all__ = [
    "MAX_FUSED_SAMPLE_VOCAB",
    "frame_lookup_attention",
    "seeded_sample_attend",
]
