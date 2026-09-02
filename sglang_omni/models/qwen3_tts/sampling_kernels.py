# SPDX-License-Identifier: Apache-2.0
"""Optional sampling kernels for Qwen3-TTS."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_GATHER_SUPPORTED = hasattr(tl, "gather")
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None
    _TRITON_GATHER_SUPPORTED = False


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
    def _gumbel_from_hash(h: tl.uint32):
        """Match SGLang multinomial_with_seed float64 endpoint handling.

        SGLang does ``x.log_().clamp_(min=finfo.min, max=-(2**-32)).neg_().log_().neg_()``.
        Clamp after log so hash 0 becomes finfo.min, not a tiny positive u.
        """
        u = h.to(tl.float64) / 4294967295.0
        log_u = tl.log(u)
        log_u = tl.maximum(log_u, -1.7976931348623157e308)
        log_u = tl.minimum(log_u, -2.3283064365386963e-10)
        return -tl.log(-log_u)

    @triton.jit
    def _seeded_gumbel_sample_sorted_kernel(
        logprobs,
        sorted_idx,
        seeds,
        positions,
        out,
        num_cols: tl.constexpr,
        logprobs_stride_b: tl.constexpr,
        logprobs_stride_k: tl.constexpr,
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

        gumbel = _gumbel_from_hash(h)
        weights = tl.load(
            logprobs + row * logprobs_stride_b + offsets * logprobs_stride_k,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float64)
        scores = tl.where(mask, weights + gumbel, -float("inf"))
        max_score = tl.max(scores, axis=0)
        candidates = tl.where(scores == max_score, offsets, num_cols)
        rank = tl.min(candidates, axis=0)
        token = tl.load(sorted_idx + row * idx_stride_b + rank * idx_stride_k)
        tl.store(out + row, token)

    @triton.jit
    def _bitonic_compare_selected_32_desc(
        scores,
        token_ids,
        valid,
        offsets,
        stride: tl.constexpr,
        network_size: tl.constexpr,
        final_round: tl.constexpr,
    ):
        """One compare-exchange round of PyTorch's small descending sort.

        Under the repository's ``torch==2.13.0`` pin,
        ``torch.topk(..., sorted=True)`` first gathers the selected entries and
        then uses this unstable 32-entry bitonic network for k <= 32. The
        equality behavior is observable by the seeded sampler, so a regular
        lexicographic sort would not be a compatible replacement. The CUDA
        parity tests compare this network with ``torch.topk`` so a PyTorch
        upgrade fails before this implementation can silently diverge.
        """
        partner_offsets = offsets ^ stride
        partner_scores = tl.gather(scores, partner_offsets, axis=0)
        partner_token_ids = tl.gather(token_ids, partner_offsets, axis=0)
        partner_valid = tl.gather(valid, partner_offsets, axis=0)

        is_right = (offsets & stride) != 0
        left_scores = tl.where(is_right, partner_scores, scores)
        right_scores = tl.where(is_right, scores, partner_scores)
        left_valid = tl.where(is_right, partner_valid, valid)
        right_valid = tl.where(is_right, valid, partner_valid)

        if final_round:
            direction = offsets != offsets
        else:
            left_offsets = offsets - tl.where(is_right, stride, 0)
            thread_ids = (left_offsets // (2 * stride)) * stride + (
                left_offsets % stride
            )
            direction = (thread_ids & (network_size // 2)) != 0

        should_swap = ((left_scores > right_scores) & left_valid) | (right_valid == 0)
        take_partner = should_swap == direction
        return (
            tl.where(take_partner, partner_scores, scores),
            tl.where(take_partner, partner_token_ids, token_ids),
            tl.where(take_partner, partner_valid, valid),
        )

    @triton.jit
    def _bitonic_sort_selected_32_desc(scores, token_ids, max_top_k: tl.constexpr):
        """Match the CUDA ``SmallBitonicSort`` network used by torch.topk."""
        offsets = tl.arange(0, 32)
        valid = offsets < max_top_k

        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=1,
            network_size=2,
            final_round=False,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=2,
            network_size=4,
            final_round=False,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=1,
            network_size=4,
            final_round=False,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=4,
            network_size=8,
            final_round=False,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=2,
            network_size=8,
            final_round=False,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=1,
            network_size=8,
            final_round=False,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=8,
            network_size=16,
            final_round=False,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=4,
            network_size=16,
            final_round=False,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=2,
            network_size=16,
            final_round=False,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=1,
            network_size=16,
            final_round=False,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=16,
            network_size=32,
            final_round=True,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=8,
            network_size=32,
            final_round=True,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=4,
            network_size=32,
            final_round=True,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=2,
            network_size=32,
            final_round=True,
        )
        scores, token_ids, valid = _bitonic_compare_selected_32_desc(
            scores,
            token_ids,
            valid,
            offsets,
            stride=1,
            network_size=32,
            final_round=True,
        )
        return scores, token_ids

    @triton.jit
    def _seeded_top_k_top_p_sample_kernel(
        logits,
        temperatures,
        top_ks,
        top_ps,
        seeds,
        positions,
        out,
        logits_stride_b: tl.constexpr,
        max_top_k: tl.constexpr,
        block_k: tl.constexpr,
        has_top_p: tl.constexpr,
    ):
        row = tl.program_id(0)
        vocab_offsets = tl.arange(0, 2048)
        scores = tl.load(logits + row * logits_stride_b + vocab_offsets).to(tl.float32)
        temperature = tl.maximum(tl.load(temperatures + row).to(tl.float32), 1e-5)
        scores = scores / temperature

        # Note (Jun Liu): PyTorch's top-k gather chooses lower input indices at
        # its threshold.
        # The packed order is score descending, then index ascending. We unpack
        # the selected FP32 score exactly rather than reloading it from logits.
        score_bits = scores.to(tl.uint32, bitcast=True)
        one_vocab = tl.full(vocab_offsets.shape, 1, tl.uint32)
        high_bit_vocab = one_vocab << 31
        all_ones_vocab = high_bit_vocab | (high_bit_vocab - one_vocab)
        ordered_score_bits = score_bits ^ tl.where(
            (score_bits >> 31) != 0,
            all_ones_vocab,
            high_bit_vocab,
        )
        packed = (ordered_score_bits.to(tl.uint64) << 32) | (
            all_ones_vocab - vocab_offsets.to(tl.uint32)
        ).to(tl.uint64)
        top_packed = tl.topk(packed, k=block_k)

        ranks = tl.arange(0, block_k)
        one_rank = tl.full(ranks.shape, 1, tl.uint32)
        high_bit_rank = one_rank << 31
        all_ones_rank = high_bit_rank | (high_bit_rank - one_rank)
        ordered_top_score_bits = (top_packed >> 32).to(tl.uint32)
        top_score_bits = ordered_top_score_bits ^ tl.where(
            (ordered_top_score_bits >> 31) != 0,
            high_bit_rank,
            all_ones_rank,
        )
        sorted_scores = top_score_bits.to(tl.float32, bitcast=True)
        sorted_token_ids = all_ones_rank - (
            top_packed & all_ones_rank.to(tl.uint64)
        ).to(tl.uint32)

        if max_top_k <= 32:
            # Note (Jun Liu): gatherTopK first writes every score strictly above
            # the threshold
            # in source-index order. It then appends threshold-equal scores in
            # source-index order. Its following 32-entry bitonic sort is
            # unstable, so reproducing only the final top-k membership is not
            # sufficient for seeded sampling.
            threshold_ordered_score_bits = tl.max(
                tl.where(ranks == max_top_k - 1, ordered_top_score_bits, 0),
                axis=0,
            )
            # Note (Jun Liu): CUDA's threshold gather compares the float rank
            # representation.
            # In particular, it selects positive zero ahead of negative zero.
            # A numerical score comparison would collapse that distinction and
            # could change the seeded codec ID.
            greater_than_threshold = ordered_score_bits > threshold_ordered_score_bits
            equal_to_threshold = ordered_score_bits == threshold_ordered_score_bits
            gather_order_keys = (
                tl.where(
                    greater_than_threshold,
                    0,
                    tl.where(equal_to_threshold, 1, all_ones_vocab),
                ).to(tl.uint64)
                << 32
            ) | vocab_offsets.to(tl.uint64)
            # Note (Jun Liu): ``tl.topk`` only supports descending selection.
            # Complementing
            # the unsigned key turns its descending order into the ascending
            # gather order used by PyTorch's top-k implementation.
            all_ones_key = (all_ones_vocab.to(tl.uint64) << 32) | all_ones_vocab.to(
                tl.uint64
            )
            gathered_complement = tl.topk(
                all_ones_key - gather_order_keys,
                k=32,
            )
            gather_source_ids = (
                all_ones_rank
                - (gathered_complement & all_ones_rank.to(tl.uint64)).to(tl.uint32)
            ).to(tl.int32)
            sorted_scores = tl.gather(
                scores,
                gather_source_ids,
                axis=0,
            )
            sorted_token_ids = gather_source_ids.to(tl.uint32)
            sorted_scores, sorted_token_ids = _bitonic_sort_selected_32_desc(
                sorted_scores, sorted_token_ids, max_top_k
            )

        keep_top_k = ranks < tl.load(top_ks + row)
        masked_scores = tl.where(keep_top_k, sorted_scores, -float("inf"))
        max_score = tl.max(masked_scores, axis=0)
        probs = tl.exp(masked_scores - max_score)
        probs = probs / tl.sum(probs, axis=0)

        if has_top_p:
            top_p = tl.load(top_ps + row).to(tl.float32)
            active_top_p = (top_p > 0.0) & (top_p < 1.0)
            cdf = tl.cumsum(probs, axis=0)
            remove = (cdf - probs >= top_p) & active_top_p
            remove = remove & (ranks != 0)
            keep_top_k = keep_top_k & ~remove

        logprobs = tl.where(keep_top_k, tl.log(probs), -float("inf"))

        seed = tl.load(seeds + row).to(tl.uint64)
        pos = tl.load(positions + row).to(tl.uint32)
        col = ranks.to(tl.uint32)

        h: tl.uint32 = 0
        h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
        h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
        h = _murmur3_mix(h, pos)
        h = _murmur3_mix(h, col)
        h ^= 16
        h = _fmix32(h)

        gumbel = _gumbel_from_hash(h)
        sampled_scores = logprobs.to(tl.float64) + gumbel
        max_sampled_score = tl.max(sampled_scores, axis=0)
        candidates = tl.where(sampled_scores == max_sampled_score, ranks, block_k)
        sampled_rank = tl.min(candidates, axis=0)
        token = tl.max(
            tl.where(
                ranks == sampled_rank,
                sorted_token_ids.to(tl.int64),
                0,
            ),
            axis=0,
        )
        tl.store(out + row, token)

else:
    _seeded_gumbel_sample_sorted_kernel = None
    _bitonic_compare_selected_32_desc = None
    _bitonic_sort_selected_32_desc = None
    _seeded_top_k_top_p_sample_kernel = None


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def sample_from_sorted_logprobs_with_seed_small_k(
    logprobs: torch.Tensor,
    sorted_idx: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor | None:
    if (
        _seeded_gumbel_sample_sorted_kernel is None
        or not logprobs.is_cuda
        or not sorted_idx.is_cuda
        or not seeds.is_cuda
        or not positions.is_cuda
    ):
        return None
    if logprobs.ndim != 2 or sorted_idx.shape != logprobs.shape:
        return None
    if seeds.ndim != 1 or positions.ndim != 1:
        return None
    batch_size, num_cols = logprobs.shape
    if batch_size == 0:
        return torch.empty((0,), device=logprobs.device, dtype=torch.long)
    if seeds.shape[0] != batch_size or positions.shape[0] != batch_size:
        return None
    if num_cols <= 0 or num_cols > 1024:
        return None

    block_size = _next_power_of_2(num_cols)
    out = torch.empty((batch_size,), device=logprobs.device, dtype=torch.long)
    _seeded_gumbel_sample_sorted_kernel[(batch_size,)](
        logprobs,
        sorted_idx,
        seeds,
        positions,
        out,
        int(num_cols),
        logprobs.stride(0),
        logprobs.stride(1),
        sorted_idx.stride(0),
        sorted_idx.stride(1),
        block_size,
    )
    return out


_FUSED_RAW_LOGIT_TOP_KS = frozenset((4, 8, 16, 32, 50, 64, 128, 256, 512, 1024))


def _fused_raw_logit_block_k(max_top_k: int) -> int | None:
    """Return the power-of-two Triton selection width for a graph signature."""
    if max_top_k not in _FUSED_RAW_LOGIT_TOP_KS:
        return None
    if max_top_k <= 32:
        # Note (Jun Liu): PyTorch uses a fixed 32-entry bitonic network for all
        # these widths.
        return 32
    return _next_power_of_2(max_top_k)


def sample_from_logits_with_seed_top_k_top_p(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    *,
    max_top_k: int,
    has_top_p: bool,
) -> torch.Tensor | None:
    """Fuse Qwen3-TTS's bounded seeded sampling path when its contract fits.

    The caller owns the graph signature. This function deliberately returns
    ``None`` for any unproven shape or layout so the production reference
    remains the fallback. It does not inspect device values because that would
    introduce a host synchronization during CUDA graph replay.
    """
    block_k = _fused_raw_logit_block_k(int(max_top_k))
    if (
        _seeded_top_k_top_p_sample_kernel is None
        or not _TRITON_GATHER_SUPPORTED
        or block_k is None
        or not logits.is_cuda
        or logits.ndim != 2
        or logits.shape[1] != 2048
        or logits.dtype is not torch.bfloat16
        or not logits.is_contiguous()
    ):
        return None

    batch_size = int(logits.shape[0])
    if batch_size == 0:
        return torch.empty((0,), device=logits.device, dtype=torch.long)

    row_tensors = (temperatures, top_ks, top_ps, seeds, positions)
    if any(
        tensor.device != logits.device
        or tensor.ndim != 1
        or tensor.shape[0] != batch_size
        or not tensor.is_contiguous()
        for tensor in row_tensors
    ):
        return None
    if (
        temperatures.dtype is not torch.float32
        or top_ks.dtype is not torch.long
        or top_ps.dtype is not torch.float32
        or seeds.dtype is not torch.long
        or positions.dtype is not torch.long
    ):
        return None

    out = torch.empty((batch_size,), device=logits.device, dtype=torch.long)
    _seeded_top_k_top_p_sample_kernel[(batch_size,)](
        logits,
        temperatures,
        top_ks,
        top_ps,
        seeds,
        positions,
        out,
        logits.stride(0),
        int(max_top_k),
        int(block_k),
        bool(has_top_p),
        num_warps=8,
    )
    return out
