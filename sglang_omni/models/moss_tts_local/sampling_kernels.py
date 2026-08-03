# SPDX-License-Identifier: Apache-2.0
"""Optional fused sampling kernels for MOSS-TTS-Local."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    from sglang_omni.sampling.triton.hash import murmur_hash_seed_position_key
except ImportError:  # pragma: no cover - depends on the runtime image
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _sample_seeded_full_vocab_kernel(
        logits,
        temperatures,
        top_ps,
        top_ks,
        seeds,
        positions,
        out,
        vocab_size: tl.constexpr,
        logits_stride_b: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        valid_col = offsets < vocab_size

        raw_scores = tl.load(
            logits + row * logits_stride_b + offsets,
            mask=valid_col,
            other=-float("inf"),
        ).to(tl.float32)
        temperature = tl.load(temperatures + row).to(tl.float32)
        do_sample = temperature > 0.0
        safe_temperature = tl.where(do_sample, temperature, 1.0)
        scores = raw_scores / safe_temperature

        # Match the eager threshold form of top-k: values tied with the kth
        # score remain eligible, even when that retains more than k tokens.
        sorted_scores = tl.sort(scores, dim=0, descending=True)
        top_k = tl.load(top_ks + row)
        active_top_k = (top_k > 0) & (top_k < vocab_size)
        clamped_top_k = tl.minimum(tl.maximum(top_k, 1), vocab_size)
        kth_score = tl.sum(
            tl.where(offsets == clamped_top_k - 1, sorted_scores, 0.0),
            axis=0,
        )
        top_k_threshold = tl.where(active_top_k, kth_score, -float("inf"))
        keep_top_k = valid_col & (scores >= top_k_threshold)

        # Compute nucleus probabilities in sorted order after top-k. The eager
        # path shifts its removal mask right, so rank i is retained while the
        # cumulative probability through rank i-1 is <= top_p.
        sorted_masked = tl.where(
            sorted_scores >= top_k_threshold,
            sorted_scores,
            -float("inf"),
        )
        max_score = tl.max(sorted_masked, axis=0)
        exp_scores = tl.exp(sorted_masked - max_score)
        probs = exp_scores / tl.sum(exp_scores, axis=0)
        cumulative_before = tl.cumsum(probs, axis=0) - probs
        top_p = tl.load(top_ps + row).to(tl.float32)
        active_top_p = (top_p > 0.0) & (top_p < 1.0)
        keep_sorted_top_p = (
            (offsets == 0) | (cumulative_before <= top_p) | (~active_top_p)
        )
        top_p_threshold = tl.min(
            tl.where(
                keep_sorted_top_p & (sorted_masked != -float("inf")),
                sorted_masked,
                float("inf"),
            ),
            axis=0,
        )
        keep = keep_top_k & ((scores >= top_p_threshold) | (~active_top_p))

        seed = tl.load(seeds + row)
        position = tl.load(positions + row)
        hashes = murmur_hash_seed_position_key(seed, position, offsets)
        u = hashes.to(tl.float64) / 4294967295.0
        u = tl.maximum(u, 2.2250738585072014e-308)
        gumbel = -tl.log(-tl.log(u))
        sampled_scores = tl.where(
            keep,
            scores.to(tl.float64) + gumbel,
            -float("inf"),
        )
        sampled_token = tl.argmax(sampled_scores, axis=0, tie_break_left=True)
        greedy_token = tl.argmax(raw_scores, axis=0, tie_break_left=True)

        allowed_max = tl.max(
            tl.where(keep, scores, -float("inf")),
            axis=0,
        )
        valid_distribution = (
            (allowed_max == allowed_max)
            & (allowed_max != -float("inf"))
            & (allowed_max != float("inf"))
        )
        token = tl.where(do_sample & valid_distribution, sampled_token, greedy_token)
        tl.store(out + row, token)

else:
    _sample_seeded_full_vocab_kernel = None


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def fused_sampler_available() -> bool:
    return _sample_seeded_full_vocab_kernel is not None


def sample_seeded_full_vocab(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor | None:
    """Sample ``[batch, vocab]`` logits, or return ``None`` for eager fallback."""

    tensors = (logits, temperatures, top_ps, top_ks, seeds, positions)
    if _sample_seeded_full_vocab_kernel is None or any(
        not tensor.is_cuda for tensor in tensors
    ):
        return None
    if logits.ndim != 2:
        return None
    batch_size, vocab_size = logits.shape
    if any(tensor.ndim != 1 for tensor in tensors[1:]):
        return None
    if any(int(tensor.shape[0]) != batch_size for tensor in tensors[1:]):
        return None
    if batch_size == 0:
        return torch.empty(0, dtype=torch.long, device=logits.device)
    if vocab_size <= 0 or vocab_size > 1024:
        return None
    if any(tensor.device != logits.device for tensor in tensors[1:]):
        return None
    if any(not tensor.is_contiguous() for tensor in tensors):
        return None

    block_size = _next_power_of_2(vocab_size)
    out = torch.empty(batch_size, dtype=torch.long, device=logits.device)
    _sample_seeded_full_vocab_kernel[(batch_size,)](
        logits,
        temperatures,
        top_ps,
        top_ks,
        seeds,
        positions,
        out,
        int(vocab_size),
        logits.stride(0),
        block_size,
    )
    return out


__all__ = [
    "fused_sampler_available",
    "sample_seeded_full_vocab",
]
