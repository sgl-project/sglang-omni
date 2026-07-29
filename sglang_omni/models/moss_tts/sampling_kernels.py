# SPDX-License-Identifier: Apache-2.0
"""Optional fused sampling kernels for MOSS-TTS."""

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
    def _sample_two_candidates_kernel(
        logits,
        temperatures,
        top_ps,
        top_ks,
        seeds,
        positions,
        token_ids,
        out,
        logits_stride_b: tl.constexpr,
    ):
        row = tl.program_id(0)
        # load one row
        logit0 = tl.load(logits + row * logits_stride_b).to(tl.float32)
        logit1 = tl.load(logits + row * logits_stride_b + 1).to(tl.float32)
        temperature = tl.load(temperatures + row).to(tl.float32)
        top_p = tl.load(top_ps + row).to(tl.float32)
        top_k = tl.load(top_ks + row)

        # Fallback for temperature <= 0 or probs are not valid
        # fallback = (~do_sample) | (probs.sum(dim=-1) <= 0) in eager version
        greedy = tl.where(logit1 > logit0, 1, 0)
        valid0 = (logit0 == logit0) & (logit0 != -float("inf"))
        valid1 = (logit1 == logit1) & (logit1 != -float("inf"))
        no_pos_inf = (logit0 != float("inf")) & (logit1 != float("inf"))
        has_value = valid0 | valid1
        do_sample = (temperature > 0.0) & has_value & no_pos_inf
        safe_temperature = tl.where(do_sample, temperature, 1.0)
        score0 = logit0 / safe_temperature
        score1 = logit1 / safe_temperature

        # top_k <= 0 and top_k >= vocab leave the row unrestricted.
        top1 = top_k == 1
        remove0_top_k = top1 & (score0 < score1)
        remove1_top_k = top1 & (score1 < score0)
        score0 = tl.where(remove0_top_k, -float("inf"), score0)
        score1 = tl.where(remove1_top_k, -float("inf"), score1)

        # Mask the lower-scored candidate exactly
        # when the leading candidate’s probability exceeds top_p
        active_top_p = (top_p > 0.0) & (top_p < 1.0)
        max_score = tl.maximum(score0, score1)
        exp0 = tl.exp(score0 - max_score)
        exp1 = tl.exp(score1 - max_score)
        leading_prob = tl.maximum(exp0, exp1) / (exp0 + exp1)
        remove_lower = active_top_p & (leading_prob > top_p)
        remove0_top_p = remove_lower & (score0 < score1)
        remove1_top_p = remove_lower & (score1 <= score0)
        score0 = tl.where(remove0_top_p, -float("inf"), score0)
        score1 = tl.where(remove1_top_p, -float("inf"), score1)

        seed = tl.load(seeds + row)
        position = tl.load(positions + row)
        token0 = tl.load(token_ids)
        token1 = tl.load(token_ids + 1)
        hash0 = murmur_hash_seed_position_key(seed, position, token0)
        hash1 = murmur_hash_seed_position_key(seed, position, token1)

        # Sample Gumbel noise and add to the scores
        # noise = -log(-log(u))
        denom = 4294967295.0
        u0 = hash0.to(tl.float64) / denom
        u1 = hash1.to(tl.float64) / denom
        gumbel0 = tl.where(hash0 == 0, -709.782712893384, -tl.log(-tl.log(u0)))
        gumbel1 = tl.where(hash1 == 0, -709.782712893384, -tl.log(-tl.log(u1)))
        sampled0 = score0.to(tl.float64) + gumbel0
        sampled1 = score1.to(tl.float64) + gumbel1
        sampled = tl.where(sampled1 > sampled0, 1, 0)
        local_token = tl.where(do_sample, sampled, greedy)
        # output mapped id
        tl.store(out + row, tl.where(local_token == 0, token0, token1))

else:
    _sample_two_candidates_kernel = None


def sample_two_candidates(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor | None:
    """Sample two fixed vocabulary candidates, or return ``None`` to fall back."""

    tensors = (logits, temperatures, top_ps, top_ks, seeds, positions, token_ids)
    if _sample_two_candidates_kernel is None or any(
        not tensor.is_cuda for tensor in tensors
    ):
        return None
    if logits.ndim != 2 or tuple(logits.shape[1:]) != (2,):
        return None
    batch_size = int(logits.shape[0])
    if any(tensor.ndim != 1 for tensor in tensors[1:]):
        return None
    if any(int(tensor.shape[0]) != batch_size for tensor in tensors[1:-1]):
        return None
    if int(token_ids.shape[0]) != 2:
        return None
    if batch_size == 0:
        return torch.empty(0, dtype=torch.long, device=logits.device)
    if any(tensor.device != logits.device for tensor in tensors[1:]):
        return None
    if any(not tensor.is_contiguous() for tensor in tensors):
        return None

    out = torch.empty(batch_size, dtype=torch.long, device=logits.device)
    _sample_two_candidates_kernel[(batch_size,)](
        logits,
        temperatures,
        top_ps,
        top_ks,
        seeds,
        positions,
        token_ids,
        out,
        logits.stride(0),
    )
    return out
