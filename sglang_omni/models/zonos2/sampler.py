# SPDX-License-Identifier: Apache-2.0
"""Per-codebook TTS sampler with per-request sampling params.

Order: repetition penalty, temperature, top-k, top-p, min-p, then ``torch.multinomial``.
Each request applies its own temperature/top_k/top_p/min_p/repetition_penalty
(vectorized per row, no per-request Python loop). The draw uses torch.multinomial
(the default RNG): a per-request seeded hash-Gumbel draw was tried but regressed
ZH CER (~9.7 -> ~12.6, catastrophic samples doubled), so it was reverted. Rows
with temperature <= 0 or left fully masked fall back to greedy argmax. Logits
arrive soft-capped."""

from __future__ import annotations

from typing import Optional

import torch

_NEG_INF = float("-inf")


def _apply_repetition_penalty(
    logits: torch.Tensor,
    rep_token_ids: Optional[torch.Tensor],
    penalties: torch.Tensor,
) -> torch.Tensor:
    """Penalize per-codebook repeats. logits (B,C,V), rep_token_ids (B,C,W),
    penalties (B,) -- one penalty strength per request."""
    if rep_token_ids is None or rep_token_ids.numel() == 0:
        return logits
    B, C, V = logits.shape
    safe = rep_token_ids.clamp(min=0, max=V - 1).long()
    valid = (rep_token_ids >= 0) & (rep_token_ids < V)
    counts = torch.zeros((B, C, V), dtype=torch.int32, device=logits.device)
    counts.scatter_add_(-1, safe, valid.to(torch.int32))
    repeated = counts > 0
    pen = penalties.view(B, 1, 1).clamp(min=1.0)
    adjusted = torch.where(logits > 0, logits / pen, logits * pen)
    return torch.where(repeated, adjusted, logits)


def _apply_top_k(
    scores: torch.Tensor, top_k_row: torch.Tensor, max_top_k: int
) -> torch.Tensor:
    """Per-row top-k mask; rows with k <= 0 or k >= vocab are left untouched.
    ``max_top_k`` is the largest active k, supplied by the caller from the host
    params so this stays sync-free (no ``.item()`` stall in the decode loop)."""
    vocab = scores.shape[-1]
    active = (top_k_row > 0) & (top_k_row < vocab)
    k_clamped = top_k_row.clamp(min=1, max=vocab)
    topk_scores, _ = torch.topk(scores, k=max_top_k, dim=-1)
    gather_k = torch.where(active, k_clamped, torch.ones_like(k_clamped))
    gather_k = gather_k.clamp(min=1, max=max_top_k)
    kth = topk_scores.gather(1, (gather_k - 1).unsqueeze(1))
    threshold = torch.where(active.unsqueeze(1), kth, torch.full_like(kth, _NEG_INF))
    return scores.masked_fill(scores < threshold, _NEG_INF)


def _apply_top_p(scores: torch.Tensor, top_p_row: torch.Tensor) -> torch.Tensor:
    """Per-row nucleus mask; rows with p <= 0 or p >= 1 are left untouched."""
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    probs = torch.softmax(sorted_scores, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative > top_p_row.unsqueeze(1)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove = remove & ((top_p_row > 0.0) & (top_p_row < 1.0)).unsqueeze(1)
    remove_scattered = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, sorted_indices, remove
    )
    return scores.masked_fill(remove_scattered, _NEG_INF)


def _apply_min_p(scores: torch.Tensor, min_p_row: torch.Tensor) -> torch.Tensor:
    """Per-row min-p mask (relative to the row's max prob); min_p <= 0 untouched."""
    probs = torch.softmax(scores, dim=-1)
    top = probs.max(dim=-1, keepdim=True).values
    remove = probs < (min_p_row.clamp(min=0.0).unsqueeze(1) * top)
    return scores.masked_fill(remove, _NEG_INF)


@torch.no_grad()
def sample_tts(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    top_k_max: int,
    rep_token_ids: Optional[torch.Tensor] = None,
    any_top_p: bool = True,
    any_min_p: bool = True,
) -> torch.Tensor:
    """logits (B, C, V) soft-capped -> sampled codes (B, C) int64.

    temperature/top_k/top_p/min_p/repetition_penalty are per-request (B,) tensors.
    ``top_k_max`` is the largest active top-k across the batch (host int; 0 disables
    top-k). ``any_top_p``/``any_min_p`` are host flags to skip those passes when no
    in-flight request uses them."""
    B, C, V = logits.shape
    device = logits.device
    logits = _apply_repetition_penalty(logits, rep_token_ids, repetition_penalty)

    def _rows(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return t.to(device=device, dtype=dtype).repeat_interleave(C)

    temp_row = _rows(temperature, torch.float32)
    do_sample = temp_row > 0
    safe_temp = torch.where(do_sample, temp_row, torch.ones_like(temp_row))
    scores = logits.reshape(B * C, V).float() / safe_temp.unsqueeze(1)
    if top_k_max > 0:
        scores = _apply_top_k(scores, _rows(top_k, torch.long), top_k_max)
    # any_top_p / any_min_p are host-computed flags (params are Python scalars),
    # so no GPU sync. top-p's full-vocab sort is pure waste when no row uses it.
    if any_top_p:
        scores = _apply_top_p(scores, _rows(top_p, torch.float32))
    if any_min_p:
        scores = _apply_min_p(scores, _rows(min_p, torch.float32))

    probs = torch.nan_to_num(
        torch.softmax(scores, dim=-1), nan=0.0, posinf=0.0, neginf=0.0
    )
    fallback = (~do_sample) | (probs.sum(dim=-1) <= 0)
    greedy = torch.argmax(scores, dim=-1)
    # Draw with torch.multinomial (proper RNG). The seeded hash-Gumbel path
    # (multinomial_with_seed) regressed ZH CER (~9.7 -> ~12.6, catastrophic
    # samples doubled). Replace fallback rows with a greedy one-hot so
    # multinomial never sees a zero-sum row.
    onehot = torch.zeros_like(probs)
    onehot.scatter_(-1, greedy.unsqueeze(-1), 1.0)
    probs = torch.where(fallback.unsqueeze(-1), onehot, probs)
    drawn = torch.multinomial(probs, num_samples=1).view(-1)
    return drawn.view(B, C)
