# SPDX-License-Identifier: Apache-2.0
"""Repetition-aware sampling: redraw a sampled token that already appears among
the latest generated tokens, as VALL-E 2 and upstream CosyVoice do."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from sglang.srt.layers.sampler import sampling_from_probs_torch

# note (Yucheng Hu): the redraw reusing the first draw's seeded noise would be
# conditioned on the rejected candidate having won that draw.
REDRAW_SEED_SALT = 0x2545F491


def repetition_aware_redraw(
    probs: torch.Tensor,
    sampled_ids: torch.Tensor,
    output_ids: Sequence[Sequence[int]],
    window_size: int,
    redraw_allowed: torch.Tensor,
    sampling_seeds: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Redraw each allowed row whose sampled token is among its last outputs.

    probs holds the full-vocabulary probabilities after penalties and
    temperature but before top-k/top-p, so masking the repeated token keeps a
    stop token reachable after truncation collapsed onto it. Returns the token
    ids and a mask of the redrawn rows.
    """
    recent_rows = [list(row_ids[-window_size:]) for row_ids in output_ids]
    recent_ids = torch.tensor(
        [row + [-1] * (window_size - len(row)) for row in recent_rows],
        dtype=torch.long,
    ).to(probs.device, non_blocking=True)
    candidate_ids = sampled_ids.long().unsqueeze(1)
    is_repeated = (recent_ids == candidate_ids).any(dim=1) & redraw_allowed
    redraw_probs = probs.to(torch.float32, copy=True)
    redraw_probs.scatter_(1, candidate_ids, 0.0)
    # note (Yucheng Hu): multinomial rejects an all-zero row, and a row whose
    # whole mass sat on the candidate has nothing else to draw.
    redraw_probs = torch.where(
        redraw_probs.sum(dim=1, keepdim=True) > 0, redraw_probs, probs
    )
    redraw_seeds = None if sampling_seeds is None else sampling_seeds ^ REDRAW_SEED_SALT
    redraw_ids = sampling_from_probs_torch(redraw_probs, redraw_seeds, positions)
    return torch.where(is_repeated, redraw_ids, sampled_ids), is_repeated
