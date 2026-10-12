# SPDX-License-Identifier: Apache-2.0
"""Repetition-aware sampling: redraw a sampled token that already appears among
the latest generated tokens."""

from __future__ import annotations

import torch
from sglang.srt.layers.sampler import sampling_from_probs_torch

# note (Yucheng Hu): the redraw reusing the first draw's seeded noise would be
# conditioned on the rejected candidate having won that draw.
COSYVOICE3_RAS_SEED_SALT = 0x2545F491


def repetition_aware_redraw(
    probs: torch.Tensor,
    sampled_ids: torch.Tensor,
    rows: torch.Tensor,
    sampling_seeds: torch.Tensor | None,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Redraw the given rows from their full distribution with the sampled token masked.

    probs holds the probabilities after penalties and temperature but before
    top-k/top-p, so masking the repeated token keeps a stop token reachable.
    """
    # note (ratish): float64, as SGLang's own seeded draw feeds its compiled
    # multinomial_with_seed, so a seeded redraw reuses that graph, not a new one.
    row_probs = probs[rows].to(torch.float64)
    redraw_probs = row_probs.scatter(1, sampled_ids[rows].long().unsqueeze(1), 0.0)
    # note (Yucheng Hu): multinomial rejects an all-zero row, and a row whose
    # whole mass sat on the candidate has nothing else to draw.
    redraw_probs = torch.where(
        redraw_probs.sum(dim=1, keepdim=True) > 0, redraw_probs, row_probs
    )
    redraw_seeds = (
        None
        if sampling_seeds is None
        else sampling_seeds[rows] ^ COSYVOICE3_RAS_SEED_SALT
    )
    token_ids = sampled_ids.clone()
    token_ids[rows] = sampling_from_probs_torch(
        redraw_probs, redraw_seeds, positions[rows]
    ).to(token_ids.dtype)
    return token_ids
