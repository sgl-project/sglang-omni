# SPDX-License-Identifier: Apache-2.0
"""Shared S2-Pro runtime components (step output + sampling helpers)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class S2ProStepOutput:
    """Per-step output containing multi-codebook tokens."""

    codes: torch.Tensor  # [num_codebooks+1, 1]


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _multinomial_no_sync(probs: Tensor) -> Tensor:
    return torch.multinomial(probs, num_samples=1).to(dtype=torch.int)


def _sample_with_topk(
    logits: Tensor,
    temperature: Tensor,
    top_p: Tensor,
    top_k: int = 30,
    repetition_penalty: Tensor | None = None,
    previous_tokens: Tensor | None = None,
) -> Tensor:
    # Repetition penalty
    if previous_tokens is not None and repetition_penalty is not None:
        prev = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=prev)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits = logits.clone()
        logits.scatter_(dim=-1, index=prev, src=score.to(logits.dtype))

    # Top-k filtering
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(
            logits, min(top_k, logits.size(-1)), dim=-1
        )
        logits = torch.full_like(logits, -float("Inf"))
        logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

    # Top-p filtering
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cum_probs > top_p
    sorted_mask[..., 0] = False  # keep at least one
    indices_to_remove = sorted_mask.scatter(
        dim=-1, index=sorted_indices, src=sorted_mask
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    # Temperature
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return _multinomial_no_sync(probs)
