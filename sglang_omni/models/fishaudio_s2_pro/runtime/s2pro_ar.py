# SPDX-License-Identifier: Apache-2.0
"""Shared S2-Pro runtime components (step output + sampling helpers)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class S2ProStepOutput:
    """Per-step output containing multi-codebook tokens."""

    codes: torch.Tensor  # [num_codebooks+1, 1]


# ---------------------------------------------------------------------------
# Sampling helpers
#
# CUDA-graph-safe: uses torch.topk + gather (no scatter), Gumbel-max trick
# (no torch.multinomial sync), and top_k as a Python int so torch.topk can
# compile with a fixed k.
# ---------------------------------------------------------------------------


def _sample_with_topk(
    logits: Tensor,
    temperature: Tensor,
    top_p: Tensor,
    top_k: int = 30,
    repetition_penalty: Tensor | None = None,
    previous_tokens: Tensor | None = None,
) -> Tensor:
    if previous_tokens is not None and repetition_penalty is not None:
        prev = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=prev)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits = logits.clone()
        logits.scatter_(dim=-1, index=prev, src=score.to(logits.dtype))

    top_vals, top_indices = torch.topk(logits, k=top_k, dim=-1)

    # Top-p nucleus filtering on un-scaled probabilities
    probs_raw = F.softmax(top_vals, dim=-1)
    cum_probs = torch.cumsum(probs_raw, dim=-1)
    mask = cum_probs > top_p
    mask[..., 0] = False

    top_vals = top_vals / torch.clip(temperature, min=1e-5)
    top_vals = top_vals.masked_fill(mask, -float("inf"))
    probs = F.softmax(top_vals, dim=-1)

    # Gumbel-max sampling (no CUDA sync)
    q = torch.empty_like(probs).exponential_(1)
    sampled_idx = torch.argmax(probs / q, dim=-1, keepdim=True)
    return torch.gather(top_indices, dim=-1, index=sampled_idx).to(dtype=torch.int)
