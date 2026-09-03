# SPDX-License-Identifier: Apache-2.0
"""Pure-torch top-k/top-p renorm fallbacks for platforms without ``sgl_kernel``."""

from __future__ import annotations

import torch


def top_k_renorm_prob(probs: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """torch fallback for ``sgl_kernel.top_k_renorm_prob`` (Ascend NPU).

    Zero entries below each row's k-th largest value, then renormalize.
    """
    v = probs.shape[-1]
    k_safe = k.long().clamp(min=1, max=v)
    kth = probs.sort(dim=-1, descending=True).values.gather(
        -1, (k_safe - 1).unsqueeze(-1)
    ).squeeze(-1)
    masked = torch.where(probs < kth.unsqueeze(-1), torch.zeros_like(probs), probs)
    denom = masked.sum(dim=-1, keepdim=True)
    return masked / denom.clamp_min(torch.finfo(probs.dtype).tiny)


def top_p_renorm_prob(probs: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """torch fallback for ``sgl_kernel.top_p_renorm_prob`` (Ascend NPU).
    
    Nucleus filtering (always keeps the highest-probability token).
    """
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    remove = cum_probs > p.unsqueeze(-1)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    scatter = torch.zeros_like(remove)
    scatter.scatter_(-1, sorted_indices, remove)
    masked = torch.where(scatter, torch.zeros_like(probs), probs)
    denom = masked.sum(dim=-1, keepdim=True)
    return masked / denom.clamp_min(torch.finfo(probs.dtype).tiny)
