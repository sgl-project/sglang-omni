# SPDX-License-Identifier: Apache-2.0
# Adapted and modified from inclusionAI/LLaDA2.0-Uni (Apache-2.0):
# decoder/sigvq.py at commit 3457030a9c737f77f38ad5ff657e7659243d3444.
"""Semantic token embedding used by the LLaDA2-Uni image decoder."""

from __future__ import annotations

import torch
from torch import nn


class _LinearWrapper(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


class _FeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            _LinearWrapper(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class SigVQ(nn.Module):
    """Map discrete image codebook ids to decoder conditioning features."""

    def __init__(self, vocab_size: int = 16384, inner_dim: int = 4096):
        super().__init__()
        self.prior_token_embedding = nn.Embedding(vocab_size, inner_dim)
        self.prior_projector = _FeedForward(inner_dim)
        self.requires_grad_(False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.prior_projector(self.prior_token_embedding(token_ids))
