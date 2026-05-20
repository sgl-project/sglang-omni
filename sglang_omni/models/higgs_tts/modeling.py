# SPDX-License-Identifier: Apache-2.0
"""Fused multi-codebook modules for HiggsMultimodalQwen3 (discrete TTS path).

Standalone ``nn.Module``s with no framework dependencies — the full
sglang-integrated model class composes these with a Qwen3 backbone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HiggsFusedMultiTextEmbedding(nn.Module):
    """Fused multi-codebook embedding: one ``[N*V, D]`` weight + offset lookup.

    Equivalent to an ensemble of ``N`` per-codebook embeddings but stored
    contiguously. ``codes_LN[..., N]`` → ``[..., D]`` summed across the
    codebook axis.
    """

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

    def forward(self, codes_LN: torch.Tensor) -> torch.Tensor:
        N = self.num_codebooks
        V = self.vocab_size
        offsets = torch.arange(N, device=codes_LN.device, dtype=codes_LN.dtype) * V
        fused_ids = codes_LN + offsets
        return F.embedding(fused_ids, self.weight).sum(dim=-2)


class HiggsFusedMultiTextHead(nn.Module):
    """Fused multi-codebook head: ``[L, D]`` → ``[L, N, V]`` via one linear.

    Tied with :class:`HiggsFusedMultiTextEmbedding` when ``tie_word_embeddings``.
    """

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

    def generate(self, hidden_LD: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_LD, self.weight)
        return logits.reshape(
            hidden_LD.shape[0],
            self.num_codebooks,
            self.vocab_size,
        )


__all__ = ["HiggsFusedMultiTextEmbedding", "HiggsFusedMultiTextHead"]
