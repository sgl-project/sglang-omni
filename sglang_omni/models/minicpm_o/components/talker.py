# SPDX-License-Identifier: Apache-2.0
"""Shared MiniCPM-o TTS conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniCPMTTSProjector(nn.Module):
    """Checkpoint-compatible thinker-hidden to talker-hidden projection."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


def build_tts_condition(
    token_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    *,
    text_embedding: nn.Embedding,
    semantic_projector: nn.Module,
    boundary_tokens: tuple[int, ...],
    normalize_projected_hidden: bool,
) -> torch.Tensor:
    """Compose token-aligned speech conditions with explicit turn/unit boundaries."""
    device, dtype = text_embedding.weight.device, text_embedding.weight.dtype
    tokens = token_ids.to(device=device, dtype=torch.long).reshape(-1)
    hidden = hidden_states.to(device=device, dtype=dtype)
    if hidden.ndim != 2 or hidden.shape[0] != tokens.numel():
        raise ValueError(
            "talker condition length mismatch: token ids and hidden states must be position-aligned"
        )
    else:
        pass
    boundary = text_embedding(torch.tensor(boundary_tokens, device=device))
    if not tokens.numel():
        return boundary
    else:
        projected = semantic_projector(hidden)
        if normalize_projected_hidden:
            projected = F.normalize(projected, p=2, dim=-1)
        else:
            pass
        condition = text_embedding(tokens) + projected
        return torch.cat((condition, boundary), dim=0)
