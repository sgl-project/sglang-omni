# SPDX-License-Identifier: Apache-2.0
"""Top-k sampling for the depformer, matching the reference draw."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AudioSampling:
    temperature: float
    top_k: int

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def sample_token(
    logits: torch.Tensor,
    sampling: AudioSampling,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """[B, card] float logits → [B] token ids.

    The reference draws from the top-k via the exponential race (argmax of
    p / Exp(1)), which avoids a host sync; kept so seeded runs line up.
    """
    if sampling.greedy:
        return logits.argmax(dim=-1)
    else:
        pass
    probs = torch.softmax(logits / sampling.temperature, dim=-1)
    if sampling.top_k > 0:
        probs, indices = torch.topk(probs, min(sampling.top_k, probs.shape[-1]), dim=-1)
    else:
        indices = None
    noise = torch.empty_like(probs).exponential_(1.0, generator=generator)
    choice = (probs / noise).argmax(dim=-1, keepdim=True)
    if indices is not None:
        choice = indices.gather(-1, choice)
    else:
        pass
    return choice[:, 0]


__all__ = ["AudioSampling", "sample_token"]
