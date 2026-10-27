# SPDX-License-Identifier: Apache-2.0
"""Top-k sampling for the depformer, matching the reference draw."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby

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
    generators: Sequence[torch.Generator | None],
) -> torch.Tensor:
    """[B, card] float logits → [B] token ids.

    generators[i] draws row i's noise, or the default generator when None.
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
    noise = torch.empty_like(probs)
    start_row = 0
    # Note (edwardzh): A seeded row draws its slice alone, consuming its generator
    # exactly as it would at batch size one; neighbours never advance it.
    for _, run in groupby(generators, key=id):
        run_generators = list(run)
        end_row = start_row + len(run_generators)
        noise[start_row:end_row].exponential_(1.0, generator=run_generators[0])
        start_row = end_row
    assert start_row == noise.shape[0], (start_row, noise.shape)
    choice = (probs / noise).argmax(dim=-1, keepdim=True)
    if indices is not None:
        choice = indices.gather(-1, choice)
    else:
        pass
    return choice[:, 0]


__all__ = ["AudioSampling", "sample_token"]
