# SPDX-License-Identifier: Apache-2.0
"""Top-k sampling for the depformer, matching the reference draw."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import Protocol

import torch


@dataclass(frozen=True)
class AudioSampling:
    temperature: float
    top_k: int

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


class AudioTokenSampler(Protocol):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, kw_only=True)
class SamplingRun:
    """Adjacent batch rows that share one sampling call."""

    row_slice: slice
    sampling: AudioSampling
    generators: list[torch.Generator | None]


def fill_sampling_noise(
    noise: torch.Tensor, generators: Sequence[torch.Generator | None]
) -> None:
    start_row = 0
    for _, run in groupby(generators, key=id):
        run_generators = list(run)
        end_row = start_row + len(run_generators)
        noise[start_row:end_row].exponential_(1.0, generator=run_generators[0])
        start_row = end_row
    assert start_row == noise.shape[0], (start_row, noise.shape)


def sample_runs(logits: torch.Tensor, runs: Sequence[SamplingRun]) -> torch.Tensor:
    picks = [
        sample_token(logits[run.row_slice], run.sampling, run.generators)
        for run in runs
    ]
    return picks[0] if len(picks) == 1 else torch.cat(picks)


def sample_token(
    logits: torch.Tensor,
    sampling: AudioSampling,
    generators: Sequence[torch.Generator | None] = (),
    *,
    noise: torch.Tensor | None = None,
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
    if noise is None:
        noise = torch.empty_like(probs)
        fill_sampling_noise(noise, generators)
    else:
        assert noise.shape == probs.shape, (noise.shape, probs.shape)
    choice = (probs / noise).argmax(dim=-1, keepdim=True)
    if indices is not None:
        choice = indices.gather(-1, choice)
    else:
        pass
    return choice[:, 0]


__all__ = ["AudioSampling", "sample_token"]
