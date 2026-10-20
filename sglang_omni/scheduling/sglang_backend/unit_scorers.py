# SPDX-License-Identifier: Apache-2.0
"""Scorers that rank a session's retained history units for a window cut.

A scorer reads one layer's cached keys with the rotary position taken out, so units compare by
content rather than by where they sit, and gives each unit one importance. A window drops the
least important units first; the importance of a unit is the mean over the layers it reads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F


class UnitScorer(ABC):
    """One importance per history unit from position-free keys; a cut drops the lowest first."""

    # Keys come from every layer_stride-th layer, starting at the first.
    layer_stride: int

    @abstractmethod
    def score(self, keys: torch.Tensor, unit_lengths: list[int]) -> torch.Tensor:
        """Importance of each unit, from keys [tokens, kv heads, head dim] that hold the units back to back."""


@dataclass(frozen=True, kw_only=True)
class KeyDiffUnitScorer(UnitScorer):
    """KeyDiff over whole units: a unit whose keys sit close to the mean key adds little, so it goes first.

    Silent stretches, where the model listens, give keys near the mean and go before speech.
    """

    layer_stride: int = 4

    def score(self, keys: torch.Tensor, unit_lengths: list[int]) -> torch.Tensor:
        similarity = F.cosine_similarity(
            keys, keys.mean(dim=0, keepdim=True), dim=-1
        ).mean(dim=-1)
        return -torch.stack([unit.mean() for unit in similarity.split(unit_lengths)])
