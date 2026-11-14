# SPDX-License-Identifier: Apache-2.0
"""Drop policies: the order in which a window cut drops whole units of a session's history."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

# note (0xtoward): Every fourth layer keeps a KeyDiff plan near 7 ms per cut on H200.
KEYDIFF_LAYER_STRIDE = 4


@dataclass(frozen=True)
class HistoryUnit:
    """One finished unit at session positions [start, end)."""

    start: int
    end: int


class UnitKeys(Protocol):
    def read(self, start: int, end: int, layer_stride: int) -> Iterator[torch.Tensor]:
        """Float32 keys of positions [start, end) rotated back to position zero, one [tokens, kv heads, head dim] tensor per layer_stride-th layer; none while the session holds no KV."""


class UnitDropPolicy(Protocol):
    def drop_order(self, units: Sequence[HistoryUnit], keys: UnitKeys) -> list[int]:
        """Every index into units once, the first to drop first."""


class OldestFirst:
    """The checkpoint's basic window."""

    def drop_order(self, units: Sequence[HistoryUnit], keys: UnitKeys) -> list[int]:
        return list(range(len(units)))


class KeyDiff:
    """Units whose keys sit closest to the mean key go first; without retained KV, the oldest."""

    def drop_order(self, units: Sequence[HistoryUnit], keys: UnitKeys) -> list[int]:
        layer_similarities = [
            F.cosine_similarity(layer, layer.mean(dim=0, keepdim=True), dim=-1).mean(
                dim=-1
            )
            for layer in keys.read(units[0].start, units[-1].end, KEYDIFF_LAYER_STRIDE)
        ]
        if not layer_similarities:
            return list(range(len(units)))
        else:
            row_similarity = torch.stack(layer_similarities).mean(dim=0)
            unit_similarity = torch.stack(
                [
                    rows.mean()
                    for rows in row_similarity.split(
                        [unit.end - unit.start for unit in units]
                    )
                ]
            ).tolist()
            return sorted(range(len(units)), key=lambda index: -unit_similarity[index])


BUILT_IN_DROP_POLICIES: dict[str, type[UnitDropPolicy]] = {
    "oldest": OldestFirst,
    "keydiff": KeyDiff,
}


def load_drop_policy(name: str) -> UnitDropPolicy:
    """A built-in policy by name, or an instance of the class package.module:ClassName names."""
    if name in BUILT_IN_DROP_POLICIES:
        return BUILT_IN_DROP_POLICIES[name]()
    elif ":" in name:
        module_name, _, class_name = name.partition(":")
        return getattr(importlib.import_module(module_name), class_name)()
    else:
        raise ValueError(
            f"unknown drop policy {name!r}: use one of "
            f"{sorted(BUILT_IN_DROP_POLICIES)} or package.module:ClassName"
        )
