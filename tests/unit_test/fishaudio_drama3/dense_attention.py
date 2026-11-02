# SPDX-License-Identifier: Apache-2.0
"""Test-side DecoderAttention keeping dense key/value rows per (layer, request)."""

from __future__ import annotations

import math

import torch

# (position, key [kv_heads, head_dim], value [kv_heads, head_dim])
CacheEntry = tuple[int, torch.Tensor, torch.Tensor]


class DenseAttention:
    """Writing a position again replaces its row, as rewriting a paged cache slot does."""

    def __init__(self, *, head_count: int, kv_head_count: int, head_dim: int) -> None:
        self.head_count = head_count
        self.kv_head_count = kv_head_count
        self.head_dim = head_dim
        self.entries: dict[tuple[int, str], list[CacheEntry]] = {}
        self.request_ids: list[str] = []
        self.positions: list[int] = []

    def bind(self, request_ids: list[str], positions: list[int]) -> None:
        self.request_ids = request_ids
        self.positions = positions

    def __call__(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        for row, (request_id, position) in enumerate(
            zip(self.request_ids, self.positions)
        ):
            entry = (
                position,
                key[row].view(self.kv_head_count, self.head_dim),
                value[row].view(self.kv_head_count, self.head_dim),
            )
            entries = self.entries.setdefault((layer_index, request_id), [])
            written_positions = [cached[0] for cached in entries]
            if position in written_positions:
                entries[written_positions.index(position)] = entry
            else:
                entries.append(entry)
        group_size = self.head_count // self.kv_head_count
        outputs = []
        for row, (request_id, position) in enumerate(
            zip(self.request_ids, self.positions)
        ):
            visible = [
                entry
                for entry in self.entries[(layer_index, request_id)]
                if entry[0] <= position
            ]
            keys = torch.stack([entry[1] for entry in visible]).float()
            values = torch.stack([entry[2] for entry in visible]).float()
            scores = torch.einsum(
                "hd,nhd->hn",
                query[row].view(self.head_count, self.head_dim).float(),
                keys.repeat_interleave(group_size, dim=1),
            ) / math.sqrt(self.head_dim)
            attended = torch.einsum(
                "hn,nhd->hd",
                torch.softmax(scores, dim=-1),
                values.repeat_interleave(group_size, dim=1),
            )
            outputs.append(attended.reshape(-1).to(query.dtype))
        return torch.stack(outputs)
