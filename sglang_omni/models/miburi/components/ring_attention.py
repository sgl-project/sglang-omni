# SPDX-License-Identifier: Apache-2.0
"""Causal attention of one step over a bounded ring of past keys and values.

Each row keeps its own position, the number of steps it has taken. A step rotates its query and key to that position, writes the key and value into slot position % capacity and attends to every slot written so far; once the ring is full those are exactly the last capacity steps, the current one included. Which slots hold data follows from the position alone, so a row restarts by zeroing its position and its stale slots stay masked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch.nn import functional

from sglang_omni.models.personaplex.components.mimi_transformer import (
    apply_interleaved_rope,
)


@dataclass
class AttentionRing:
    """Rotated keys and values [rows, heads, capacity, head_dim], updated in place."""

    keys: torch.Tensor
    values: torch.Tensor

    @classmethod
    def zeros(
        cls,
        rows: int,
        heads: int,
        capacity: int,
        head_dim: int,
        *,
        device: torch.device,
    ) -> AttentionRing:
        return cls(
            keys=torch.zeros(rows, heads, capacity, head_dim, device=device),
            values=torch.zeros(rows, heads, capacity, head_dim, device=device),
        )


def attend_ring(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    ring: AttentionRing,
    rope_max_period: float,
) -> torch.Tensor:
    """q, k, v [rows, heads, 1, head_dim] of the step each row takes at positions [rows] → [rows, 1, heads * head_dim]."""
    q, k = apply_interleaved_rope(q, k, positions[:, None], rope_max_period)
    capacity = ring.keys.shape[2]
    slots = (positions % capacity).view(-1, 1, 1, 1).expand_as(k)
    ring.keys.scatter_(2, slots, k)
    ring.values.scatter_(2, slots, v)
    # Note (Junnan Li): The whole ring in slot order with a boolean mask, not the written slots gathered, keeps the reference's CPU attention arithmetic bit for bit.
    written = torch.arange(capacity, device=positions.device) <= positions[:, None]
    attended = functional.scaled_dot_product_attention(
        q, ring.keys, ring.values, attn_mask=written[:, None, None]
    )
    return rearrange(attended, "b h t d -> b t (h d)")


__all__ = ["AttentionRing", "attend_ring"]
