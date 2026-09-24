# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity buffers for variable-length DiT graph capture."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class FixedPackedLayout:
    valid: torch.Tensor
    positions: torch.Tensor
    row_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    conv_positions: torch.Tensor
    conv_valid: torch.Tensor
    capacity: int


def build_fixed_packed_layout(
    lengths: torch.Tensor, padded_length: int, capacity: int, guard_width: int
) -> FixedPackedLayout:
    """Build static-shape packed metadata with a final dummy sequence."""
    assert capacity > 0 and padded_length > 0 and guard_width >= 0
    rows = lengths.numel()
    valid = (
        torch.arange(padded_length, device=lengths.device)[None, :] < lengths[:, None]
    )
    positions = valid.flatten().to(torch.int64).cumsum(0) - 1
    cu_seqlens = torch.cat(
        (
            lengths.new_zeros(1),
            lengths.cumsum(0, dtype=torch.int32),
            lengths.new_full((1,), capacity),
        )
    )
    row_ids = torch.searchsorted(
        cu_seqlens[1:],
        torch.arange(capacity, device=lengths.device, dtype=torch.int32),
        right=True,
    )
    conv_positions = (
        torch.arange(capacity, device=lengths.device) + (row_ids + 1) * guard_width
    )
    conv_valid = torch.zeros(
        capacity + (rows + 1) * guard_width,
        device=lengths.device,
        dtype=torch.bool,
    )
    conv_valid.scatter_(
        0, conv_positions, torch.ones_like(conv_positions, dtype=torch.bool)
    )
    return FixedPackedLayout(
        valid=valid,
        positions=positions,
        row_ids=row_ids,
        cu_seqlens=cu_seqlens,
        conv_positions=conv_positions,
        conv_valid=conv_valid,
        capacity=capacity,
    )


def pack_fixed_capacity(x: torch.Tensor, layout: FixedPackedLayout) -> torch.Tensor:
    """Pack valid rows into a static capacity without dynamic indexing."""
    flat = x.flatten(0, 1)
    valid = layout.valid.flatten()
    destination = torch.where(valid, layout.positions, layout.capacity).long()
    source = torch.where(valid[:, None], flat, torch.zeros_like(flat))
    packed = x.new_zeros(layout.capacity + 1, flat.shape[1])
    packed.index_add_(0, destination, source)
    return packed[: layout.capacity]


def unpack_fixed_capacity(x: torch.Tensor, layout: FixedPackedLayout) -> torch.Tensor:
    """Restore packed outputs to their padded row positions."""
    valid = layout.valid.flatten()
    source = x[torch.where(valid, layout.positions, 0).long()]
    dense = torch.where(valid[:, None], source, torch.zeros_like(source))
    return dense.reshape(*layout.valid.shape, x.shape[-1])
