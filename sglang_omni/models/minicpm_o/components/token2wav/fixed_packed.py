# SPDX-License-Identifier: Apache-2.0
"""Fixed-capacity buffers for variable-length DiT graph capture."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class FixedPackedLayout:
    padding_mask: torch.Tensor
    positions: torch.Tensor
    source_positions: torch.Tensor
    packed_padding_mask: torch.Tensor
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
    cu_seqlens = torch.cat(
        (
            lengths.new_zeros(1),
            lengths.cumsum(0, dtype=torch.int32),
            lengths.new_full((1,), capacity),
        )
    )
    positions = (
        torch.where(
            valid,
            cu_seqlens[:rows, None]
            + torch.arange(padded_length, device=lengths.device)[None, :],
            0,
        )
        .flatten()
        .long()
    )
    packed_positions = torch.arange(capacity, device=lengths.device, dtype=torch.int32)
    row_ids = torch.searchsorted(
        cu_seqlens[1:],
        packed_positions,
        right=True,
    )
    packed_valid = packed_positions < cu_seqlens[-2]
    source_positions = torch.where(
        packed_valid,
        row_ids * padded_length + packed_positions - cu_seqlens[row_ids],
        0,
    ).long()
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
        padding_mask=~valid,
        positions=positions,
        source_positions=source_positions,
        packed_padding_mask=~packed_valid,
        row_ids=row_ids,
        cu_seqlens=cu_seqlens,
        conv_positions=conv_positions,
        conv_valid=conv_valid,
        capacity=capacity,
    )


def pack_fixed_capacity(
    x: torch.Tensor, layout: FixedPackedLayout, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Gather valid rows without reducing padded rows into a shared sink."""
    packed = torch.index_select(x.flatten(0, 1), 0, layout.source_positions, out=out)
    return packed.masked_fill_(layout.packed_padding_mask[:, None], 0)


def unpack_fixed_capacity(x: torch.Tensor, layout: FixedPackedLayout) -> torch.Tensor:
    """Restore packed outputs to their padded row positions."""
    source = x[layout.positions]
    source.masked_fill_(layout.padding_mask.flatten()[:, None], 0)
    return source.reshape(*layout.padding_mask.shape, x.shape[-1])
