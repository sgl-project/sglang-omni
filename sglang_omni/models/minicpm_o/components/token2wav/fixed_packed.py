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
    cumulative_sequence_lengths: torch.Tensor
    convolution_positions: torch.Tensor
    convolution_valid_mask: torch.Tensor
    capacity_frames: int


def build_fixed_packed_layout(
    sequence_lengths: torch.Tensor,
    padded_frames: int,
    capacity_frames: int,
    convolution_guard_frames: int,
) -> FixedPackedLayout:
    """Build static-shape packed metadata with a final dummy sequence."""
    assert capacity_frames > 0 and padded_frames > 0 and convolution_guard_frames >= 0
    rows = sequence_lengths.numel()
    valid = (
        torch.arange(padded_frames, device=sequence_lengths.device)[None, :]
        < sequence_lengths[:, None]
    )
    cumulative_sequence_lengths = torch.cat(
        (
            sequence_lengths.new_zeros(1),
            sequence_lengths.cumsum(0, dtype=torch.int32),
            sequence_lengths.new_full((1,), capacity_frames),
        )
    )
    positions = (
        torch.where(
            valid,
            cumulative_sequence_lengths[:rows, None]
            + torch.arange(padded_frames, device=sequence_lengths.device)[None, :],
            0,
        )
        .flatten()
        .long()
    )
    packed_positions = torch.arange(
        capacity_frames, device=sequence_lengths.device, dtype=torch.int32
    )
    row_ids = torch.searchsorted(
        cumulative_sequence_lengths[1:],
        packed_positions,
        right=True,
    )
    packed_valid = packed_positions < cumulative_sequence_lengths[-2]
    source_positions = torch.where(
        packed_valid,
        row_ids * padded_frames
        + packed_positions
        - cumulative_sequence_lengths[row_ids],
        0,
    ).long()
    convolution_positions = (
        torch.arange(capacity_frames, device=sequence_lengths.device)
        + (row_ids + 1) * convolution_guard_frames
    )
    convolution_valid_mask = torch.zeros(
        capacity_frames + (rows + 1) * convolution_guard_frames,
        device=sequence_lengths.device,
        dtype=torch.bool,
    )
    convolution_valid_mask.scatter_(
        0,
        convolution_positions,
        torch.ones_like(convolution_positions, dtype=torch.bool),
    )
    return FixedPackedLayout(
        padding_mask=~valid,
        positions=positions,
        source_positions=source_positions,
        packed_padding_mask=~packed_valid,
        row_ids=row_ids,
        cumulative_sequence_lengths=cumulative_sequence_lengths,
        convolution_positions=convolution_positions,
        convolution_valid_mask=convolution_valid_mask,
        capacity_frames=capacity_frames,
    )


def unpack_fixed_capacity(x: torch.Tensor, layout: FixedPackedLayout) -> torch.Tensor:
    """Restore packed outputs to their padded row positions."""
    source = x[layout.positions]
    source.masked_fill_(layout.padding_mask.flatten()[:, None], 0)
    return source.reshape(*layout.padding_mask.shape, x.shape[-1])
