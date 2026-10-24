# SPDX-License-Identifier: Apache-2.0
"""Per-stream code2wav decoder state held by slot: the state's layout, and the pool whose
histories, overlaps and K/V rings a decode's kernels read and write in place."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.utils.common import is_pin_memory_available


@dataclass(frozen=True, kw_only=True)
class CodecStateSpec:
    """One stream's incremental codec state: (key, channels, length) conv histories and
    transposed-conv overlaps in decode order, zero lengths kept, and its K/V shape."""

    conv_histories: tuple[tuple[str, int, int], ...]
    transconv_overlaps: tuple[tuple[str, int, int], ...]
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    retained_context: int


@dataclass(frozen=True, kw_only=True)
class RingAddresses:
    """A decode's addresses in the per-slot K/V rings: the absolute positions of its new
    frames and the ring rows their K/V go to, and each row's attention window as ring rows in
    position order with its length."""

    frame_positions: torch.Tensor
    write_rows: torch.Tensor
    window_rows: torch.Tensor
    window_lengths: torch.Tensor


def ring_addresses(
    slot_index: torch.Tensor,
    start_positions: torch.Tensor,
    new_frames: int,
    ring_frames: int,
    retained_frames: int,
) -> RingAddresses:
    """Position p of a slot's stream lives at ring row slot * ring_frames + p % ring_frames; a
    row's window is the retained_frames positions before its first new frame and its new
    frames, on distinct rows while ring_frames >= retained_frames + new_frames."""
    frame_offsets = torch.arange(new_frames, device=slot_index.device)
    frame_positions = start_positions[:, None] + frame_offsets
    slot_rows = slot_index[:, None] * ring_frames
    window_starts = (start_positions - retained_frames).clamp(min=0)
    window_positions = window_starts[:, None] + torch.arange(
        ring_frames, device=slot_index.device
    )
    return RingAddresses(
        frame_positions=frame_positions.reshape(-1),
        write_rows=(slot_rows + frame_positions % ring_frames).reshape(-1),
        window_rows=slot_rows + window_positions % ring_frames,
        window_lengths=start_positions - window_starts + new_frames,
    )


class CodecStatePool:
    """Per-slot codec state where the decode's kernels read and write it: a history pool per
    causal conv, an overlap pool per transposed conv, per-layer K/V rings and frame positions,
    then capture_slots scratch slots past the streams' slots, one per captured row."""

    def __init__(
        self,
        spec: CodecStateSpec,
        *,
        num_slots: int,
        capture_slots: int,
        ring_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        assert (
            ring_frames > spec.retained_context
        ), "a ring holds the window and new frames"
        total_slots = num_slots + capture_slots
        self.device = device
        self.num_slots = num_slots
        self.free_slots = list(reversed(range(num_slots)))
        self.ring_frames = ring_frames
        self.retained_frames = spec.retained_context
        self.scratch_slots = tuple(range(num_slots, total_slots))
        self.histories = {
            state_key: torch.zeros(
                total_slots, length, channels, device=device, dtype=dtype
            )
            for state_key, channels, length in spec.conv_histories
            if length
        }
        self.overlaps = {
            state_key: torch.zeros(
                total_slots, length, channels, device=device, dtype=dtype
            )
            for state_key, channels, length in spec.transconv_overlaps
            if length
        }
        ring_width = spec.num_key_value_heads * spec.head_dim
        self.keys = [
            torch.zeros(
                total_slots * ring_frames, ring_width, device=device, dtype=dtype
            )
            for _ in range(spec.num_layers)
        ]
        self.values = [torch.zeros_like(ring) for ring in self.keys]
        self.frame_positions = torch.zeros(total_slots, device=device, dtype=torch.long)
        self.bytes_per_slot = dtype.itemsize * (
            sum(pool[0].numel() for pool in self.histories.values())
            + sum(pool[0].numel() for pool in self.overlaps.values())
            + 2 * spec.num_layers * ring_frames * ring_width
        )

    def available_size(self) -> int:
        return len(self.free_slots)

    def acquire(self) -> int | None:
        """A free slot, or None when every slot is taken. Every decode runs on the serving
        thread's one stream, so a slot's next owner decodes after its previous owner."""
        if not self.free_slots:
            return None
        else:
            pass
        return self.free_slots.pop()

    def release(self, slot: int) -> None:
        self.free_slots.append(slot)

    def stage_index(self, slots: list[int]) -> torch.Tensor:
        """The slots' ids on the device, copied from pinned memory so the host never waits."""
        return torch.tensor(
            slots, dtype=torch.long, pin_memory=is_pin_memory_available(self.device)
        ).to(self.device, non_blocking=True)

    def zero_slots(self, slots: torch.Tensor) -> None:
        """Clear the slots' state, for a capture whose eager and replayed decodes must start
        from one state."""
        for pool in (*self.histories.values(), *self.overlaps.values()):
            pool.index_fill_(0, slots, 0)
        for ring in (*self.keys, *self.values):
            ring.view(-1, self.ring_frames, ring.shape[1]).index_fill_(0, slots, 0)
        self.frame_positions.index_fill_(0, slots, 0)
