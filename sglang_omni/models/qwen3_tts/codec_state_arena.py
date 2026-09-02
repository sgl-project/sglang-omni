# SPDX-License-Identifier: Apache-2.0
"""Slot-indexed storage for Qwen3-TTS incremental Codec decoder state."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import torch

from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)


class Qwen3TTSCodecStateArena:
    """Bounded, reusable storage for per-stream incremental Codec state.

    Every buffer from ``Qwen3TTSIncrementalDecoder.state_spec`` carries a
    leading slot dimension, so the arena is exactly the state a decode of
    ``num_slots`` rows would use: ``gather`` selects a cohort's rows into a
    contiguous state and ``scatter`` writes the advanced state back.

    Note (Qihao Liu): ``acquire`` zeroes a slot before handing it out, so a
    reused slot is provably a cold start; ``release`` therefore returns a slot
    without clearing it.
    """

    def __init__(
        self,
        decoder: Qwen3TTSIncrementalDecoder,
        *,
        num_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if num_slots <= 0:
            raise ValueError("Qwen3-TTS codec state arena needs at least one slot")
        self._decoder = decoder
        self._device = torch.device(device)
        self._dtype = dtype
        self._num_slots = int(num_slots)
        self._storage = decoder.init_state(
            self._num_slots, device=self._device, dtype=dtype
        )
        self._lock = threading.Lock()
        self._free: list[int] = list(reversed(range(self._num_slots)))
        self._retired: set[int] = set()
        self._exhausted_count = 0
        spec = decoder.state_spec()
        self._bytes_per_slot = spec.bytes_per_stream(dtype)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def bytes_per_slot(self) -> int:
        return self._bytes_per_slot

    @property
    def total_bytes(self) -> int:
        return self._bytes_per_slot * self._num_slots

    @property
    def exhausted_count(self) -> int:
        return self._exhausted_count

    def active_slots(self) -> int:
        with self._lock:
            return self._num_slots - len(self._free) - len(self._retired)

    def acquire(self) -> int | None:
        """Take a zeroed slot, or ``None`` when the arena is full."""
        with self._lock:
            if not self._free:
                self._exhausted_count += 1
                return None
            slot = self._free.pop()
        self._zero_slot(slot)
        return slot

    def release(self, slot: int) -> None:
        with self._lock:
            if slot in self._retired:
                return
            if slot in self._free:
                raise RuntimeError(
                    f"Qwen3-TTS codec state slot {slot} was released twice"
                )
            self._free.append(slot)

    def retire(self, slot: int) -> None:
        """Withdraw a slot for the life of the process.

        Note (Qihao Liu): used when a decode that touched the slot could not be
        proven complete, so its memory must never be handed to later work.
        """
        with self._lock:
            self._retired.add(slot)
            if slot in self._free:
                self._free.remove(slot)

    def _buffers(self, state: Qwen3TTSIncrementalCodecState) -> list[torch.Tensor]:
        return [
            *state.conv_histories.values(),
            *state.transconv_overlaps.values(),
            *state.transformer_keys.values(),
            *state.transformer_values.values(),
        ]

    def _zero_slot(self, slot: int) -> None:
        for buffer in self._buffers(self._storage):
            buffer[slot].zero_()
        self._storage.frame_positions[slot] = 0

    def _index(self, slots: Sequence[int]) -> torch.Tensor:
        if not slots:
            raise ValueError("Qwen3-TTS codec state arena needs at least one slot")
        return torch.as_tensor(list(slots), device=self._device, dtype=torch.long)

    def gather(self, slots: Sequence[int]) -> Qwen3TTSIncrementalCodecState:
        """Select a cohort's rows into one contiguous state."""
        index = self._index(slots)
        storage = self._storage
        state = Qwen3TTSIncrementalCodecState(
            transformer_context_length=storage.transformer_context_length,
            frame_positions=storage.frame_positions.index_select(0, index),
        )
        for key, buffer in storage.conv_histories.items():
            state.conv_histories[key] = buffer.index_select(0, index)
        for key, buffer in storage.transconv_overlaps.items():
            state.transconv_overlaps[key] = buffer.index_select(0, index)
        for layer_index, buffer in storage.transformer_keys.items():
            state.transformer_keys[layer_index] = buffer.index_select(0, index)
        for layer_index, buffer in storage.transformer_values.items():
            state.transformer_values[layer_index] = buffer.index_select(0, index)
        return state

    def scatter(
        self, slots: Sequence[int], state: Qwen3TTSIncrementalCodecState
    ) -> None:
        """Write an advanced cohort state back into its slots."""
        index = self._index(slots)
        storage = self._storage
        if state.frame_positions is None:
            raise RuntimeError(
                "Qwen3-TTS codec state arena requires per-row frame positions"
            )
        self._copy_rows(storage.frame_positions, index, state.frame_positions)
        for key, buffer in storage.conv_histories.items():
            self._copy_rows(buffer, index, state.conv_histories[key], key)
        for key, buffer in storage.transconv_overlaps.items():
            self._copy_rows(buffer, index, state.transconv_overlaps[key], key)
        for layer_index, buffer in storage.transformer_keys.items():
            self._copy_rows(
                buffer, index, state.transformer_keys[layer_index], f"key.{layer_index}"
            )
        for layer_index, buffer in storage.transformer_values.items():
            self._copy_rows(
                buffer,
                index,
                state.transformer_values[layer_index],
                f"value.{layer_index}",
            )

    @staticmethod
    def _copy_rows(
        buffer: torch.Tensor,
        index: torch.Tensor,
        rows: torch.Tensor,
        key: str = "frame_positions",
    ) -> None:
        expected = (int(index.shape[0]), *buffer.shape[1:])
        if tuple(rows.shape) != expected:
            raise RuntimeError(
                f"Qwen3-TTS codec state arena expected {expected} for {key}, "
                f"got {tuple(rows.shape)}"
            )
        if rows.dtype != buffer.dtype:
            raise RuntimeError(
                f"Qwen3-TTS codec state arena expected {buffer.dtype} for {key}, "
                f"got {rows.dtype}"
            )
        buffer.index_copy_(0, index, rows.contiguous())

    def describe(self) -> dict[str, Any]:
        return {
            "slots": self._num_slots,
            "active_slots": self.active_slots(),
            "bytes_per_slot": self._bytes_per_slot,
            "total_bytes": self.total_bytes,
            "exhausted": self._exhausted_count,
        }


__all__ = ["Qwen3TTSCodecStateArena"]
