"""Slot-indexed storage for Qwen3-TTS incremental Codec decoder state."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

import torch

from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)

if TYPE_CHECKING:
    from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
        IncrementalCodecGraphStats,
    )
else:
    pass


class DisabledCodecGraphStats(TypedDict):
    enabled: bool


class CodecCudaGraphStats(TypedDict):
    cold: IncrementalCodecGraphStats | DisabledCodecGraphStats
    window: IncrementalCodecGraphStats | DisabledCodecGraphStats
    warm: list[IncrementalCodecGraphStats]


class CodecStateStats(TypedDict, total=False):
    """Arena/scheduler snapshot; a disabled decoder reports only enabled."""

    slots: int
    active_slots: int
    bytes_per_slot: int
    total_bytes: int
    exhausted: int
    enabled: bool
    left_context_fallbacks: int
    cuda_graphs: CodecCudaGraphStats


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
        else:
            pass
        self.decoder = decoder
        self.device = torch.device(device)
        self.dtype = dtype
        self._num_slots = int(num_slots)  # noqa: leading-underscore
        self.scratch_slot = self._num_slots  # noqa: leading-underscore
        self.storage = decoder.init_state(
            self._num_slots + 1,
            device=self.device,
            dtype=dtype,  # noqa: leading-underscore
        )
        self.lock = threading.Lock()
        self.staging = threading.local()
        self.release_events: dict[int, torch.cuda.Event] = {}
        self.free: list[int] = list(
            reversed(range(self._num_slots))
        )  # noqa: leading-underscore
        self.retired: set[int] = set()
        self._exhausted_count = 0  # noqa: leading-underscore
        spec = decoder.state_spec()
        self._bytes_per_slot = spec.bytes_per_stream(dtype)  # noqa: leading-underscore

    @property
    def num_slots(self) -> int:
        return self._num_slots  # noqa: leading-underscore

    @property
    def bytes_per_slot(self) -> int:
        return self._bytes_per_slot  # noqa: leading-underscore

    @property
    def total_bytes(self) -> int:
        return self._bytes_per_slot * self._num_slots  # noqa: leading-underscore

    @property
    def exhausted_count(self) -> int:
        return self._exhausted_count  # noqa: leading-underscore

    def active_slots(self) -> int:
        with self.lock:
            return (
                self._num_slots - len(self.free) - len(self.retired)
            )  # noqa: leading-underscore

    def acquire(self) -> int | None:
        """Take a zeroed slot, or ``None`` when the arena is full."""
        with self.lock:
            if not self.free:
                self._exhausted_count += 1  # noqa: leading-underscore
                return None
            else:
                pass
            slot = self.free.pop()
            released = self.release_events.pop(slot, None)
        if released is not None:
            torch.cuda.current_stream(self.device).wait_event(released)
        else:
            pass
        self.zero_slot(slot)
        return slot

    def release(self, slot: int) -> None:
        released = None
        if self.device.type == "cuda":
            released = torch.cuda.Event()
            released.record(torch.cuda.current_stream(self.device))
        else:
            pass
        with self.lock:
            if slot in self.retired:
                return
            else:
                pass
            if slot in self.free:
                raise RuntimeError(
                    f"Qwen3-TTS codec state slot {slot} was released twice"
                )
            else:
                pass
            if released is not None:
                self.release_events[slot] = released
            else:
                pass
            self.free.append(slot)

    def retire(self, slot: int) -> None:
        """Withdraw a slot for the life of the process.

        Note (Qihao Liu): used when a decode that touched the slot could not be
        proven complete, so its memory must never be handed to later work.
        """
        with self.lock:
            self.retired.add(slot)
            self.release_events.pop(slot, None)
            if slot in self.free:
                self.free.remove(slot)
            else:
                pass

    def _buffers(self, state: Qwen3TTSIncrementalCodecState) -> list[torch.Tensor]:
        return [
            *state.conv_histories.values(),
            *state.transconv_overlaps.values(),
            *state.transformer_keys.values(),
            *state.transformer_values.values(),
        ]

    def zero_slot(self, slot: int) -> None:
        for buffer in self._buffers(self.storage):
            buffer[slot].zero_()
        self.storage.frame_positions[slot] = 0

    STAGING_RING = 4

    def staged(self, name: str, values: Sequence[int]) -> torch.Tensor:
        if self.device.type != "cuda":
            return torch.as_tensor(list(values), dtype=torch.long)
        else:
            pass
        count = len(values)
        if count == 0:
            raise ValueError("Qwen3-TTS codec state arena needs at least one slot")
        else:
            pass
        ring = getattr(self.staging, f"{name}_ring", None)
        if ring is None:
            ring = [
                (
                    torch.empty(
                        self._num_slots + 1, dtype=torch.long
                    ).pin_memory(),  # noqa: leading-underscore
                    torch.empty(
                        self._num_slots + 1,
                        dtype=torch.long,
                        device=self.device,  # noqa: leading-underscore
                    ),
                )
                for _ in range(self.STAGING_RING)
            ]
            setattr(self.staging, f"{name}_ring", ring)
            setattr(self.staging, f"{name}_turn", 0)
        else:
            pass
        turn = (getattr(self.staging, f"{name}_turn") + 1) % self.STAGING_RING
        setattr(self.staging, f"{name}_turn", turn)
        host, device = ring[turn]
        host[:count].copy_(torch.as_tensor(list(values), dtype=torch.long))
        device[:count].copy_(host[:count], non_blocking=True)
        return device[:count]

    def stage_index(self, slots: Sequence[int]) -> torch.Tensor:
        """Stage a cohort's slot ids on the device without a host sync."""
        return self.staged("index", slots)

    def gather(self, slots: Sequence[int]) -> Qwen3TTSIncrementalCodecState:
        """Select a cohort's rows into one contiguous state."""
        return self.gather_by_index(self.stage_index(slots))

    def gather_by_index(self, index: torch.Tensor) -> Qwen3TTSIncrementalCodecState:
        """Select the rows named by a device index tensor.

        Capturable: with a static index this is the gather half of a graph
        that reads the arena directly.
        """
        storage = self.storage
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
        self.scatter_by_index(self.stage_index(slots), state)

    def scatter_by_index(
        self, index: torch.Tensor, state: Qwen3TTSIncrementalCodecState
    ) -> None:
        """Write a cohort state into the rows named by a device index tensor."""
        storage = self.storage
        if state.frame_positions is None:
            raise RuntimeError(
                "Qwen3-TTS codec state arena requires per-row frame positions"
            )
        else:
            pass
        self.copy_rows(storage.frame_positions, index, state.frame_positions)
        for key, buffer in storage.conv_histories.items():
            self.copy_rows(buffer, index, state.conv_histories[key], key)
        for key, buffer in storage.transconv_overlaps.items():
            self.copy_rows(buffer, index, state.transconv_overlaps[key], key)
        for layer_index, buffer in storage.transformer_keys.items():
            self.copy_rows(
                buffer, index, state.transformer_keys[layer_index], f"key.{layer_index}"
            )
        for layer_index, buffer in storage.transformer_values.items():
            self.copy_rows(
                buffer,
                index,
                state.transformer_values[layer_index],
                f"value.{layer_index}",
            )

    @staticmethod
    def copy_rows(
        buffer: torch.Tensor,
        index: torch.Tensor,
        rows: torch.Tensor,
        key: str = "frame_positions",
    ) -> None:
        expected = (int(index.shape[0]), *buffer.shape[1:])
        if tuple(rows.shape) != expected:
            raise RuntimeError(
                f"Qwen3-TTS codec state arena expected {expected} for {key}, got {tuple(rows.shape)}"
            )
        else:
            pass
        if rows.dtype != buffer.dtype:
            raise RuntimeError(
                f"Qwen3-TTS codec state arena expected {buffer.dtype} for {key}, got {rows.dtype}"
            )
        else:
            pass
        buffer.index_copy_(0, index, rows.contiguous())

    def describe(self) -> CodecStateStats:
        return {
            "slots": self._num_slots,  # noqa: leading-underscore
            "active_slots": self.active_slots(),
            "bytes_per_slot": self._bytes_per_slot,  # noqa: leading-underscore
            "total_bytes": self.total_bytes,
            "exhausted": self._exhausted_count,  # noqa: leading-underscore
        }


__all__ = ["Qwen3TTSCodecStateArena"]
