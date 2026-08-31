# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduling for Ming-Omni-TTS."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch

from sglang_omni.models.ming_tts.audio_decode import (
    MingAudioDecoder,
    decode_ming_tts_audio_payload,
)
from sglang_omni.models.ming_tts.payload_types import load_ming_tts_state
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase

logger = logging.getLogger(__name__)


class _AudioVAEStreamingSlotBindings:
    def __init__(
        self,
        decoder: MingAudioDecoder,
    ) -> None:
        self._decoder = decoder
        self._request_to_slot: dict[str, int] = {}
        self._free_slots = list(reversed(range(decoder.stream_capacity)))

    def try_bind(self, request_id: str) -> int | None:
        slot = self._request_to_slot.get(request_id)
        if slot is not None:
            return slot
        if not self._free_slots:
            return None
        slot = self._free_slots.pop()
        self._request_to_slot[request_id] = slot
        return slot

    def slot_for(self, request_id: str) -> int | None:
        return self._request_to_slot.get(request_id)

    def resolve_slots(self, request_ids: Sequence[str]) -> tuple[int, ...]:
        slots = []
        for request_id in request_ids:
            slot = self._request_to_slot.get(request_id)
            if slot is None:
                raise RuntimeError(
                    f"Ming-Omni-TTS stream {request_id!r} has no AudioVAE slot"
                )
            slots.append(slot)
        return tuple(slots)

    def reset_and_release(self, request_ids: Sequence[str]) -> None:
        bindings = {
            request_id: self._request_to_slot[request_id]
            for request_id in request_ids
            if request_id in self._request_to_slot
        }
        if not bindings:
            return

        slots = tuple(bindings.values())
        try:
            self._decoder.reset_stream_rows(slots)
        finally:
            for request_id in bindings:
                del self._request_to_slot[request_id]

        for slot in bindings.values():
            self._free_slots.append(slot)

    def release_clean(self, request_ids: Sequence[str]) -> None:
        slots = self.resolve_slots(request_ids)
        for request_id, slot in zip(request_ids, slots, strict=True):
            del self._request_to_slot[request_id]
            self._free_slots.append(slot)

    def reset_all(self) -> None:
        self._decoder.reset_all_stream_rows()
        self._request_to_slot.clear()
        self._free_slots = list(reversed(range(self._decoder.stream_capacity)))


@dataclass(slots=True)
class _StreamState:
    expected_chunk_id: int = 0
    pending_patches: list[torch.Tensor] = field(default_factory=list)
    terminal_received: bool = False
    initial_group_consumed: bool = False
    emitted_samples: int = 0
    terminal_committed: bool = False


@dataclass(frozen=True, slots=True)
class _StreamingStepItem:
    patches: tuple[torch.Tensor, ...]
    terminal: bool


_StreamingStepPlan = tuple[_StreamingStepItem, ...]


class MingTTSStreamingVocoderScheduler(
    StreamingVocoderBase[_StreamState, _StreamingStepPlan]
):
    _can_batch_stream_chunks = True

    def __init__(
        self,
        decoder: MingAudioDecoder,
        *,
        patch_size: int,
        latent_dim: int,
        initial_chunk_patches: int,
        steady_chunk_patches: int,
        keep_latents: bool = False,
    ) -> None:
        self._decoder = decoder
        self._slot_bindings = _AudioVAEStreamingSlotBindings(decoder)
        self._stream_chunk_batch_max = decoder.stream_capacity
        self._patch_size = int(patch_size)
        self._latent_dim = int(latent_dim)
        self._initial_chunk_patches = int(initial_chunk_patches)
        self._steady_chunk_patches = int(steady_chunk_patches)
        self._pending_release_ids: set[str] = set()
        self._stop_requested = threading.Event()
        self._serving_stopped = False
        super().__init__(
            partial(
                decode_ming_tts_audio_payload,
                decoder=decoder,
                keep_latents=bool(keep_latents),
            ),
            sample_rate=decoder.sample_rate,
            stream_source_hint="Ming-Omni-TTS",
            stream_input_modality="audio_latents",
        )

    def stop(self) -> None:
        self._stop_requested.set()
        super().stop()

    def _next_message(self) -> IncomingMessage | None:
        if self._stop_requested.is_set():
            self._running = False
            return None
        msg = super()._next_message()
        if self._stop_requested.is_set():
            self._running = False
            return None
        return msg

    def create_stream_state(self, request_id: str) -> _StreamState:
        del request_id
        return _StreamState()

    def _ingest_stream_item(
        self,
        request_id: str,
        item: StreamItem,
    ) -> _StreamState | None:
        state = self._get_or_create_stream_state(request_id)
        if state is None:
            return None
        metadata = item.metadata
        if not isinstance(metadata, dict):
            raise TypeError(
                f"Ming-Omni-TTS stream chunk for {request_id!r} must include "
                "metadata"
            )
        if item.chunk_id != state.expected_chunk_id:
            raise ValueError(
                f"Ming-Omni-TTS stream chunk for {request_id!r} has "
                f"chunk_id={item.chunk_id}, expected {state.expected_chunk_id}"
            )
        if state.terminal_received:
            raise ValueError(
                f"Ming-Omni-TTS stream chunk arrived after the terminal patch "
                f"for {request_id!r}"
            )
        is_last = metadata.get("is_last")
        if not isinstance(is_last, bool):
            raise TypeError(
                f"Ming-Omni-TTS stream chunk for {request_id!r} must include "
                "boolean metadata['is_last']"
            )
        super()._ingest_stream_item(request_id, item)
        state.expected_chunk_id += 1
        if is_last:
            state.terminal_received = True
        return state

    def validate_chunk(
        self,
        request_id: str,
        state: _StreamState,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        if latents.device.type != "cpu":
            raise ValueError(
                "Ming-Omni-TTS stream latent must be on CPU, "
                f"got device {latents.device}"
            )
        if latents.dtype != torch.float32:
            raise TypeError(
                "Ming-Omni-TTS stream latent dtype must be torch.float32, "
                f"got {latents.dtype}"
            )
        expected_shape = (self._patch_size, self._latent_dim)
        if tuple(latents.shape) != expected_shape:
            raise ValueError(
                f"Ming-Omni-TTS stream latent shape must be {expected_shape}, "
                f"got {tuple(latents.shape)}"
            )
        return latents.contiguous()

    def ingest(
        self,
        request_id: str,
        state: _StreamState,
        latents: torch.Tensor,
    ) -> None:
        del request_id
        state.pending_patches.append(latents)

    def _has_executable_work(self, state: _StreamState) -> bool:
        if state.terminal_committed:
            return False
        if state.terminal_received:
            return bool(state.pending_patches)
        return len(state.pending_patches) >= self._next_chunk_patches(state)

    def _next_chunk_patches(self, state: _StreamState) -> int:
        if state.initial_group_consumed:
            return self._steady_chunk_patches
        return self._initial_chunk_patches

    def select_step_participants(self) -> list[tuple[str, _StreamState]]:
        # Note (yzxiao): External abort only marks a binding dirty; the scheduler
        # thread resets its CUDA row before that slot can be reused.
        self._drain_pending_releases()
        participants = []
        for request_id, state in self._stream_state_items():
            if self._is_aborted(request_id) or not self._has_executable_work(state):
                continue
            if self._slot_bindings.try_bind(request_id) is None:
                continue
            participants.append((request_id, state))
        return participants

    def build_step_plan(
        self,
        participants: list[tuple[str, _StreamState]],
    ) -> _StreamingStepPlan:
        plan = []
        for _, state in participants:
            pending_count = len(state.pending_patches)
            target = self._next_chunk_patches(state)
            if state.terminal_received:
                consume = min(target, pending_count)
                terminal = pending_count <= target
            else:
                consume = target
                terminal = False
            plan.append(
                _StreamingStepItem(
                    patches=tuple(state.pending_patches[:consume]),
                    terminal=terminal,
                )
            )
        return tuple(plan)

    def run_step(
        self,
        participants: list[tuple[str, _StreamState]],
        plan: _StreamingStepPlan,
    ) -> dict[str, torch.Tensor]:
        request_ids = tuple(request_id for request_id, _ in participants)
        slot_ids = self._slot_bindings.resolve_slots(request_ids)
        # Note (yzxiao): The decoder returns owned CPU waveforms all-or-error, so
        # request progress is committed only after it succeeds. Terminal transitions
        # already clean their rows and need no second reset.
        waveforms = self._decoder.run_streaming(
            slot_ids=slot_ids,
            patch_groups=tuple(item.patches for item in plan),
            terminal_flags=tuple(item.terminal for item in plan),
        )
        step_results = tuple(zip(participants, plan, waveforms, strict=True))
        for (_, state), item, waveform in step_results:
            del state.pending_patches[: len(item.patches)]
            if item.terminal:
                state.terminal_committed = True
            elif not state.initial_group_consumed:
                state.initial_group_consumed = True
            state.emitted_samples += int(waveform.numel())

        terminal_request_ids = tuple(
            request_id for (request_id, _), item, _ in step_results if item.terminal
        )
        if terminal_request_ids:
            self._slot_bindings.release_clean(terminal_request_ids)

        return {
            request_id: waveform
            for (request_id, _), _, waveform in step_results
            if waveform.numel() > 0
        }

    def on_step_failure(
        self,
        participants: list[tuple[str, _StreamState]],
        exc: BaseException,
    ) -> list[str]:
        failed = super().on_step_failure(participants, exc)
        self._drain_pending_releases()
        return failed

    def decode_delta(
        self,
        request_id: str,
        state: _StreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        del is_final
        if not state.terminal_received:
            raise RuntimeError(
                f"Ming-Omni-TTS stream {request_id!r} ended without a "
                "terminal latent patch"
            )
        if not state.terminal_committed:
            if self._slot_bindings.slot_for(request_id) is None:
                return None
            raise RuntimeError(
                f"Ming-Omni-TTS stream {request_id!r} ended before its terminal "
                "AudioVAE transition completed"
            )
        if state.emitted_samples <= 0:
            raise RuntimeError(
                f"Ming-Omni-TTS stream {request_id!r} completed without audio"
            )
        return None

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: _StreamState,
    ) -> torch.Tensor:
        del payload
        latents = torch.stack(state.pending_patches, dim=0)
        waveform = self._decoder.decode_full(latents)
        sample_count = int(waveform.numel())
        if sample_count == 0:
            raise RuntimeError(
                f"Ming-Omni-TTS stream {request_id!r} completed without audio"
            )
        state.emitted_samples = sample_count
        return waveform

    def _drain_pending_releases(self) -> None:
        pending = tuple(self._pending_release_ids)
        if not pending:
            return
        try:
            self._slot_bindings.reset_and_release(pending)
        except Exception:
            logger.exception(
                "Ming-Omni-TTS failed to reset AudioVAE rows; their slots "
                "will remain unavailable"
            )
        finally:
            self._pending_release_ids.difference_update(pending)

    def release_stream_resources(
        self,
        request_id: str,
        state: _StreamState,
    ) -> None:
        del state
        if self._slot_bindings.slot_for(request_id) is None:
            return
        self._pending_release_ids.add(request_id)

    def warmup_now(self) -> None:
        self._decoder.prepare_streaming()

    def on_serving_start(self) -> None:
        if self._stop_requested.is_set():
            return
        if not self._decoder.streaming_ready:
            raise RuntimeError(
                "Ming-Omni-TTS streaming AudioVAE backend is not prepared"
            )

    def on_serving_stop(self) -> None:
        if self._serving_stopped:
            return
        self._serving_stopped = True
        try:
            self._slot_bindings.reset_all()
        finally:
            self._pending_release_ids.clear()
            self._decoder.close()

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _StreamState,
    ) -> dict[str, Any]:
        del request_id
        final_state = load_ming_tts_state(payload)
        final_state.sample_rate = int(self._decoder.sample_rate)
        final_state.duration_s = float(
            state.emitted_samples / int(self._decoder.sample_rate)
        )
        data = final_state.to_dict()
        data["modality"] = "audio"
        usage = build_usage(final_state)
        if usage is not None:
            data["usage"] = usage
        return data


__all__ = ["MingTTSStreamingVocoderScheduler"]
