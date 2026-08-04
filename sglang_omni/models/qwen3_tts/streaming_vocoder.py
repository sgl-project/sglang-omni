# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Qwen3-TTS."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Mapping

import torch

from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload

DEFAULT_QWEN3_TTS_STREAM_STRIDE = 16
DEFAULT_QWEN3_TTS_STREAM_FOLLOWUP_STRIDE = 8
DEFAULT_QWEN3_TTS_STREAM_INITIAL_FOLLOWUP_STRIDE = 8
DEFAULT_QWEN3_TTS_INITIAL_CHUNK_FRAMES = 1
DEFAULT_QWEN3_TTS_LEFT_CONTEXT_FRAMES = 16
_QWEN3_TTS_CODEBOOK_SIZE = 2048


@dataclass
class _Qwen3TTSStreamState:
    code_chunks: list[torch.Tensor] = field(default_factory=list)
    total_frames: int = 0
    ref_frames: int = 0
    emitted_generated_frames: int = 0
    next_decode_generated_frames: int = 0
    decoded_chunks: int = 0
    num_quantizers: int | None = None
    pending_ref_frames: int = 0
    initial_chunk_frames: int = DEFAULT_QWEN3_TTS_INITIAL_CHUNK_FRAMES
    initial_pending: bool = False
    followup_pending: bool = False
    final_pending: bool = False
    playback_deadline_s: float = 0.0


@dataclass(frozen=True)
class _Qwen3TTSDecodePlan:
    decoder_input: torch.Tensor
    absolute_emitted_frames: int
    generated_frames: int
    window_start: int


_ASYNC_STOP = None


class Qwen3TTSStreamingVocoderScheduler(
    StreamingVocoderBase[_Qwen3TTSStreamState, None]
):
    """Decode Qwen3-TTS codec frames on a priority CUDA stream."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        device: str,
        stream_stride: int = DEFAULT_QWEN3_TTS_STREAM_STRIDE,
        stream_followup_stride: int = DEFAULT_QWEN3_TTS_STREAM_FOLLOWUP_STRIDE,
        stream_initial_followup_stride: int | None = None,
        initial_chunk_frames: int = DEFAULT_QWEN3_TTS_INITIAL_CHUNK_FRAMES,
        stream_left_context_frames: int = DEFAULT_QWEN3_TTS_LEFT_CONTEXT_FRAMES,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
        async_decode: bool | None = None,
        initial_max_batch_size: int = 32,
        initial_batch_wait_ms: int = 2,
        followup_max_batch_size: int = 8,
        followup_batch_wait_ms: int = 1,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream strides must be > 0")
        if (
            stream_initial_followup_stride is not None
            and stream_initial_followup_stride <= 0
        ):
            raise ValueError("stream_initial_followup_stride must be > 0")
        if initial_chunk_frames < 0:
            raise ValueError("initial_chunk_frames must be >= 0")
        if stream_left_context_frames < 0:
            raise ValueError("stream_left_context_frames must be >= 0")
        if initial_max_batch_size <= 0 or followup_max_batch_size <= 0:
            raise ValueError("async batch sizes must be > 0")
        if initial_batch_wait_ms < 0 or followup_batch_wait_ms < 0:
            raise ValueError("async batch waits must be >= 0")
        self._tokenizer = tokenizer
        self._device = torch.device(device)
        self._decoder = tokenizer.model.decoder
        self._samples_per_frame = int(self._decoder.total_upsample)
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_initial_followup_stride = int(
            min(
                DEFAULT_QWEN3_TTS_STREAM_INITIAL_FOLLOWUP_STRIDE,
                self._stream_followup_stride,
            )
            if stream_initial_followup_stride is None
            else stream_initial_followup_stride
        )
        self._initial_max_batch_size = int(initial_max_batch_size)
        self._initial_batch_wait_s = float(initial_batch_wait_ms) / 1000.0
        self._followup_max_batch_size = int(followup_max_batch_size)
        self._followup_batch_wait_s = float(followup_batch_wait_ms) / 1000.0
        self._default_initial_chunk_frames = int(initial_chunk_frames)
        self._stream_left_context_frames = int(stream_left_context_frames)
        self._async_decode = (
            self._device.type == "cuda" if async_decode is None else bool(async_decode)
        )
        if self._device.type == "cuda":
            least_priority, greatest_priority = torch.cuda.Stream.priority_range()
            followup_priority = min(least_priority, greatest_priority + 1)
            self._decode_stream = torch.cuda.Stream(
                device=self._device,
                priority=greatest_priority,
            )
            self._followup_decode_stream = (
                torch.cuda.Stream(
                    device=self._device,
                    priority=followup_priority,
                )
                if self._async_decode
                else None
            )
        else:
            self._decode_stream = None
            self._followup_decode_stream = None
        self._initial_queue: queue.Queue[tuple[str, _Qwen3TTSStreamState] | None] = (
            queue.Queue()
        )
        self._followup_queue: queue.PriorityQueue[
            tuple[float, int, str, _Qwen3TTSStreamState | None]
        ] = queue.PriorityQueue()
        self._followup_sequence = count()
        self._async_stop = threading.Event()
        self._initial_worker: threading.Thread | None = None
        self._followup_worker: threading.Thread | None = None
        sample_rate = int(tokenizer.get_output_sample_rate())

        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            sample_rate=sample_rate,
            stream_source_hint="Qwen3-TTS",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def start(self) -> None:
        try:
            super().start()
        finally:
            self._join_async_workers()

    def stop(self) -> None:
        self._signal_async_stop()
        super().stop()
        self._join_async_workers()

    def on_serving_start(self) -> None:
        if not self._async_decode:
            return
        self._initial_queue = queue.Queue()
        self._followup_queue = queue.PriorityQueue()
        self._followup_sequence = count()
        self._async_stop.clear()
        self._initial_worker = threading.Thread(
            target=self._run_initial_worker,
            name="qwen3-tts-vocoder-initial",
            daemon=True,
        )
        self._followup_worker = threading.Thread(
            target=self._run_followup_worker,
            name="qwen3-tts-vocoder-followup",
            daemon=True,
        )
        self._initial_worker.start()
        self._followup_worker.start()

    def on_serving_stop(self) -> None:
        self._signal_async_stop()

    def _signal_async_stop(self) -> None:
        if self._async_stop.is_set():
            return
        self._async_stop.set()
        if self._initial_worker is not None:
            self._initial_queue.put(_ASYNC_STOP)
        if self._followup_worker is not None:
            self._followup_queue.put(
                (float("inf"), next(self._followup_sequence), "", _ASYNC_STOP)
            )

    def _join_async_workers(self) -> None:
        for worker in (self._initial_worker, self._followup_worker):
            if worker is not None and worker is not threading.current_thread():
                worker.join()
        self._initial_worker = None
        self._followup_worker = None

    def create_stream_state(self, request_id: str) -> _Qwen3TTSStreamState:
        del request_id
        return _Qwen3TTSStreamState(
            initial_chunk_frames=self._default_initial_chunk_frames
        )

    def latch_stream_contract(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            params = source.request.params
            if isinstance(params, Mapping):
                state.initial_chunk_frames = resolve_initial_codec_chunk_frames(
                    params,
                    steady_chunk_frames=self._stream_stride,
                    default_frames=self._default_initial_chunk_frames,
                )
            return

        metadata: Mapping[str, Any] = source
        if "num_quantizers" not in metadata and state.num_quantizers is None:
            raise RuntimeError(
                f"Qwen3-TTS stream chunk for {request_id!r} is missing num_quantizers"
            )
        if "num_quantizers" in metadata:
            num_quantizers = int(metadata["num_quantizers"])
            if num_quantizers <= 0:
                raise ValueError("Qwen3-TTS num_quantizers must be > 0")
            if (
                state.num_quantizers is not None
                and state.num_quantizers != num_quantizers
            ):
                raise ValueError(
                    f"Qwen3-TTS num_quantizers changed for {request_id!r}: "
                    f"{state.num_quantizers} -> {num_quantizers}"
                )
            state.num_quantizers = num_quantizers
        if "ref_code_len" in metadata:
            ref_frames = int(metadata["ref_code_len"])
            if ref_frames < 0:
                raise ValueError("Qwen3-TTS ref_code_len must be >= 0")
            if state.total_frames or state.ref_frames:
                raise ValueError(
                    f"Qwen3-TTS reference codes arrived after stream start for "
                    f"{request_id!r}"
                )
            state.pending_ref_frames = ref_frames
        if INITIAL_CODEC_CHUNK_FRAMES_PARAM in metadata:
            state.initial_chunk_frames = resolve_initial_codec_chunk_frames(
                metadata,
                steady_chunk_frames=self._stream_stride,
                default_frames=self._default_initial_chunk_frames,
            )

    def validate_chunk(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        chunk = codes.detach().to(dtype=torch.long)
        if chunk.ndim == 1:
            chunk = chunk.unsqueeze(0)
        elif chunk.ndim != 2:
            raise ValueError(
                f"Qwen3-TTS stream chunk must be [Q] or [T, Q], "
                f"got {tuple(chunk.shape)}"
            )
        if chunk.shape[0] == 0:
            raise ValueError("Qwen3-TTS stream chunk must not be empty")
        if state.num_quantizers is None:
            raise RuntimeError(
                f"Qwen3-TTS stream contract for {request_id!r} is missing "
                "num_quantizers"
            )
        if int(chunk.shape[1]) != state.num_quantizers:
            raise ValueError(
                f"Qwen3-TTS stream chunk has {int(chunk.shape[1])} quantizers, "
                f"expected {state.num_quantizers}"
            )
        if not chunk.is_cuda and (
            bool((chunk < 0).any()) or bool((chunk >= _QWEN3_TTS_CODEBOOK_SIZE).any())
        ):
            raise ValueError(
                f"Qwen3-TTS stream chunk for {request_id!r} contains codec ids "
                f"outside [0, {_QWEN3_TTS_CODEBOOK_SIZE})"
            )
        return chunk

    def ingest(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        if state.pending_ref_frames:
            if state.pending_ref_frames >= int(codes.shape[0]):
                raise ValueError(
                    "Qwen3-TTS first stream chunk must include at least one "
                    "generated codec frame after the reference"
                )
            state.ref_frames = state.pending_ref_frames
            state.pending_ref_frames = 0
        state.code_chunks.append(codes)
        state.total_frames += int(codes.shape[0])

    def should_decode(self, state: _Qwen3TTSStreamState, *, is_final: bool) -> bool:
        if is_final:
            return True
        generated_frames = state.total_frames - state.ref_frames
        next_frames = self._next_decode_threshold(state)
        return generated_frames >= next_frames

    def _next_decode_threshold(self, state: _Qwen3TTSStreamState) -> int:
        if state.next_decode_generated_frames:
            return state.next_decode_generated_frames
        return state.initial_chunk_frames or self._stream_stride

    def decode_delta(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        del request_id
        plan = self._build_decode_plan(state, is_final=is_final)
        if plan is None:
            return None
        waveform = self._run_decode_plan(plan, stream=self._decode_stream)
        return self._commit_decode_plan(state, plan, waveform)

    def _build_decode_plan(
        self,
        state: _Qwen3TTSStreamState,
        *,
        is_final: bool,
        max_generated_frames: int | None = None,
    ) -> _Qwen3TTSDecodePlan | None:
        available_generated_frames = state.total_frames - state.ref_frames
        if available_generated_frames <= state.emitted_generated_frames:
            return None
        next_frames = self._next_decode_threshold(state)
        if not is_final and available_generated_frames < next_frames:
            state.next_decode_generated_frames = next_frames
            return None

        generated_frames = available_generated_frames
        if max_generated_frames is not None:
            generated_frames = min(generated_frames, max_generated_frames)

        absolute_emitted = state.ref_frames + state.emitted_generated_frames
        window_start = max(0, absolute_emitted - self._stream_left_context_frames)
        window_end = state.ref_frames + generated_frames
        codes = torch.cat(state.code_chunks, dim=0)
        decoder_input = codes[window_start:window_end].transpose(0, 1).unsqueeze(0)
        return _Qwen3TTSDecodePlan(
            decoder_input=decoder_input,
            absolute_emitted_frames=absolute_emitted,
            generated_frames=generated_frames,
            window_start=window_start,
        )

    def _run_decode_plan(
        self,
        plan: _Qwen3TTSDecodePlan,
        *,
        stream: torch.cuda.Stream | None,
    ) -> torch.Tensor:
        return self._run_decode_plans([plan], stream=stream)[0]

    def _run_decode_plans(
        self,
        plans: list[_Qwen3TTSDecodePlan],
        *,
        stream: torch.cuda.Stream | None,
    ) -> list[torch.Tensor]:
        decoder_input = torch.cat([plan.decoder_input for plan in plans], dim=0)
        with torch.inference_mode():
            if stream is None:
                waveform = self._decoder.chunked_decode(decoder_input)
            else:
                stream.wait_stream(torch.cuda.current_stream(self._device))
                with torch.cuda.stream(stream):
                    waveform = self._decoder.chunked_decode(
                        decoder_input.to(self._device)
                    )
                stream.synchronize()
        if waveform.ndim == 3:
            if waveform.shape[0] != len(plans):
                raise RuntimeError(
                    "Qwen3-TTS streaming decoder returned the wrong batch size"
                )
            return [waveform[index, 0] for index in range(len(plans))]
        elif waveform.ndim == 2:
            if len(plans) != 1:
                raise RuntimeError(
                    "Qwen3-TTS streaming decoder dropped the batch dimension"
                )
            return [waveform[0]]
        raise ValueError(
            "Qwen3-TTS decoder returned unexpected waveform shape "
            f"{tuple(waveform.shape)}"
        )

    def _commit_decode_plan(
        self,
        state: _Qwen3TTSStreamState,
        plan: _Qwen3TTSDecodePlan,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        expected_emitted = plan.absolute_emitted_frames - state.ref_frames
        if state.emitted_generated_frames != expected_emitted:
            raise RuntimeError("Qwen3-TTS streaming decode plan committed out of order")
        trim_frames = plan.absolute_emitted_frames - plan.window_start
        trim_samples = min(
            trim_frames * self._samples_per_frame,
            int(waveform.shape[-1]),
        )
        new_frames = plan.generated_frames - state.emitted_generated_frames
        emit_samples = new_frames * self._samples_per_frame
        delta = waveform[trim_samples : trim_samples + emit_samples]
        if delta.numel() == 0:
            raise RuntimeError("Qwen3-TTS streaming decoder returned an empty delta")

        state.emitted_generated_frames = plan.generated_frames
        state.decoded_chunks += 1
        followup_stride = (
            self._stream_initial_followup_stride
            if state.decoded_chunks == 1
            else self._stream_followup_stride
        )
        state.next_decode_generated_frames = plan.generated_frames + followup_stride
        delta = delta.detach().to(torch.float32).contiguous()
        now = time.monotonic()
        duration_s = float(delta.numel()) / float(self._sample_rate)
        state.playback_deadline_s = max(state.playback_deadline_s, now) + duration_s
        return delta

    def _decode_and_emit(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
    ) -> list[OutgoingMessage]:
        if not self.should_decode(state, is_final=False):
            return []
        if self._async_decode:
            if state.decoded_chunks:
                self._schedule_followup(request_id, state)
            else:
                self._schedule_initial(request_id, state)
            return []
        waveform = self.decode_delta(request_id, state, is_final=False)
        if waveform is None:
            return []

        self._mark_stream_emitted(request_id)
        split_samples = state.initial_chunk_frames * self._samples_per_frame
        if (
            state.decoded_chunks == 1
            and split_samples > 0
            and split_samples < int(waveform.shape[-1])
        ):
            return [
                self._stream_chunk_message(request_id, waveform[:split_samples]),
                self._stream_chunk_message(request_id, waveform[split_samples:]),
            ]
        return [self._stream_chunk_message(request_id, waveform)]

    def _schedule_initial(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
    ) -> None:
        if state.initial_pending:
            return
        if self._initial_worker is None:
            raise RuntimeError("Qwen3-TTS initial decoder is not running")
        state.initial_pending = True
        self._initial_queue.put((request_id, state))

    def _schedule_followup(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
    ) -> None:
        if state.followup_pending:
            return
        if self._followup_worker is None:
            raise RuntimeError("Qwen3-TTS follow-up decoder is not running")
        state.followup_pending = True
        self._enqueue_followup(request_id, state)

    def _enqueue_followup(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
    ) -> None:
        self._followup_queue.put(
            (
                state.playback_deadline_s,
                next(self._followup_sequence),
                request_id,
                state,
            )
        )

    def _collect_async_batch(
        self,
        work_queue: queue.Queue[tuple[str, _Qwen3TTSStreamState] | None],
        *,
        max_batch_size: int,
        batch_wait_s: float,
    ) -> list[tuple[str, _Qwen3TTSStreamState]] | None:
        queued = work_queue.get()
        if queued is None or self._async_stop.is_set():
            return None
        batch = [queued]
        deadline = time.monotonic() + batch_wait_s
        while len(batch) < max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                next_queued = work_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if next_queued is None:
                return None
            batch.append(next_queued)
        return batch

    def _run_initial_worker(self) -> None:
        while True:
            batch = self._collect_async_batch(
                self._initial_queue,
                max_batch_size=self._initial_max_batch_size,
                batch_wait_s=self._initial_batch_wait_s,
            )
            if batch is None:
                return
            self._run_initial_batch(batch)

    def _run_initial_batch(
        self,
        batch: list[tuple[str, _Qwen3TTSStreamState]],
    ) -> None:
        planned: list[tuple[str, _Qwen3TTSStreamState, _Qwen3TTSDecodePlan]] = []
        with self._state_lock:
            for request_id, state in batch:
                if (
                    self._stream_states.get(request_id) is not state
                    or state.decoded_chunks
                ):
                    continue
                plan = self._build_decode_plan(
                    state,
                    is_final=state.final_pending,
                    max_generated_frames=(
                        state.initial_chunk_frames or self._stream_stride
                    ),
                )
                if plan is None:
                    state.initial_pending = False
                    continue
                planned.append((request_id, state, plan))

        for group in self._group_decode_plans(planned):
            try:
                waveforms = self._run_decode_plans(
                    [entry[2] for entry in group],
                    stream=self._decode_stream,
                )
            except Exception as exc:
                for request_id, state, _ in group:
                    self._fail_async_stream(request_id, state, exc)
                continue
            for entry, waveform in zip(group, waveforms):
                request_id, state, plan = entry
                self._commit_initial(request_id, state, plan, waveform)

    @staticmethod
    def _group_decode_plans(
        planned: list[tuple[str, _Qwen3TTSStreamState, _Qwen3TTSDecodePlan]],
    ) -> list[list[tuple[str, _Qwen3TTSStreamState, _Qwen3TTSDecodePlan]]]:
        groups: dict[
            tuple[int, ...],
            list[tuple[str, _Qwen3TTSStreamState, _Qwen3TTSDecodePlan]],
        ] = {}
        for entry in planned:
            groups.setdefault(tuple(entry[2].decoder_input.shape), []).append(entry)
        return list(groups.values())

    def _commit_initial(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        plan: _Qwen3TTSDecodePlan,
        waveform: torch.Tensor,
    ) -> None:
        cleanup_abort = False
        with self._state_lock:
            if self._stream_states.get(request_id) is not state:
                return
            try:
                delta = self._commit_decode_plan(state, plan, waveform)
            except Exception as exc:
                self._emit_error(request_id, exc)
                self._abort_state(request_id)
                cleanup_abort = True
            else:
                state.initial_pending = False
                if not self._is_aborted(request_id):
                    self._mark_stream_emitted(request_id)
                    self.outbox.put(self._stream_chunk_message(request_id, delta))
                has_remainder = (
                    state.total_frames - state.ref_frames
                    > state.emitted_generated_frames
                )
                if state.final_pending and not has_remainder:
                    self._finish_async_stream(request_id, state)
                elif state.final_pending or self.should_decode(state, is_final=False):
                    self._schedule_followup(request_id, state)
        if cleanup_abort:
            self._cleanup_aborted_request(request_id)

    def _run_followup_worker(self) -> None:
        while True:
            batch = self._collect_followup_batch()
            if batch is None:
                return
            self._run_followup_batch(batch)

    def _collect_followup_batch(
        self,
    ) -> list[tuple[str, _Qwen3TTSStreamState]] | None:
        _, _, request_id, state = self._followup_queue.get()
        if state is None or self._async_stop.is_set():
            return None
        batch = [(request_id, state)]
        deadline = time.monotonic() + self._followup_batch_wait_s
        while len(batch) < self._followup_max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                _, _, request_id, state = self._followup_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if state is None:
                return None
            batch.append((request_id, state))
        return batch

    def _run_followup_batch(
        self,
        batch: list[tuple[str, _Qwen3TTSStreamState]],
    ) -> None:
        planned: list[tuple[str, _Qwen3TTSStreamState, _Qwen3TTSDecodePlan]] = []
        with self._state_lock:
            for request_id, state in batch:
                if self._stream_states.get(request_id) is not state:
                    continue
                plan = self._build_decode_plan(
                    state,
                    is_final=state.final_pending,
                    max_generated_frames=self._next_decode_threshold(state),
                )
                if plan is None:
                    state.followup_pending = False
                    if state.final_pending:
                        self._finish_async_stream(request_id, state)
                    continue
                planned.append((request_id, state, plan))

        for group in self._group_decode_plans(planned):
            try:
                waveforms = self._run_decode_plans(
                    [entry[2] for entry in group],
                    stream=self._followup_decode_stream,
                )
            except Exception as exc:
                for request_id, state, _ in group:
                    self._fail_async_stream(request_id, state, exc)
                continue
            for entry, waveform in zip(group, waveforms):
                request_id, state, plan = entry
                self._commit_followup(request_id, state, plan, waveform)

    def _commit_followup(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        plan: _Qwen3TTSDecodePlan,
        waveform: torch.Tensor,
    ) -> None:
        cleanup_abort = False
        with self._state_lock:
            if self._stream_states.get(request_id) is not state:
                return
            try:
                delta = self._commit_decode_plan(state, plan, waveform)
            except Exception as exc:
                self._emit_error(request_id, exc)
                self._abort_state(request_id)
                cleanup_abort = True
            else:
                if not self._is_aborted(request_id):
                    self._mark_stream_emitted(request_id)
                    self.outbox.put(self._stream_chunk_message(request_id, delta))
                has_remainder = (
                    state.total_frames - state.ref_frames
                    > state.emitted_generated_frames
                )
                if state.final_pending and not has_remainder:
                    state.followup_pending = False
                    self._finish_async_stream(request_id, state)
                elif state.final_pending or self.should_decode(state, is_final=False):
                    self._enqueue_followup(request_id, state)
                else:
                    state.followup_pending = False
        if cleanup_abort:
            self._cleanup_aborted_request(request_id)

    def _fail_async_stream(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        exc: BaseException,
    ) -> None:
        cleanup_abort = False
        with self._state_lock:
            if self._stream_states.get(request_id) is state:
                self._emit_error(request_id, exc)
                self._abort_state(request_id)
                cleanup_abort = True
        if cleanup_abort:
            self._cleanup_aborted_request(request_id)

    def _handle_stream_done(self, request_id: str) -> None:
        with self._state_lock:
            if request_id not in self._stream_payloads:
                if request_id in self._completed_non_streaming_request_ids:
                    return
                self._pending_done.add(request_id)
                return
            state = self._get_or_create_stream_state(request_id)
            if (
                self._async_decode
                and state is not None
                and (state.initial_pending or state.decoded_chunks)
            ):
                state.final_pending = True
                if not state.initial_pending:
                    self._schedule_followup(request_id, state)
                return
        super()._handle_stream_done(request_id)

    def _finish_async_stream(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
    ) -> None:
        payload = self._stream_payloads.get(request_id)
        if payload is None or self._is_aborted(request_id):
            return
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=self.final_result_data(request_id, payload, state),
                ),
            )
        )
        self._record_completed_stream_request_id(request_id)
        self._clear_request_state(request_id)

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: _Qwen3TTSStreamState,
    ) -> torch.Tensor | None:
        del request_id, state
        return self._decode_state_audio(Qwen3TTSState.from_dict(payload.data))

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _Qwen3TTSStreamState,
    ) -> dict[str, Any]:
        del request_id, state
        final_state = Qwen3TTSState.from_dict(payload.data)
        data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = build_usage(final_state)
        if usage is not None:
            data["usage"] = usage
        return data

    async def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        return (await self._vocode_payloads([payload]))[0]

    async def _vocode_payloads(
        self, payloads: list[StagePayload]
    ) -> list[StagePayload]:
        states = [Qwen3TTSState.from_dict(payload.data) for payload in payloads]
        codes = []
        for state in states:
            if state.audio_codes is None:
                raise RuntimeError(
                    "Qwen3-TTS vocoder requires audio_codes from tts_engine"
                )
            codes.append(torch.as_tensor(state.audio_codes, dtype=torch.long))

        wavs, sample_rate = self._tokenizer.decode(
            [{"audio_codes": item} for item in codes]
        )
        if len(wavs) != len(payloads):
            raise RuntimeError(
                f"Qwen3-TTS speech tokenizer returned {len(wavs)} audios for "
                f"{len(payloads)} requests"
            )
        return [
            self._store_vocoder_result(payload, state, wav, sample_rate)
            for payload, state, wav in zip(payloads, states, wavs)
        ]

    def _store_vocoder_result(
        self,
        payload: StagePayload,
        state: Qwen3TTSState,
        waveform: Any,
        sample_rate: int,
    ) -> StagePayload:
        if waveform is None:
            raise RuntimeError("Qwen3-TTS speech tokenizer did not return audio")
        if state.ref_code_len:
            total_frames = len(state.audio_codes)
            cut = int(state.ref_code_len / max(total_frames, 1) * waveform.shape[0])
            waveform = waveform[cut:]

        data = audio_waveform_payload(
            waveform,
            sample_rate=int(sample_rate),
            modality="audio",
            source_hint="Qwen3-TTS",
        )
        usage = build_usage(state)
        if usage is not None:
            data["usage"] = usage
        payload.data = data
        return payload

    def _decode_state_audio(self, state: Qwen3TTSState) -> torch.Tensor | None:
        if state.audio_codes is None:
            return None
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        wavs, _ = self._tokenizer.decode([{"audio_codes": codes}])
        if not wavs:
            return None
        waveform = torch.as_tensor(wavs[0], dtype=torch.float32)
        if state.ref_code_len:
            total_frames = len(codes)
            cut = int(state.ref_code_len / max(total_frames, 1) * waveform.shape[0])
            waveform = waveform[cut:]
        return waveform.contiguous()


__all__ = ["Qwen3TTSStreamingVocoderScheduler"]
