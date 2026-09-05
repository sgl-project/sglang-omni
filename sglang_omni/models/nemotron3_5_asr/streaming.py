# SPDX-License-Identifier: Apache-2.0
"""Native cache-aware PCM streaming scheduler for Nemotron 3.5 ASR."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler

from .model_runner import (
    Nemotron3_5ASRDecodeState,
    Nemotron3_5ASRModelRunner,
    Nemotron3_5ASRStreamingBatchResult,
)
from .request_builders import (
    normalize_nemotron_language,
    validate_nemotron_greedy_params,
)

_PCM16_SCALE = 32768.0


@dataclass(frozen=True, slots=True)
class Nemotron3_5ASRStreamingChunkSpec:
    sample_rate: int
    first_samples: int
    subsequent_samples: int
    first_frames: int
    subsequent_frames: int
    hop_length: int
    n_fft: int
    streaming_latency_ms: int

    @classmethod
    def from_runner(
        cls, runner: Nemotron3_5ASRModelRunner
    ) -> Nemotron3_5ASRStreamingChunkSpec:
        return cls(**runner.streaming_chunk_spec)


@dataclass(frozen=True, slots=True)
class Nemotron3_5ASRAudioWindow:
    waveform: np.ndarray
    model_chunk_index: int
    is_first: bool
    is_final: bool
    raw_start_sample: int
    real_samples: int
    left_padding_samples: int
    right_padding_samples: int
    ready_wait_s: float


@dataclass(slots=True)
class Nemotron3_5ASRStreamMetrics:
    request_started_s: float = field(default_factory=time.perf_counter)
    first_packet_s: float | None = None
    input_done_s: float | None = None
    first_text_s: float | None = None
    finalized_s: float | None = None
    model_compute_s: float = 0.0
    packet_count: int = 0
    model_chunk_count: int = 0
    cache_reuse_count: int = 0
    max_queue_depth: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    chunk_latency_ms: list[float] = field(default_factory=list)
    chunk_ready_wait_ms: list[float] = field(default_factory=list)


@dataclass(slots=True)
class Nemotron3_5ASRStreamState:
    request_id: str
    payload: StagePayload
    language: str
    spec: Nemotron3_5ASRStreamingChunkSpec
    decode: Nemotron3_5ASRDecodeState
    max_new_tokens: int | None = None
    pcm_bytes: bytearray = field(default_factory=bytearray)
    total_samples: int = 0
    covered_audio_end: int = 0
    model_chunk_index: int = 0
    next_mel_frame: int = 0
    input_done: bool = False
    ready_since_s: float | None = None
    raw_text: str = ""
    clean_text: str = ""
    detected_language: str | None = None
    metrics: Nemotron3_5ASRStreamMetrics = field(
        default_factory=Nemotron3_5ASRStreamMetrics
    )

    def append_pcm16(self, tensor: torch.Tensor, metadata: Mapping[str, Any]) -> None:
        if tensor.device.type != "cpu":
            raise ValueError("Nemotron streaming chunks must be CPU PCM16 tensors")
        if tensor.dtype not in {torch.int16, torch.uint8}:
            raise TypeError(
                "Nemotron streaming chunks must use PCM16 samples (torch.int16) "
                f"or raw little-endian bytes (torch.uint8), got {tensor.dtype}"
            )
        if tensor.ndim not in {1, 2}:
            raise ValueError(
                "Nemotron streaming PCM16 tensors must be one-dimensional or mono"
            )
        if tensor.ndim == 2 and 1 not in tensor.shape:
            raise ValueError("Nemotron streaming accepts mono PCM16 only")
        sample_rate = metadata.get("sample_rate", self.spec.sample_rate)
        if isinstance(sample_rate, bool) or int(sample_rate) != self.spec.sample_rate:
            raise ValueError(
                f"Nemotron streaming requires sample_rate={self.spec.sample_rate}"
            )
        modality = metadata.get("modality")
        if modality not in {None, "audio", "pcm16"}:
            raise ValueError(
                f"Nemotron streaming chunk modality must be audio or pcm16, got {modality!r}"
            )
        if self.input_done:
            raise RuntimeError(f"Nemotron stream {self.request_id!r} is already done")

        flat = tensor.detach().contiguous().reshape(-1)
        if flat.numel() == 0:
            raise ValueError("Nemotron streaming PCM16 chunks must not be empty")
        if flat.dtype == torch.int16:
            packet = flat.numpy().astype("<i2", copy=False).tobytes()
        else:
            packet = flat.numpy().tobytes()
        self.pcm_bytes.extend(packet)
        self.total_samples = len(self.pcm_bytes) // 2
        now = time.perf_counter()
        if self.metrics.first_packet_s is None:
            self.metrics.first_packet_s = now
        self.metrics.packet_count += 1
        self._mark_ready(now)

    @property
    def decode_limit_reached(self) -> bool:
        return (
            self.max_new_tokens is not None
            and self.decode.decoder_steps >= self.max_new_tokens
        )

    def mark_done(self) -> None:
        if self.input_done:
            raise RuntimeError(f"Nemotron stream {self.request_id!r} is already done")
        if self.total_samples == 0:
            raise ValueError("Nemotron streaming input contains no PCM16 samples")
        if len(self.pcm_bytes) % 2:
            raise ValueError(
                "Nemotron streaming input ends with an incomplete PCM16 sample"
            )
        self.input_done = True
        self.metrics.input_done_s = time.perf_counter()
        self._mark_ready(self.metrics.input_done_s)

    def _next_bounds(self) -> tuple[int, int]:
        if self.model_chunk_index == 0:
            return 0, self.spec.first_samples
        start = self.next_mel_frame * self.spec.hop_length - self.spec.n_fft // 2
        return start, start + self.spec.subsequent_samples

    def has_ready_window(self, *, finalizing: bool = False) -> bool:
        if self.model_chunk_index == 0:
            return self.total_samples >= self.spec.first_samples or (
                finalizing and self.total_samples > 0
            )
        _, end = self._next_bounds()
        if self.total_samples >= end:
            return True
        return finalizing and self.total_samples > self.covered_audio_end

    def _mark_ready(self, now: float | None = None) -> None:
        if self.ready_since_s is None and self.has_ready_window(
            finalizing=self.input_done
        ):
            self.ready_since_s = now if now is not None else time.perf_counter()

    def pop_ready_window(
        self, *, finalizing: bool = False
    ) -> Nemotron3_5ASRAudioWindow:
        if not self.has_ready_window(finalizing=finalizing):
            raise RuntimeError(
                f"Nemotron stream {self.request_id!r} has no ready window"
            )
        now = time.perf_counter()
        start, end = self._next_bounds()
        size = end - start
        source_start = max(start, 0)
        source_end = min(end, self.total_samples)
        complete_bytes = memoryview(self.pcm_bytes)[: self.total_samples * 2]
        raw = np.frombuffer(complete_bytes, dtype="<i2")
        real = raw[source_start : max(source_start, source_end)]
        left_padding = max(-start, 0)
        right_padding = size - left_padding - int(real.shape[0])
        waveform = np.pad(
            real.astype(np.float32) / _PCM16_SCALE,
            (left_padding, right_padding),
        ).astype(np.float32, copy=False)
        is_first = self.model_chunk_index == 0
        is_final = finalizing and end >= self.total_samples
        ready_wait_s = (
            max(now - self.ready_since_s, 0.0)
            if self.ready_since_s is not None
            else 0.0
        )
        window = Nemotron3_5ASRAudioWindow(
            waveform=waveform,
            model_chunk_index=self.model_chunk_index,
            is_first=is_first,
            is_final=is_final,
            raw_start_sample=start,
            real_samples=int(real.shape[0]),
            left_padding_samples=left_padding,
            right_padding_samples=right_padding,
            ready_wait_s=ready_wait_s,
        )

        self.covered_audio_end = max(
            self.covered_audio_end, min(end, self.total_samples)
        )
        self.model_chunk_index += 1
        if is_first:
            self.next_mel_frame = self.spec.first_frames
        else:
            self.next_mel_frame += self.spec.subsequent_frames
        self.ready_since_s = None
        self._mark_ready(now)
        return window


class Nemotron3_5ASRStreamingScheduler(StreamingSimpleScheduler):
    """Offline batch scheduler plus request-owned native RNNT streaming state."""

    supports_external_input_stream = True
    _can_batch_stream_chunks = True
    _stream_chunk_batch_distinct_requests = True

    def __init__(
        self,
        runner: Nemotron3_5ASRModelRunner,
        compute_fn,
        *,
        batch_compute_fn,
        prompt_dictionary: Mapping[str, int],
        max_batch_size: int,
        max_batch_wait_ms: float,
        max_pending_messages: int,
    ) -> None:
        self.runner = runner
        self.chunk_spec = Nemotron3_5ASRStreamingChunkSpec.from_runner(runner)
        self.prompt_dictionary = dict(prompt_dictionary)
        self._stream_states: dict[str, Nemotron3_5ASRStreamState] = {}
        self._closed = False
        self._aggregate = {
            "input_packets": 0,
            "model_chunks": 0,
            "model_batches": 0,
            "cache_reuses": 0,
            "audio_samples": 0,
            "model_compute_s": 0.0,
            "max_batch_size": 0,
            "completed_streams": 0,
            "aborted_streams": 0,
        }
        self._stream_chunk_batch_max = max_batch_size
        super().__init__(
            compute_fn,
            batch_compute_fn=batch_compute_fn,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            max_pending_messages=max_pending_messages,
        )

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return bool(getattr(payload, "external_input_stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        if request_id in self._stream_states:
            raise ValueError(f"Nemotron stream {request_id!r} already exists")
        params = payload.request.params or {}
        max_new_tokens = validate_nemotron_greedy_params(params)
        language = normalize_nemotron_language(
            params.get("language"), self.prompt_dictionary
        )
        self._stream_states[request_id] = Nemotron3_5ASRStreamState(
            request_id=request_id,
            payload=payload,
            language=language,
            spec=self.chunk_spec,
            decode=self.runner.new_streaming_decode_state(),
            max_new_tokens=max_new_tokens,
        )

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        raise RuntimeError(
            "Nemotron streaming chunks must use the cross-request batch path"
        )

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        failed: list[str] = []
        with self._state_lock:
            touched: set[str] = set()
            for request_id, item in items:
                if self._is_aborted(request_id):
                    continue
                try:
                    state = self._require_stream_state(request_id)
                    metadata = item.metadata or {}
                    if not isinstance(metadata, dict):
                        raise TypeError(
                            "Nemotron streaming chunk metadata must be a dict"
                        )
                    if not isinstance(item.data, torch.Tensor):
                        raise TypeError(
                            "Nemotron streaming chunks must carry torch.Tensor"
                        )
                    samples_before = state.total_samples
                    state.append_pcm16(item.data, metadata)
                    state.metrics.max_queue_depth = max(
                        state.metrics.max_queue_depth, self.inbox.qsize()
                    )
                    self._aggregate["input_packets"] += 1
                    self._aggregate["audio_samples"] += (
                        state.total_samples - samples_before
                    )
                    touched.add(request_id)
                except Exception as exc:
                    self._emit_error(request_id, exc)
                    self._abort_state(request_id)
                    self._aggregate["aborted_streams"] += 1
                    failed.append(request_id)
            failed.extend(self._process_one_ready_window_per_request(touched))
        for request_id in dict.fromkeys(failed):
            self._cleanup_aborted_request(request_id)

    def _process_one_ready_window_per_request(self, request_ids: set[str]) -> list[str]:
        ready: list[
            tuple[str, Nemotron3_5ASRStreamState, Nemotron3_5ASRAudioWindow]
        ] = []
        for request_id in self._stream_states:
            if request_id not in request_ids or self._is_aborted(request_id):
                continue
            state = self._stream_states[request_id]
            if not state.decode_limit_reached and state.has_ready_window():
                ready.append((request_id, state, state.pop_ready_window()))
        return self._run_ready_windows(ready)

    def _run_ready_windows(
        self,
        ready: Sequence[
            tuple[str, Nemotron3_5ASRStreamState, Nemotron3_5ASRAudioWindow]
        ],
    ) -> list[str]:
        groups: dict[
            tuple[int, bool],
            list[tuple[str, Nemotron3_5ASRStreamState, Nemotron3_5ASRAudioWindow]],
        ] = defaultdict(list)
        for item in ready:
            groups[(item[2].model_chunk_index, item[2].is_first)].append(item)

        failed: list[str] = []
        for group in groups.values():
            for offset in range(0, len(group), self._max_batch_size):
                batch = group[offset : offset + self._max_batch_size]
                try:
                    prepared = [
                        self.runner.prepare_streaming_chunk(
                            window.waveform,
                            language=state.language,
                            is_first=window.is_first,
                        )
                        for _, state, window in batch
                    ]
                    result = self.runner.run_streaming_batch(
                        [state.decode for _, state, _ in batch],
                        prepared,
                        requested_languages=[state.language for _, state, _ in batch],
                        max_new_tokens=[state.max_new_tokens for _, state, _ in batch],
                    )
                    self._record_batch(batch, result)
                    for index, (request_id, state, _) in enumerate(batch):
                        message = self._partial_message(state, result, index)
                        if message is not None and not self._is_aborted(request_id):
                            self.outbox.put(message)
                except Exception as exc:
                    for request_id, _, _ in batch:
                        self._emit_error(request_id, exc)
                        self._abort_state(request_id)
                        self._aggregate["aborted_streams"] += 1
                        failed.append(request_id)
        return failed

    def _record_batch(
        self,
        batch: Sequence[
            tuple[str, Nemotron3_5ASRStreamState, Nemotron3_5ASRAudioWindow]
        ],
        result: Nemotron3_5ASRStreamingBatchResult,
    ) -> None:
        batch_size = len(batch)
        self._aggregate["model_batches"] += 1
        self._aggregate["model_chunks"] += batch_size
        self._aggregate["model_compute_s"] += result.elapsed_s
        self._aggregate["max_batch_size"] = max(
            self._aggregate["max_batch_size"], batch_size
        )
        per_request_compute_s = result.elapsed_s / batch_size
        for _, state, window in batch:
            state.metrics.model_compute_s += per_request_compute_s
            state.metrics.model_chunk_count += 1
            state.metrics.batch_sizes.append(batch_size)
            state.metrics.chunk_latency_ms.append(result.elapsed_s * 1000.0)
            state.metrics.chunk_ready_wait_ms.append(window.ready_wait_s * 1000.0)
            if window.model_chunk_index > 0:
                state.metrics.cache_reuse_count += 1
                self._aggregate["cache_reuses"] += 1

    def _partial_message(
        self,
        state: Nemotron3_5ASRStreamState,
        result: Nemotron3_5ASRStreamingBatchResult,
        index: int,
    ) -> OutgoingMessage | None:
        previous = state.clean_text
        state.raw_text = result.raw_texts[index]
        state.clean_text = result.clean_texts[index]
        state.detected_language = result.languages[index]
        if not state.clean_text or state.clean_text == previous:
            return None
        if previous and not state.clean_text.startswith(previous):
            raise RuntimeError(
                "Nemotron streaming transcript changed a previously emitted prefix"
            )
        delta = state.clean_text[len(previous) :]
        if not delta:
            return None
        now = time.perf_counter()
        if state.metrics.first_text_s is None:
            state.metrics.first_text_s = now
        return OutgoingMessage(
            request_id=state.request_id,
            type="stream",
            data={
                "text": delta,
                "full_text": state.clean_text,
                "raw_text": state.raw_text,
                "language": state.detected_language,
                "token_ids": list(state.decode.tokens),
                "modality": "text",
                "metrics": self._metrics_snapshot(state, now=now),
            },
            metadata={"modality": "text"},
        )

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self._require_stream_state(request_id)
        state.mark_done()
        messages: list[OutgoingMessage] = []
        while not state.decode_limit_reached and state.has_ready_window(
            finalizing=True
        ):
            window = state.pop_ready_window(finalizing=True)
            prepared = self.runner.prepare_streaming_chunk(
                window.waveform,
                language=state.language,
                is_first=window.is_first,
            )
            result = self.runner.run_streaming_batch(
                [state.decode],
                [prepared],
                requested_languages=[state.language],
                max_new_tokens=[state.max_new_tokens],
            )
            self._record_batch([(request_id, state, window)], result)
            partial = self._partial_message(state, result, 0)
            if partial is not None:
                messages.append(partial)

        state.metrics.finalized_s = time.perf_counter()
        self._aggregate["completed_streams"] += 1
        metrics = self._metrics_snapshot(state, now=state.metrics.finalized_s)
        payload = state.payload
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data={
                        "text": state.clean_text,
                        "raw_text": state.raw_text,
                        "language": state.detected_language,
                        "duration_s": state.total_samples / state.spec.sample_rate,
                        "token_ids": list(state.decode.tokens),
                        "durations": list(state.decode.durations),
                        "encoder_frames": state.decode.encoder_frames,
                        "decoder_steps": state.decode.decoder_steps,
                        "streaming_latency_ms": state.spec.streaming_latency_ms,
                        "metrics": metrics,
                        "usage": {"engine_time_s": state.metrics.model_compute_s},
                        "modality": "text",
                    },
                ),
            )
        )
        return messages

    def _metrics_snapshot(
        self, state: Nemotron3_5ASRStreamState, *, now: float
    ) -> dict[str, Any]:
        audio_s = state.total_samples / state.spec.sample_rate
        compute_s = state.metrics.model_compute_s
        elapsed_s = max(now - state.metrics.request_started_s, 0.0)
        ttft_s = (
            state.metrics.first_text_s - state.metrics.request_started_s
            if state.metrics.first_text_s is not None
            else None
        )
        speech_end_to_final_s = (
            state.metrics.finalized_s - state.metrics.input_done_s
            if state.metrics.finalized_s is not None
            and state.metrics.input_done_s is not None
            else None
        )
        latencies = state.metrics.chunk_latency_ms
        return {
            "ttft_s": ttft_s,
            "speech_end_to_final_s": speech_end_to_final_s,
            "elapsed_s": elapsed_s,
            "model_compute_s": compute_s,
            "rtf": compute_s / audio_s if audio_s > 0 else None,
            "rtfx": audio_s / compute_s if compute_s > 0 else None,
            "throughput_audio_s_per_s": audio_s / elapsed_s if elapsed_s > 0 else None,
            "chunk_latency_ms": list(latencies),
            "chunk_latency_p50_ms": self._percentile(latencies, 50),
            "chunk_latency_p99_ms": self._percentile(latencies, 99),
            "chunk_ready_wait_ms": list(state.metrics.chunk_ready_wait_ms),
            "input_packets": state.metrics.packet_count,
            "model_chunks": state.metrics.model_chunk_count,
            "cache_reuses": state.metrics.cache_reuse_count,
            "batch_sizes": list(state.metrics.batch_sizes),
            "max_queue_depth": state.metrics.max_queue_depth,
        }

    @staticmethod
    def _percentile(values: Sequence[float], percentile: int) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = max(math.ceil(percentile / 100 * len(ordered)) - 1, 0)
        return float(ordered[index])

    def _require_stream_state(self, request_id: str) -> Nemotron3_5ASRStreamState:
        state = self._stream_states.get(request_id)
        if state is None:
            raise ValueError(f"No active Nemotron stream for request {request_id!r}")
        return state

    def clear_stream_state(self, request_id: str) -> None:
        self._stream_states.pop(request_id, None)

    def stats(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                **self._aggregate,
                "active_streams": len(self._stream_states),
                "inbox_depth": self.inbox.qsize(),
            }

    def start(self) -> None:
        try:
            super().start()
        finally:
            with self._state_lock:
                self._stream_states.clear()
            self._close_runner()

    def stop(self) -> None:
        was_running = self._running
        super().stop()
        if not was_running:
            with self._state_lock:
                self._stream_states.clear()
            self._close_runner()

    def _close_runner(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.runner.close()


__all__ = [
    "Nemotron3_5ASRAudioWindow",
    "Nemotron3_5ASRStreamMetrics",
    "Nemotron3_5ASRStreamState",
    "Nemotron3_5ASRStreamingChunkSpec",
    "Nemotron3_5ASRStreamingScheduler",
]
