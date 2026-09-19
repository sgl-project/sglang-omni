# SPDX-License-Identifier: Apache-2.0
"""Native cache-aware PCM streaming scheduler for Nemotron 3.5 ASR."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

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
    build_nemotron3_5_asr_result,
    normalize_nemotron_language,
    validate_nemotron_greedy_params,
)

PCM16_BYTES_PER_SAMPLE = 2
PCM16_AMPLITUDE_SCALE = 32768.0


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


@dataclass(frozen=True, slots=True)
class Nemotron3_5ASRAudioWindow:
    waveform: np.ndarray
    model_chunk_index: int
    is_first: bool


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
    is_input_done: bool = False
    raw_text: str = ""
    clean_text: str = ""
    detected_language: str | None = None
    request_started_s: float = field(default_factory=time.perf_counter)
    model_compute_s: float = 0.0

    def append_pcm16(
        self, tensor: torch.Tensor, metadata: Mapping[str, object]
    ) -> None:
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
        if isinstance(sample_rate, bool) or sample_rate != self.spec.sample_rate:
            raise ValueError(
                f"Nemotron streaming requires sample_rate={self.spec.sample_rate}"
            )
        modality = metadata.get("modality")
        if modality not in {None, "audio", "pcm16"}:
            raise ValueError(
                f"Nemotron streaming chunk modality must be audio or pcm16, got {modality!r}"
            )
        if self.is_input_done:
            raise RuntimeError(f"Nemotron stream {self.request_id!r} is already done")

        pcm_samples = tensor.detach().contiguous().reshape(-1)
        if pcm_samples.numel() == 0:
            raise ValueError("Nemotron streaming PCM16 chunks must not be empty")
        if pcm_samples.dtype == torch.int16:
            packet_bytes = pcm_samples.numpy().astype("<i2", copy=False).tobytes()
        else:
            packet_bytes = pcm_samples.numpy().tobytes()
        self.pcm_bytes.extend(packet_bytes)
        self.total_samples = len(self.pcm_bytes) // PCM16_BYTES_PER_SAMPLE

    @property
    def has_reached_decode_limit(self) -> bool:
        return (
            self.max_new_tokens is not None
            and self.decode.decoder_steps >= self.max_new_tokens
        )

    def mark_done(self) -> None:
        if self.is_input_done:
            raise RuntimeError(f"Nemotron stream {self.request_id!r} is already done")
        if self.total_samples == 0:
            raise ValueError("Nemotron streaming input contains no PCM16 samples")
        if len(self.pcm_bytes) % PCM16_BYTES_PER_SAMPLE:
            raise ValueError(
                "Nemotron streaming input ends with an incomplete PCM16 sample"
            )
        self.is_input_done = True

    def next_window_bounds(self) -> tuple[int, int]:
        if self.model_chunk_index == 0:
            return 0, self.spec.first_samples
        start = self.next_mel_frame * self.spec.hop_length - self.spec.n_fft // 2
        return start, start + self.spec.subsequent_samples

    def has_ready_window(self, *, finalizing: bool = False) -> bool:
        if self.model_chunk_index == 0:
            return self.total_samples >= self.spec.first_samples or (
                finalizing and self.total_samples > 0
            )
        _, end = self.next_window_bounds()
        if self.total_samples >= end:
            return True
        return finalizing and self.total_samples > self.covered_audio_end

    def pop_ready_window(
        self, *, finalizing: bool = False
    ) -> Nemotron3_5ASRAudioWindow:
        assert self.has_ready_window(
            finalizing=finalizing
        ), f"Nemotron stream {self.request_id!r} has no ready window"
        start, end = self.next_window_bounds()
        window_samples = end - start
        source_start = max(start, 0)
        source_end = min(end, self.total_samples)
        complete_bytes = memoryview(self.pcm_bytes)[
            : self.total_samples * PCM16_BYTES_PER_SAMPLE
        ]
        pcm_samples = np.frombuffer(complete_bytes, dtype="<i2")
        window_pcm = pcm_samples[source_start : max(source_start, source_end)]
        left_padding = max(-start, 0)
        right_padding = window_samples - left_padding - int(window_pcm.shape[0])
        waveform = np.pad(
            window_pcm.astype(np.float32) / PCM16_AMPLITUDE_SCALE,
            (left_padding, right_padding),
        ).astype(np.float32, copy=False)
        is_first = self.model_chunk_index == 0
        window = Nemotron3_5ASRAudioWindow(
            waveform=waveform,
            model_chunk_index=self.model_chunk_index,
            is_first=is_first,
        )

        self.covered_audio_end = max(
            self.covered_audio_end, min(end, self.total_samples)
        )
        self.model_chunk_index += 1
        if is_first:
            self.next_mel_frame = self.spec.first_frames
        else:
            self.next_mel_frame += self.spec.subsequent_frames
        return window


class Nemotron3_5ASRStreamingScheduler(StreamingSimpleScheduler):
    """Serialize request-owned RNNT state with abort cleanup through state_lock."""

    supports_external_input_stream = True
    can_batch_stream_chunks = True
    stream_chunk_batch_distinct_requests = True

    def __init__(
        self,
        runner: Nemotron3_5ASRModelRunner,
        compute_fn: Callable[[StagePayload], StagePayload],
        *,
        batch_compute_fn: Callable[
            [Sequence[StagePayload]], list[StagePayload | BaseException]
        ],
        prompt_dictionary: Mapping[str, int],
        max_batch_size: int,
        max_batch_wait_ms: float,
        max_pending_messages: int,
    ) -> None:
        self.runner = runner
        self.chunk_spec = Nemotron3_5ASRStreamingChunkSpec(
            **runner.streaming_chunk_spec
        )
        self.prompt_dictionary = dict(prompt_dictionary)
        self.stream_states: OrderedDict[str, Nemotron3_5ASRStreamState] = OrderedDict()
        self.is_closed = False
        self.completed_streams = 0
        self.aborted_streams = 0
        self.stream_chunk_batch_max = max_batch_size
        super().__init__(
            compute_fn,
            batch_compute_fn=batch_compute_fn,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            max_pending_messages=max_pending_messages,
        )

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return payload.external_input_stream

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        if request_id in self.stream_states:
            raise ValueError(f"Nemotron stream {request_id!r} already exists")
        params = payload.request.params or {}
        max_new_tokens = validate_nemotron_greedy_params(params)
        language = normalize_nemotron_language(
            params.get("language"), self.prompt_dictionary
        )
        self.stream_states[request_id] = Nemotron3_5ASRStreamState(
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
        with self.state_lock:
            for request_id, item in items:
                if self.is_aborted(request_id):
                    continue
                try:
                    state = self.stream_states[request_id]
                    metadata = item.metadata or {}
                    if not isinstance(metadata, dict):
                        raise TypeError(
                            "Nemotron streaming chunk metadata must be a dict"
                        )
                    if not isinstance(item.data, torch.Tensor):
                        raise TypeError(
                            "Nemotron streaming chunks must carry torch.Tensor"
                        )
                    state.append_pcm16(item.data, metadata)
                except Exception as exc:
                    self.emit_error(request_id, exc)
                    self.abort_state(request_id)
                    self.aborted_streams += 1
                    failed.append(request_id)
        for request_id in dict.fromkeys(failed):
            self.cleanup_aborted_request(request_id)

    def has_ready_work(self) -> bool:
        with self.state_lock:
            return any(
                not self.is_aborted(request_id)
                and (
                    state.is_input_done
                    or (not state.has_reached_decode_limit and state.has_ready_window())
                )
                for request_id, state in self.stream_states.items()
            )

    def run_ready_step(self) -> None:
        failed: list[str] = []
        with self.state_lock:
            ready: list[
                tuple[str, Nemotron3_5ASRStreamState, Nemotron3_5ASRAudioWindow]
            ] = []
            for request_id, state in list(self.stream_states.items()):
                if self.is_aborted(request_id):
                    continue
                try:
                    if state.has_reached_decode_limit or not state.has_ready_window(
                        finalizing=state.is_input_done
                    ):
                        if state.is_input_done:
                            self.finish_stream(state)
                        continue
                    if (
                        ready
                        and state.model_chunk_index != ready[0][2].model_chunk_index
                    ):
                        continue
                    ready.append(
                        (
                            request_id,
                            state,
                            state.pop_ready_window(finalizing=state.is_input_done),
                        )
                    )
                    # Rotate selected requests behind those still waiting for a step.
                    self.stream_states.move_to_end(request_id)
                    if len(ready) == self.max_batch_size:
                        break
                except Exception as exc:
                    self.emit_error(request_id, exc)
                    self.abort_state(request_id)
                    self.aborted_streams += 1
                    failed.append(request_id)
            failed.extend(self.run_ready_windows(ready))
        for request_id in dict.fromkeys(failed):
            self.cleanup_aborted_request(request_id)

    def run_ready_windows(
        self,
        ready: Sequence[
            tuple[str, Nemotron3_5ASRStreamState, Nemotron3_5ASRAudioWindow]
        ],
    ) -> list[str]:
        if not ready:
            return []
        failed: list[str] = []
        try:
            prepared_chunks = [
                self.runner.prepare_streaming_chunk(
                    window.waveform,
                    language=state.language,
                    is_first=window.is_first,
                )
                for _, state, window in ready
            ]
            batch_result = self.runner.run_streaming_batch(
                [state.decode for _, state, _ in ready],
                prepared_chunks,
                requested_languages=[state.language for _, state, _ in ready],
                max_new_tokens=[state.max_new_tokens for _, state, _ in ready],
            )
            for index, (request_id, state, _) in enumerate(ready):
                state.model_compute_s += batch_result.elapsed_s / len(ready)
                message = self.partial_message(state, batch_result, index)
                if message is None or self.is_aborted(request_id):
                    continue
                self.outbox.put(message)
        except Exception as exc:
            for request_id, _, _ in ready:
                self.emit_error(request_id, exc)
                self.abort_state(request_id)
                self.aborted_streams += 1
                failed.append(request_id)
        return failed

    def partial_message(
        self,
        state: Nemotron3_5ASRStreamState,
        batch_result: Nemotron3_5ASRStreamingBatchResult,
        index: int,
    ) -> OutgoingMessage | None:
        previous_text = state.clean_text
        state.raw_text = batch_result.raw_texts[index]
        state.clean_text = batch_result.clean_texts[index]
        state.detected_language = batch_result.languages[index]
        if not state.clean_text or state.clean_text == previous_text:
            return None
        if previous_text and not state.clean_text.startswith(previous_text):
            raise RuntimeError(
                "Nemotron streaming transcript changed a previously emitted prefix"
            )
        text_delta = state.clean_text[len(previous_text) :]
        return OutgoingMessage(
            request_id=state.request_id,
            type="stream",
            data={
                "text": text_delta,
                "full_text": state.clean_text,
                "raw_text": state.raw_text,
                "language": state.detected_language,
                "token_ids": list(state.decode.tokens),
                "modality": "text",
            },
            metadata={"modality": "text"},
        )

    def on_stream_done(self, request_id: str) -> None:
        self.stream_states[request_id].mark_done()

    def finish_stream(self, state: Nemotron3_5ASRStreamState) -> None:
        finalized_s = time.perf_counter()
        final_payload = build_nemotron3_5_asr_result(
            state.payload,
            raw_text=state.raw_text,
            requested_language=state.language,
            duration_s=state.total_samples / state.spec.sample_rate,
            asr_latency_s=max(finalized_s - state.request_started_s, 0.0),
            model_latency_s=state.model_compute_s,
            extra_data={
                "token_ids": list(state.decode.tokens),
                "durations": list(state.decode.durations),
                "encoder_frames": state.decode.encoder_frames,
                "decoder_steps": state.decode.decoder_steps,
                "streaming_latency_ms": state.spec.streaming_latency_ms,
            },
        )
        self.complete_stream_request(
            state.request_id,
            [
                OutgoingMessage(
                    request_id=state.request_id,
                    type="result",
                    data=final_payload,
                )
            ],
        )
        self.completed_streams += 1

    def clear_stream_state(self, request_id: str) -> None:
        self.stream_states.pop(request_id, None)

    def stats(self) -> dict[str, int]:
        with self.state_lock:
            return {
                "completed_streams": self.completed_streams,
                "aborted_streams": self.aborted_streams,
                "active_streams": len(self.stream_states),
                "inbox_depth": self.inbox.qsize(),
            }

    def start(self) -> None:
        try:
            super().start()
        finally:
            with self.state_lock:
                self.stream_states.clear()
            self.close_runner()

    def stop(self) -> None:
        was_running = self.running
        super().stop()
        if was_running:
            return
        with self.state_lock:
            self.stream_states.clear()
        self.close_runner()

    def close_runner(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        self.runner.close()


__all__ = [
    "Nemotron3_5ASRAudioWindow",
    "Nemotron3_5ASRStreamState",
    "Nemotron3_5ASRStreamingChunkSpec",
    "Nemotron3_5ASRStreamingScheduler",
]
