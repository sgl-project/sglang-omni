# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N801  # Keep the Nemotron 3.5 API spelling.
"""Rolling PCM windows for Nemotron 3.5 ASR."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from sglang_omni.models.nemotron3_5_asr.decoder import Nemotron3_5ASRDecodeState
from sglang_omni.models.nemotron3_5_asr.request_builders import (
    build_nemotron3_5_asr_result,
)
from sglang_omni.proto import StagePayload
from sglang_omni.proto.session import ResourceUsage, TimedChunk

PCM16_BYTES_PER_SAMPLE = 2
PCM16_AMPLITUDE_SCALE = 32768.0


@dataclass(frozen=True, kw_only=True)
class AppendResult:
    text: str
    full_text: str
    is_first_output: bool
    is_final: bool
    final_payload: StagePayload | None


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
    pcm_base_sample: int = 0
    covered_audio_end: int = 0
    model_chunk_index: int = 0
    next_mel_frame: int = 0
    is_input_done: bool = False
    raw_text: str = ""
    clean_text: str = ""
    detected_language: str | None = None
    request_started_s: float = field(default_factory=time.perf_counter)
    model_compute_s: float = 0.0

    def append_bytes(self, packet_bytes: bytes) -> None:
        if self.is_input_done:
            raise RuntimeError(f"Nemotron stream {self.request_id!r} is already done")
        elif self.has_reached_decode_limit:
            self.total_samples += len(packet_bytes) // PCM16_BYTES_PER_SAMPLE
            self.pcm_bytes.clear()
            self.pcm_base_sample = self.total_samples
        else:
            self.pcm_bytes.extend(packet_bytes)
            self.total_samples = self.pcm_base_sample + len(self.pcm_bytes) // PCM16_BYTES_PER_SAMPLE

    def trim_pcm(self) -> None:
        start, _ = self.next_window_bounds()
        keep_from = self.total_samples if self.has_reached_decode_limit else min(max(start, 0), self.total_samples)
        trim_bytes = (keep_from - self.pcm_base_sample) * PCM16_BYTES_PER_SAMPLE
        if trim_bytes > 0:
            self.pcm_bytes = self.pcm_bytes[trim_bytes:]
            self.pcm_base_sample = keep_from
        else:
            pass

    @property
    def has_reached_decode_limit(self) -> bool:
        return (
            self.max_new_tokens is not None
            and self.decode.decoder_steps >= self.max_new_tokens
        )

    def mark_done(self) -> None:
        if self.is_input_done:
            raise RuntimeError(f"Nemotron stream {self.request_id!r} is already done")
        else:
            pass
        if len(self.pcm_bytes) % PCM16_BYTES_PER_SAMPLE:
            raise ValueError(
                "Nemotron streaming input ends with an incomplete PCM16 sample"
            )
        else:
            pass
        self.is_input_done = True

    def next_window_bounds(self) -> tuple[int, int]:
        if self.model_chunk_index == 0:
            return 0, self.spec.first_samples
        else:
            pass
        start = self.next_mel_frame * self.spec.hop_length - self.spec.n_fft // 2
        return start, start + self.spec.subsequent_samples

    def has_ready_window(self) -> bool:
        if self.has_reached_decode_limit:
            return False
        else:
            pass
        _, end = self.next_window_bounds()
        return self.total_samples >= end or (
            self.is_input_done and self.total_samples > self.covered_audio_end
        )

    def pop_ready_window(self) -> Nemotron3_5ASRAudioWindow:
        assert (
            self.has_ready_window()
        ), f"Nemotron stream {self.request_id!r} has no ready window"
        start, end = self.next_window_bounds()
        window_samples = end - start
        source_start = max(start, 0)
        source_end = min(end, self.total_samples)
        complete_bytes = memoryview(self.pcm_bytes)[
            : self.total_samples * PCM16_BYTES_PER_SAMPLE
        ]
        pcm_samples = np.frombuffer(complete_bytes, dtype="<i2")
        window_pcm = pcm_samples[
            source_start - self.pcm_base_sample : max(source_start, source_end) - self.pcm_base_sample
        ]
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
        self.trim_pcm()
        return window


    def append_chunk(self, chunk: TimedChunk, max_pcm_bytes: int) -> None:
        if self.is_input_done:
            raise ValueError("Nemotron audio has already ended")
        elif chunk.modality != "audio" or chunk.format != "pcm16":
            raise ValueError("Nemotron requires audio/pcm16")
        elif not isinstance(chunk.payload, bytes) or len(chunk.payload) % 2:
            raise ValueError("Nemotron requires aligned PCM16 bytes")
        elif (
            not math.isfinite(chunk.t_start_ms) or chunk.t_start_ms < 0
            or not math.isfinite(chunk.duration_ms)
            or abs(chunk.duration_ms - len(chunk.payload) / 32) > 1e-6
        ):
            raise ValueError("Nemotron requires 16 kHz mono PCM16 timing")
        elif not self.has_reached_decode_limit and len(self.pcm_bytes) + len(chunk.payload) > max_pcm_bytes:
            raise RuntimeError("Nemotron accepted operation exceeds PCM budget")
        else:
            self.append_bytes(chunk.payload)
            if chunk.eos:
                self.mark_done()
            else:
                pass

    def usage(self, reservation: int) -> ResourceUsage:
        tensors: list[torch.Tensor | None] = []
        for layer in self.decode.attention_cache.layers:
            if layer.is_initialized:
                tensors.extend([layer.keys, layer.values])
            else:
                pass
        tensors.extend(layer.cache for layer in self.decode.padding_cache.layers.values())
        decoder = self.decode.decoder_cache
        tensors.extend([decoder.cache, decoder.hidden_state, decoder.cell_state])
        cache_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors if tensor is not None)
        history_bytes = (len(self.decode.tokens) + len(self.decode.durations)) * 48
        text_bytes = (len(self.raw_text) + len(self.clean_text)) * 4
        return ResourceUsage(
            kv_tokens=self.decode.attention_cache.get_seq_length(),
            bytes=cache_bytes + history_bytes + text_bytes + len(self.pcm_bytes),
            slots={"pcm_bytes": len(self.pcm_bytes), "cache_bytes": cache_bytes,
                   "history_tokens": len(self.decode.tokens), "reserved_bytes": reservation},
        )

    def append_result(self, payload: StagePayload, previous_text: str, first_output: bool) -> AppendResult:
        if self.is_input_done:
            final = build_nemotron3_5_asr_result(
                payload, raw_text=self.raw_text, requested_language=self.language,
                duration_s=self.total_samples / self.spec.sample_rate,
                asr_latency_s=time.perf_counter() - self.request_started_s,
                model_latency_s=self.model_compute_s,
                extra_data={"token_ids": list(self.decode.tokens),
                            "durations": list(self.decode.durations),
                            "encoder_frames": self.decode.encoder_frames,
                            "decoder_steps": self.decode.decoder_steps,
                            "streaming_latency_ms": self.spec.streaming_latency_ms},
            )
        else:
            final = None
        return AppendResult(
            text=self.clean_text[len(previous_text):], full_text=self.clean_text,
            is_first_output=first_output, is_final=self.is_input_done, final_payload=final,
        )
