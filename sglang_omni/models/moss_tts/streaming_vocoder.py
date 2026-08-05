# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for MOSS-TTS Delay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.moss_tts.payload_types import (
    MossTTSState,
    resolve_moss_audio_pad_code,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import (
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)


@dataclass
class _MossSegmentState:
    frames: list[torch.Tensor] = field(default_factory=list)
    emitted_frames: int = 0
    closed: bool = False
    tail_emitted: bool = False


@dataclass
class _MossStreamState:
    delay_window: deque[torch.Tensor] = field(default_factory=deque)
    pending_raw_frames: deque[torch.Tensor] = field(default_factory=deque)
    segments: list[_MossSegmentState] = field(default_factory=list)
    active_segment: int | None = None
    delayed_count: int = 0
    next_decode_rows: int = 0
    n_vq: int | None = None
    audio_pad_code: int | None = None
    sample_rate: int = 24000
    initial_codec_chunk_frames: int = 0
    samples_per_frame: int | None = None


class MossStreamingVocoderScheduler(StreamingVocoderBase[_MossStreamState, None]):
    """Incrementally reverse MOSS delay rows and decode overlap windows."""

    def __init__(
        self,
        vocoder: Any,
        *,
        stream_stride: int = 8,
        stream_followup_stride: int = 8,
        stream_overlap_tokens: int = 8,
        stream_holdback_tokens: int = 1,
        initial_chunk_frames: int = 0,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream strides must be > 0")
        if stream_overlap_tokens <= 0:
            raise ValueError("stream overlap must be > 0")
        if stream_holdback_tokens < 0:
            raise ValueError("stream holdback must be >= 0")

        self._vocoder = vocoder
        self._audio_tokenizer = vocoder._audio_tokenizer
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_overlap_tokens = int(stream_overlap_tokens)
        self._stream_holdback_tokens = int(stream_holdback_tokens)
        self._default_initial_chunk_frames = max(0, int(initial_chunk_frames))
        self._default_n_vq = int(
            getattr(getattr(vocoder._processor, "model_config", None), "n_vq", 0) or 0
        )
        self._default_audio_pad_code = resolve_moss_audio_pad_code(
            getattr(vocoder._processor, "model_config", None)
        )
        sample_rate = int(self._audio_tokenizer.sample_rate)
        self._default_samples_per_frame = self._resolve_samples_per_frame(
            self._audio_tokenizer, sample_rate
        )

        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            sample_rate=sample_rate,
            stream_source_hint="MOSS-TTS",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def create_stream_state(self, request_id: str) -> _MossStreamState:
        del request_id
        return _MossStreamState(
            n_vq=self._default_n_vq or None,
            audio_pad_code=self._default_audio_pad_code,
            sample_rate=self._sample_rate,
            samples_per_frame=self._default_samples_per_frame,
        )

    def latch_stream_contract(
        self,
        request_id: str,
        state: _MossStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            payload = source
            final_state = MossTTSState.from_dict(payload.data)
            delayed = final_state.delayed_audio_codes
            n_vq = state.n_vq
            if delayed is not None:
                delayed_tensor = torch.as_tensor(delayed)
                if delayed_tensor.ndim == 2 and int(delayed_tensor.shape[1]) > 0:
                    n_vq = int(delayed_tensor.shape[1])
            self._latch_contract_values(
                request_id,
                state,
                n_vq=n_vq,
                audio_pad_code=state.audio_pad_code,
                sample_rate=final_state.sample_rate or state.sample_rate,
                source=origin,
            )
            params = (
                payload.request.params
                if isinstance(payload.request.params, dict)
                else None
            )
            self._latch_initial_chunk_frames(state, params)
            return

        metadata: Mapping[str, Any] = source
        self._latch_contract_values(
            request_id,
            state,
            n_vq=metadata.get("n_vq", state.n_vq),
            audio_pad_code=metadata.get("audio_pad_code", state.audio_pad_code),
            sample_rate=metadata.get("sample_rate", state.sample_rate),
            source=origin,
        )
        self._latch_initial_chunk_frames(state, metadata)

    def validate_chunk(
        self,
        request_id: str,
        state: _MossStreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        rows = codes.to(dtype=torch.long)
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        elif rows.ndim != 2:
            raise ValueError(
                f"MOSS-TTS stream chunk for {request_id!r} must be [N] or "
                f"[T, N], got {tuple(rows.shape)}"
            )
        n_vq, _ = self._require_contract(state, request_id)
        if int(rows.shape[1]) != n_vq:
            raise ValueError(
                f"MOSS-TTS stream chunk has {int(rows.shape[1])} codebooks, "
                f"expected {n_vq}"
            )
        return rows

    def ingest(
        self,
        request_id: str,
        state: _MossStreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        n_vq, _ = self._require_contract(state, "<stream>")
        for row in codes.detach().to(device="cpu", dtype=torch.long).unbind(0):
            state.delay_window.append(row)
            state.delayed_count += 1
            if len(state.delay_window) < n_vq:
                continue
            raw_frame = torch.stack(
                [state.delay_window[channel][channel] for channel in range(n_vq)]
            )
            state.pending_raw_frames.append(raw_frame)
            state.delay_window.popleft()

    def decode_delta(
        self,
        request_id: str,
        state: _MossStreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        n_vq, audio_pad_code = self._require_contract(state, request_id)
        next_decode_rows = state.next_decode_rows or self._first_decode_rows(
            state, n_vq
        )
        if not is_final and state.delayed_count < next_decode_rows:
            state.next_decode_rows = next_decode_rows
            return None

        pending_count = len(state.pending_raw_frames)
        process_count = (
            pending_count
            if is_final
            else max(0, pending_count - self._stream_holdback_tokens)
        )
        for _ in range(process_count):
            self._ingest_raw_frame(
                state,
                state.pending_raw_frames.popleft(),
                audio_pad_code=audio_pad_code,
            )
        if is_final:
            self._close_active_segment(state)

        chunks: list[torch.Tensor] = []
        for segment in state.segments:
            chunk = self._decode_segment_delta(
                state,
                segment,
                flush_tail=bool(segment.closed or is_final),
            )
            if chunk is not None:
                chunks.append(chunk)

        if chunks:
            state.next_decode_rows = state.delayed_count + self._stream_followup_stride
            return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

        if not is_final:
            if len(state.pending_raw_frames) <= self._stream_holdback_tokens:
                state.next_decode_rows = max(
                    state.delayed_count + 1,
                    n_vq + self._stream_holdback_tokens,
                )
            else:
                state.next_decode_rows = (
                    state.delayed_count + self._stream_followup_stride
                )
        return None

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: _MossStreamState,
    ) -> torch.Tensor | None:
        del request_id, state
        final_state, delayed_codes = self._vocoder.prepare_item(payload)
        waveform, _ = self._vocoder._decode_audio(final_state, delayed_codes)
        return waveform

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _MossStreamState,
    ) -> dict[str, Any]:
        del request_id
        final_state = MossTTSState.from_dict(payload.data)
        final_state.delayed_audio_codes = None
        final_state.sample_rate = int(state.sample_rate or self._sample_rate)
        data = final_state.to_dict()
        data["modality"] = "audio"
        data["sample_rate"] = final_state.sample_rate
        usage = build_usage(final_state)
        if usage is not None:
            data["usage"] = usage
        return data

    async def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        return (await self._vocode_payloads([payload]))[0]

    async def _vocode_payloads(
        self, payloads: list[StagePayload]
    ) -> list[StagePayload]:
        items = [self._vocoder.prepare_item(payload) for payload in payloads]
        results = await self._vocoder.decode_batch(items)
        if len(results) != len(items):
            raise RuntimeError(
                f"MOSS-TTS vocoder returned {len(results)} results for "
                f"{len(items)} requests"
            )
        return [
            self._vocoder.store_result(payload, item[0], waveform, sample_rate)
            for payload, item, (waveform, sample_rate) in zip(payloads, items, results)
        ]

    def _decode_segment_delta(
        self,
        state: _MossStreamState,
        segment: _MossSegmentState,
        *,
        flush_tail: bool,
    ) -> torch.Tensor | None:
        total_frames = len(segment.frames)
        emitted_frames = int(segment.emitted_frames)
        if total_frames < emitted_frames:
            raise RuntimeError("MOSS-TTS streaming segment cursor moved backwards")
        if total_frames == emitted_frames and (not flush_tail or segment.tail_emitted):
            return None

        window_start = max(0, emitted_frames - self._stream_overlap_tokens)
        window = torch.stack(segment.frames[window_start:], dim=0)
        decoded = self._audio_tokenizer.decode_codes([window])
        if not decoded:
            return None
        audio = torch.as_tensor(decoded[0]).detach().reshape(-1).to(torch.float32)
        decoded_frames = total_frames - window_start
        samples_per_frame = state.samples_per_frame or max(
            int(audio.numel()) // max(decoded_frames, 1), 1
        )
        state.samples_per_frame = int(samples_per_frame)
        trim_frames = emitted_frames - window_start
        trim_samples = min(trim_frames * samples_per_frame, int(audio.numel()))
        if flush_tail:
            delta = audio[trim_samples:].contiguous()
            segment.tail_emitted = True
        else:
            new_frames = total_frames - emitted_frames
            emit_samples = new_frames * samples_per_frame
            delta = audio[trim_samples : trim_samples + emit_samples].contiguous()
        segment.emitted_frames = total_frames
        return delta if delta.numel() else None

    @staticmethod
    def _ingest_raw_frame(
        state: _MossStreamState,
        frame: torch.Tensor,
        *,
        audio_pad_code: int,
    ) -> None:
        is_pad = bool(torch.all(frame == int(audio_pad_code)))
        is_complete = bool(torch.all((frame >= 0) & (frame < int(audio_pad_code))))
        if is_pad or not is_complete:
            MossStreamingVocoderScheduler._close_active_segment(state)
            return
        if state.active_segment is None:
            state.segments.append(_MossSegmentState())
            state.active_segment = len(state.segments) - 1
        state.segments[state.active_segment].frames.append(frame)

    @staticmethod
    def _close_active_segment(state: _MossStreamState) -> None:
        if state.active_segment is None:
            return
        state.segments[state.active_segment].closed = True
        state.active_segment = None

    def _first_decode_rows(self, state: _MossStreamState, n_vq: int) -> int:
        initial_frames = int(state.initial_codec_chunk_frames)
        if initial_frames > 0:
            return n_vq - 1 + initial_frames + self._stream_holdback_tokens
        return max(n_vq, self._stream_stride)

    def _latch_initial_chunk_frames(
        self,
        state: _MossStreamState,
        values: Mapping[str, Any] | None,
    ) -> None:
        self._require_contract(state, "<stream>")
        steady_frames = self._stream_followup_stride
        state.initial_codec_chunk_frames = resolve_initial_codec_chunk_frames(
            values,
            steady_chunk_frames=steady_frames,
            default_frames=self._default_initial_chunk_frames,
        )

    @staticmethod
    def _latch_contract_values(
        request_id: str,
        state: _MossStreamState,
        *,
        n_vq: Any,
        audio_pad_code: Any,
        sample_rate: Any,
        source: str,
    ) -> None:
        try:
            n_vq_i = int(n_vq)
            audio_pad_code_i = int(audio_pad_code)
            sample_rate_i = int(sample_rate)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"MOSS-TTS {source} for {request_id!r} has invalid codec metadata"
            ) from exc
        if n_vq_i <= 0 or audio_pad_code_i <= 0 or sample_rate_i <= 0:
            raise ValueError(
                f"MOSS-TTS {source} for {request_id!r} has invalid codec metadata"
            )
        for name, value in (
            ("n_vq", n_vq_i),
            ("audio_pad_code", audio_pad_code_i),
            ("sample_rate", sample_rate_i),
        ):
            previous = getattr(state, name)
            if previous is not None and int(previous) != value:
                raise ValueError(
                    f"MOSS-TTS stream {name} changed for {request_id!r}: "
                    f"{previous} -> {value}"
                )
            setattr(state, name, value)

    @staticmethod
    def _require_contract(
        state: _MossStreamState,
        request_id: str,
    ) -> tuple[int, int]:
        if state.n_vq is None or state.audio_pad_code is None:
            raise RuntimeError(
                f"MOSS-TTS stream contract for {request_id!r} is incomplete"
            )
        return int(state.n_vq), int(state.audio_pad_code)

    @staticmethod
    def _resolve_samples_per_frame(
        audio_tokenizer: Any,
        sample_rate: int,
    ) -> int | None:
        config = getattr(getattr(audio_tokenizer, "model", None), "config", None)
        for attr in (
            "downsample_rate",
            "samples_per_frame",
            "frame_length",
            "hop_length",
        ):
            value = getattr(config, attr, None)
            if value and int(value) > 0:
                return int(value)
        frame_rate = getattr(config, "frame_rate", None)
        if frame_rate and float(frame_rate) > 0:
            return max(int(round(sample_rate / float(frame_rate))), 1)
        return None


__all__ = ["MossStreamingVocoderScheduler"]
