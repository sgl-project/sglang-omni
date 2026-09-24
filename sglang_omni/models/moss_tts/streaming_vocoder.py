# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for MOSS-TTS Delay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.moss_tts.payload_types import (
    load_moss_tts_state,
    resolve_moss_audio_pad_code,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import (
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)


@dataclass
class MossSegmentState:
    frames: list[torch.Tensor] = field(default_factory=list)
    emitted_frames: int = 0
    closed: bool = False
    tail_emitted: bool = False


@dataclass
class MossStreamState:
    delay_window: deque[torch.Tensor] = field(default_factory=deque)
    pending_raw_frames: deque[torch.Tensor] = field(default_factory=deque)
    segments: list[MossSegmentState] = field(default_factory=list)
    active_segment: int | None = None
    delayed_count: int = 0
    next_decode_rows: int = 0
    n_vq: int | None = None
    audio_pad_code: int | None = None
    sample_rate: int = 24000
    initial_codec_chunk_frames: int = 0
    samples_per_frame: int | None = None


class MossStreamingVocoderScheduler(StreamingVocoderBase[MossStreamState, None]):
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

        self.vocoder = vocoder
        self.audio_vocoder = vocoder.audio_vocoder
        self.stream_stride = int(stream_stride)
        self.stream_followup_stride = int(stream_followup_stride)
        self.stream_overlap_tokens = int(stream_overlap_tokens)
        self.stream_holdback_tokens = int(stream_holdback_tokens)
        self.default_initial_chunk_frames = max(0, int(initial_chunk_frames))
        self.default_n_vq = int(
            getattr(getattr(vocoder.processor, "model_config", None), "n_vq", 0)
            or 0  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        self.default_audio_pad_code = resolve_moss_audio_pad_code(
            getattr(
                vocoder.processor, "model_config", None
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        sample_rate = int(self.audio_vocoder.sample_rate)
        self.default_samples_per_frame = self.resolve_samples_per_frame(
            self.audio_vocoder, sample_rate
        )

        super().__init__(
            self.vocode_payload,
            batch_compute_fn=self.vocode_payloads,
            sample_rate=sample_rate,
            stream_source_hint="MOSS-TTS",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def create_stream_state(self, request_id: str) -> MossStreamState:
        del request_id
        return MossStreamState(
            n_vq=self.default_n_vq or None,
            audio_pad_code=self.default_audio_pad_code,
            sample_rate=self.sample_rate,
            samples_per_frame=self.default_samples_per_frame,
        )

    def latch_stream_contract(
        self,
        request_id: str,
        state: MossStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            payload = source
            final_state = load_moss_tts_state(payload)
            delayed = final_state.delayed_audio_codes
            n_vq = state.n_vq
            if delayed is not None:
                delayed_tensor = torch.as_tensor(delayed)
                if delayed_tensor.ndim == 2 and int(delayed_tensor.shape[1]) > 0:
                    n_vq = int(delayed_tensor.shape[1])
            self.latch_contract_values(
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
            self.latch_initial_chunk_frames(state, params)
            return

        metadata: Mapping[str, Any] = source
        self.latch_contract_values(
            request_id,
            state,
            n_vq=metadata.get("n_vq", state.n_vq),
            audio_pad_code=metadata.get("audio_pad_code", state.audio_pad_code),
            sample_rate=metadata.get("sample_rate", state.sample_rate),
            source=origin,
        )
        self.latch_initial_chunk_frames(state, metadata)

    def validate_chunk(
        self,
        request_id: str,
        state: MossStreamState,
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
        n_vq, _ = self.require_contract(state, request_id)
        if int(rows.shape[1]) != n_vq:
            raise ValueError(
                f"MOSS-TTS stream chunk has {int(rows.shape[1])} codebooks, "
                f"expected {n_vq}"
            )
        return rows

    def ingest(
        self,
        request_id: str,
        state: MossStreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        n_vq, _ = self.require_contract(state, "<stream>")
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
        state: MossStreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        n_vq, audio_pad_code = self.require_contract(state, request_id)
        next_decode_rows = state.next_decode_rows or self.first_decode_rows(state, n_vq)
        if not is_final and state.delayed_count < next_decode_rows:
            state.next_decode_rows = next_decode_rows
            return None

        pending_count = len(state.pending_raw_frames)
        process_count = (
            pending_count
            if is_final
            else max(0, pending_count - self.stream_holdback_tokens)
        )
        for _ in range(process_count):
            self.ingest_raw_frame(
                state,
                state.pending_raw_frames.popleft(),
                audio_pad_code=audio_pad_code,
            )
        if is_final:
            self.close_active_segment(state)

        chunks: list[torch.Tensor] = []
        for segment in state.segments:
            chunk = self.decode_segment_delta(
                state,
                segment,
                flush_tail=bool(segment.closed or is_final),
            )
            if chunk is not None:
                chunks.append(chunk)

        if chunks:
            state.next_decode_rows = state.delayed_count + self.stream_followup_stride
            return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

        if not is_final:
            if len(state.pending_raw_frames) <= self.stream_holdback_tokens:
                state.next_decode_rows = max(
                    state.delayed_count + 1,
                    n_vq + self.stream_holdback_tokens,
                )
            else:
                state.next_decode_rows = (
                    state.delayed_count + self.stream_followup_stride
                )
        return None

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: MossStreamState,
    ) -> torch.Tensor | None:
        del request_id, state
        final_state, delayed_codes = self.vocoder.prepare_item(payload)
        waveform, _ = self.vocoder.decode_audio(final_state, delayed_codes)
        return waveform

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: MossStreamState,
    ) -> dict[str, Any]:
        del request_id
        final_state = load_moss_tts_state(payload)
        final_state.delayed_audio_codes = None
        final_state.sample_rate = int(state.sample_rate or self.sample_rate)
        data = final_state.to_dict()
        data["modality"] = "audio"
        data["sample_rate"] = final_state.sample_rate
        usage = build_usage(final_state)
        if usage is not None:
            data["usage"] = usage
        return data

    async def vocode_payload(self, payload: StagePayload) -> StagePayload:
        return (await self.vocode_payloads([payload]))[0]

    async def vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        items = [self.vocoder.prepare_item(payload) for payload in payloads]
        results = await self.vocoder.decode_batch(items)
        if len(results) != len(items):
            raise RuntimeError(
                f"MOSS-TTS vocoder returned {len(results)} results for "
                f"{len(items)} requests"
            )
        return [
            self.vocoder.store_result(payload, item[0], waveform, sample_rate)
            for payload, item, (waveform, sample_rate) in zip(payloads, items, results)
        ]

    def decode_segment_delta(
        self,
        state: MossStreamState,
        segment: MossSegmentState,
        *,
        flush_tail: bool,
    ) -> torch.Tensor | None:
        total_frames = len(segment.frames)
        emitted_frames = int(segment.emitted_frames)
        if total_frames < emitted_frames:
            raise RuntimeError("MOSS-TTS streaming segment cursor moved backwards")
        if total_frames == emitted_frames and (not flush_tail or segment.tail_emitted):
            return None

        window_start = max(0, emitted_frames - self.stream_overlap_tokens)
        window = torch.stack(segment.frames[window_start:], dim=0)
        decoded = self.audio_vocoder.decode_codes([window])
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
    def ingest_raw_frame(
        state: MossStreamState,
        frame: torch.Tensor,
        *,
        audio_pad_code: int,
    ) -> None:
        is_pad = bool(torch.all(frame == int(audio_pad_code)))
        is_complete = bool(torch.all((frame >= 0) & (frame < int(audio_pad_code))))
        if is_pad or not is_complete:
            MossStreamingVocoderScheduler.close_active_segment(state)
            return
        if state.active_segment is None:
            state.segments.append(MossSegmentState())
            state.active_segment = len(state.segments) - 1
        state.segments[state.active_segment].frames.append(frame)

    @staticmethod
    def close_active_segment(state: MossStreamState) -> None:
        if state.active_segment is None:
            return
        state.segments[state.active_segment].closed = True
        state.active_segment = None

    def first_decode_rows(self, state: MossStreamState, n_vq: int) -> int:
        initial_frames = int(state.initial_codec_chunk_frames)
        if initial_frames > 0:
            return n_vq - 1 + initial_frames + self.stream_holdback_tokens
        return max(n_vq, self.stream_stride)

    def latch_initial_chunk_frames(
        self,
        state: MossStreamState,
        values: Mapping[str, Any] | None,
    ) -> None:
        self.require_contract(state, "<stream>")
        steady_frames = self.stream_followup_stride
        state.initial_codec_chunk_frames = resolve_initial_codec_chunk_frames(
            values,
            steady_chunk_frames=steady_frames,
            default_frames=self.default_initial_chunk_frames,
        )

    @staticmethod
    def latch_contract_values(
        request_id: str,
        state: MossStreamState,
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
    def require_contract(
        state: MossStreamState,
        request_id: str,
    ) -> tuple[int, int]:
        if state.n_vq is None or state.audio_pad_code is None:
            raise RuntimeError(
                f"MOSS-TTS stream contract for {request_id!r} is incomplete"
            )
        return int(state.n_vq), int(state.audio_pad_code)

    @staticmethod
    def resolve_samples_per_frame(
        audio_vocoder: Any,
        sample_rate: int,
    ) -> int | None:
        config = getattr(getattr(audio_vocoder, "model", None), "config", None)
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
