# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for FishAudio S2-Pro."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)


@dataclass
class _StreamVocoderState:
    codes: list[torch.Tensor] = field(default_factory=list)
    code_start_token: int = 0
    last_vocode_tokens: int = 0
    next_vocode_tokens: int = 0
    pending_tail: torch.Tensor | None = None
    total_tokens: int = 0


def resolve_stream_overlap_tokens(
    codec: Any, requested_overlap_tokens: int | None
) -> int:
    if requested_overlap_tokens is not None:
        if requested_overlap_tokens < 0:
            raise ValueError("stream_overlap_tokens must be >= 0")
        return requested_overlap_tokens

    delay_samples = int(codec.delay)
    if delay_samples <= 0:
        return 0
    frame_length = int(codec.frame_length)
    return (delay_samples + frame_length - 1) // frame_length


def build_stream_vocoder_chunk(
    state: _StreamVocoderState,
    codes: torch.Tensor,
    *,
    codec: Any,
    device: torch.device,
    stream_stride: int,
    stream_followup_stride: int,
    stream_overlap_tokens: int,
    stream_crossfade_samples: int,
) -> dict[str, Any] | None:
    assert codes.ndim == 2

    state.codes.append(
        codes.detach().to(device=device, dtype=torch.long, non_blocking=True)
    )

    total_tokens = state.total_tokens + int(codes.shape[1])
    state.total_tokens = total_tokens

    next_vocode_tokens = state.next_vocode_tokens or stream_stride
    if total_tokens < next_vocode_tokens:
        state.next_vocode_tokens = next_vocode_tokens
        return None

    chunk = _build_stream_vocoder_chunk(
        state,
        codec=codec,
        device=device,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_crossfade_samples=stream_crossfade_samples,
        is_final=False,
    )
    state.next_vocode_tokens = total_tokens + stream_followup_stride
    return chunk


def flush_stream_vocoder_chunk(
    state: _StreamVocoderState,
    *,
    codec: Any,
    device: torch.device,
    stream_overlap_tokens: int,
    stream_crossfade_samples: int,
) -> dict[str, Any] | None:
    pending_tail = state.pending_tail
    has_codes = bool(state.codes)
    has_pending_tail = pending_tail is not None and pending_tail.numel() > 0
    if not has_codes and not has_pending_tail:
        return None

    if not has_codes and has_pending_tail:
        state.pending_tail = None
        return _build_audio_chunk_payload(
            pending_tail,
            sample_rate=codec.sample_rate,
        )

    if state.total_tokens <= state.last_vocode_tokens and not has_pending_tail:
        return None

    return _build_stream_vocoder_chunk(
        state,
        codec=codec,
        device=device,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_crossfade_samples=stream_crossfade_samples,
        is_final=True,
    )


def _build_stream_vocoder_chunk(
    state: _StreamVocoderState,
    *,
    codec: Any,
    device: torch.device,
    stream_overlap_tokens: int,
    stream_crossfade_samples: int,
    is_final: bool,
) -> dict[str, Any] | None:
    if not state.codes:
        return None

    code_start_token = state.code_start_token
    total_tokens = state.total_tokens
    emitted_tokens = state.last_vocode_tokens
    if total_tokens <= emitted_tokens:
        if not is_final:
            return None
        pending_tail = state.pending_tail
        if pending_tail is None or pending_tail.numel() == 0:
            return None
        state.pending_tail = None
        return _build_audio_chunk_payload(
            pending_tail,
            sample_rate=codec.sample_rate,
        )

    output_codes = torch.cat(state.codes, dim=1)
    window_start_token = max(code_start_token, emitted_tokens - stream_overlap_tokens)
    window_offset = window_start_token - code_start_token
    window_codes = output_codes[:, window_offset:]
    codebook_codes = window_codes[1:].to(device=device, dtype=torch.long)

    with torch.no_grad():
        audio = codec.from_indices(codebook_codes[None])

    audio_tensor = audio[0, 0].float()
    overlap_token_count = emitted_tokens - window_start_token
    overlap_samples = int(overlap_token_count * codec.frame_length)
    if audio_tensor.shape[-1] <= overlap_samples:
        return None

    delta_audio = audio_tensor[overlap_samples:]
    if stream_crossfade_samples > 0:
        delta_audio = _apply_stream_crossfade(
            state,
            delta_audio,
            stream_crossfade_samples=stream_crossfade_samples,
            is_final=is_final,
        )
        if delta_audio is None:
            state.last_vocode_tokens = total_tokens
            trim_retained_stream_codes(
                state,
                keep_from_token=max(0, total_tokens - stream_overlap_tokens),
            )
            return None

    state.last_vocode_tokens = total_tokens
    if not is_final:
        trim_retained_stream_codes(
            state,
            keep_from_token=max(0, total_tokens - stream_overlap_tokens),
        )

    return _build_audio_chunk_payload(
        delta_audio,
        sample_rate=codec.sample_rate,
    )


def _apply_stream_crossfade(
    state: _StreamVocoderState,
    delta_audio: torch.Tensor,
    *,
    stream_crossfade_samples: int,
    is_final: bool,
) -> torch.Tensor | None:
    pending_tail = state.pending_tail
    if pending_tail is not None and pending_tail.numel() > 0:
        crossfade = min(
            int(stream_crossfade_samples),
            int(pending_tail.shape[-1]),
            int(delta_audio.shape[-1]),
        )
        if crossfade > 0:
            fade_in = torch.linspace(
                0.0,
                1.0,
                crossfade,
                dtype=delta_audio.dtype,
                device=delta_audio.device,
            )
            fade_out = 1.0 - fade_in
            blended = (
                pending_tail[-crossfade:] * fade_out + delta_audio[:crossfade] * fade_in
            )
            delta_audio = torch.cat(
                [pending_tail[:-crossfade], blended, delta_audio[crossfade:]]
            )
        else:
            delta_audio = torch.cat([pending_tail, delta_audio])

    if is_final:
        state.pending_tail = None
        return delta_audio

    hold = min(int(stream_crossfade_samples), int(delta_audio.shape[-1]))
    if hold > 0:
        state.pending_tail = delta_audio[-hold:].clone()
        delta_audio = delta_audio[:-hold]
    else:
        state.pending_tail = None

    if delta_audio.numel() == 0:
        return None
    return delta_audio


def trim_retained_stream_codes(
    state: _StreamVocoderState, *, keep_from_token: int
) -> None:
    retained_codes = state.codes
    if not retained_codes:
        return

    code_start_token = state.code_start_token
    if keep_from_token <= code_start_token:
        return

    drop_tokens = keep_from_token - code_start_token
    while drop_tokens > 0 and retained_codes:
        first_chunk = retained_codes[0]
        first_width = int(first_chunk.shape[1])
        if drop_tokens >= first_width:
            retained_codes.pop(0)
            code_start_token += first_width
            drop_tokens -= first_width
            continue

        retained_codes[0] = first_chunk[:, drop_tokens:].contiguous()
        code_start_token += drop_tokens
        drop_tokens = 0

    state.code_start_token = code_start_token


def _build_audio_chunk_payload(
    audio_data: torch.Tensor, *, sample_rate: int
) -> dict[str, Any]:
    return audio_waveform_payload(
        audio_data,
        sample_rate=sample_rate,
        modality="audio",
        source_hint="S2-Pro streaming",
    )


class S2ProVocoderScheduler(StreamingSimpleScheduler):
    """Fish S2-Pro vocoder scheduler with streaming and batch final paths."""

    def __init__(
        self,
        codec: Any,
        *,
        device: str,
        stream_stride: int = 10,
        stream_followup_stride: int = 90,
        stream_overlap_tokens: int | None = 20,
        stream_crossfade_samples: int = 512,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
    ):
        if stream_stride <= 0 or stream_followup_stride <= 0 or max_batch_size <= 0:
            raise ValueError(
                "stream_stride, stream_followup_stride, and max_batch_size must be > 0"
            )
        if stream_crossfade_samples < 0 or max_batch_wait_ms < 0:
            raise ValueError(
                "stream_crossfade_samples and max_batch_wait_ms must be >= 0"
            )

        self._codec = codec
        self._device = torch.device(device)
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_overlap_tokens = resolve_stream_overlap_tokens(
            codec, stream_overlap_tokens
        )
        self._stream_crossfade_samples = int(stream_crossfade_samples)
        self._stream_states: dict[str, _StreamVocoderState] = {}

        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )
        self._payloads = self._stream_payloads

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return self._is_streaming_payload(payload)

    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        self._validate_payload_state(payload)

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        del payload
        self._stream_states.setdefault(request_id, _StreamVocoderState())

    def on_stream_chunk(
        self, request_id: str, chunk: StreamItem
    ) -> list[OutgoingMessage]:
        state = self._stream_states.setdefault(request_id, _StreamVocoderState())
        codes = chunk.data
        if not isinstance(codes, torch.Tensor):
            raise TypeError(
                f"S2-Pro stream chunk for {request_id!r} must carry a torch.Tensor, "
                f"got {type(codes).__name__}"
            )
        output = build_stream_vocoder_chunk(
            state,
            codes,
            codec=self._codec,
            device=self._device,
            stream_stride=self._stream_stride,
            stream_followup_stride=self._stream_followup_stride,
            stream_overlap_tokens=self._stream_overlap_tokens,
            stream_crossfade_samples=self._stream_crossfade_samples,
        )
        if output is None:
            return []
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=output,
                metadata={"modality": "audio"},
            )
        ]

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        state = self._stream_states.get(request_id)
        if state is None:
            return []

        output = flush_stream_vocoder_chunk(
            state,
            codec=self._codec,
            device=self._device,
            stream_overlap_tokens=self._stream_overlap_tokens,
            stream_crossfade_samples=self._stream_crossfade_samples,
        )
        messages: list[OutgoingMessage] = []
        if output is not None:
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=output,
                    metadata={"modality": "audio"},
                )
            )

        payload = self._payloads[request_id]
        result = self._vocode_payload(payload)
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=result,
            )
        )
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        self._stream_states.pop(request_id, None)

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        return self._vocode_payloads([payload])[0]

    def _validate_payload_state(self, payload: StagePayload) -> S2ProState:
        state = S2ProState.from_dict(payload.data)
        if (
            state.output_codes is None
            or state.output_codes.ndim != 2
            or state.output_codes.shape[1] == 0
        ):
            raise ValueError(
                f"Request {payload.request_id}: S2-Pro generated no audio codec tokens"
            )
        return state

    def _vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        states = [self._validate_payload_state(payload) for payload in payloads]
        code_batches = [state.output_codes[1:].to(self._device) for state in states]
        lengths = [int(codes.shape[-1]) for codes in code_batches]
        max_len = max(lengths)
        padded = [
            torch.nn.functional.pad(codes, (0, max_len - length), value=0)
            for codes, length in zip(code_batches, lengths)
        ]
        batch_codes = torch.stack(padded, dim=0)

        with torch.no_grad():
            audio = self._codec.from_indices(batch_codes)

        samples_per_token = int(self._codec.frame_length)

        results: list[StagePayload] = []
        for idx, (payload, state, length) in enumerate(zip(payloads, states, lengths)):
            sample_len = int(length * samples_per_token)
            audio_np = audio[idx, 0, :sample_len].float().cpu()
            results.append(self._store_audio(payload, state, audio_np))
        return results

    def _store_audio(
        self,
        payload: StagePayload,
        state: S2ProState,
        audio_np: torch.Tensor,
    ) -> StagePayload:
        usage = payload.data.get("usage") or build_usage(state)
        state.audio_samples = audio_np
        state.sample_rate = self._codec.sample_rate
        data = state.to_dict()
        if usage is not None:
            data["usage"] = usage
        data["audio_data"] = audio_np.tolist()
        data["sample_rate"] = self._codec.sample_rate
        data["modality"] = "audio"
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=data,
        )

    @staticmethod
    def _is_streaming_payload(payload: StagePayload) -> bool:
        return bool(payload.request.params.get("stream"))
