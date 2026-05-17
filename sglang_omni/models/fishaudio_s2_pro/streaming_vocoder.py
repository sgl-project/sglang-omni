# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for FishAudio S2-Pro."""

from __future__ import annotations

import collections
import logging
import queue as _queue_mod
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_ABORTED_REQUEST_ID_LIMIT = 10000
_ABORTED_REQUEST_ID_RETAINED = 5000


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
    return {
        "audio_data": audio_data.cpu().tolist(),
        "sample_rate": sample_rate,
        "modality": "audio",
    }


def _build_usage(state: S2ProState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None

    usage = {
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "total_tokens": state.prompt_tokens + state.completion_tokens,
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


class S2ProVocoderScheduler:
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

        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._codec = codec
        self._device = torch.device(device)
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_overlap_tokens = resolve_stream_overlap_tokens(
            codec, stream_overlap_tokens
        )
        self._stream_crossfade_samples = int(stream_crossfade_samples)
        self._max_batch_size = int(max_batch_size)
        self._max_batch_wait_s = float(max_batch_wait_ms) / 1000.0
        self._running = False
        self._pending_messages: collections.deque[IncomingMessage] = collections.deque()

        self._payloads: dict[str, StagePayload] = {}
        self._stream_states: dict[str, _StreamVocoderState] = {}
        self._pending_done: set[str] = set()
        self._aborted_request_ids: set[str] = set()

    def start(self) -> None:
        self._running = True
        while self._running:
            msg = self._next_message()
            if msg is None:
                continue
            if msg.request_id in self._aborted_request_ids:
                continue
            try:
                if msg.type == "new_request":
                    self._handle_new_request_batch(self._collect_new_request_batch(msg))
                elif msg.type == "stream_chunk":
                    self._on_chunk(msg.request_id, msg.data)
                elif msg.type == "stream_done":
                    self._on_done(msg.request_id)
                else:
                    raise ValueError(f"Unsupported vocoder message type: {msg.type}")
            except Exception as exc:
                logger.exception("S2ProVocoderScheduler failed for %s", msg.request_id)
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id,
                        type="error",
                        data=exc,
                    )
                )
                self.abort(msg.request_id)

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._record_aborted_request_id(request_id)
        self._clear_request_state(request_id, keep_aborted=True)

    def _record_aborted_request_id(self, request_id: str) -> None:
        self._aborted_request_ids.add(request_id)
        if len(self._aborted_request_ids) <= _ABORTED_REQUEST_ID_LIMIT:
            return

        # Note (Ratish): keep aborted ids bounded for late queued messages;
        # do not drain the shared inbox because it owns request order.
        excess = len(self._aborted_request_ids) - _ABORTED_REQUEST_ID_RETAINED
        to_remove = list(self._aborted_request_ids)[:excess]
        self._aborted_request_ids -= set(to_remove)

    def _clear_request_state(
        self, request_id: str, *, keep_aborted: bool = False
    ) -> None:
        self._payloads.pop(request_id, None)
        self._stream_states.pop(request_id, None)
        self._pending_done.discard(request_id)
        if not keep_aborted:
            self._aborted_request_ids.discard(request_id)

    def _next_message(self) -> IncomingMessage | None:
        if self._pending_messages:
            return self._pending_messages.popleft()
        try:
            return self.inbox.get(timeout=0.1)
        except _queue_mod.Empty:
            return None

    def _collect_new_request_batch(
        self, first_msg: IncomingMessage
    ) -> list[IncomingMessage]:
        batch = [first_msg]
        if self._max_batch_size <= 1 or self._is_streaming_payload(first_msg.data):
            return batch

        deadline = time.monotonic() + self._max_batch_wait_s
        while len(batch) < self._max_batch_size:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = self.inbox.get(timeout=remaining)
                except _queue_mod.Empty:
                    break

            if msg.request_id in self._aborted_request_ids:
                continue
            if msg.type == "new_request" and not self._is_streaming_payload(msg.data):
                batch.append(msg)
            else:
                self._pending_messages.append(msg)
                break
        return batch

    def _handle_new_request_batch(self, batch: list[IncomingMessage]) -> None:
        streaming = []
        non_streaming = []
        for msg in batch:
            if self._is_streaming_payload(msg.data):
                streaming.append(msg)
            else:
                non_streaming.append(msg)
        for msg in streaming:
            self._on_streaming_new_request(msg.request_id, msg.data)
        if non_streaming:
            self._vocode_non_streaming_batch(non_streaming)

    def _on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self._aborted_request_ids.discard(request_id)
        self._payloads[request_id] = payload
        self._stream_states.setdefault(request_id, _StreamVocoderState())
        if request_id in self._pending_done:
            self._pending_done.discard(request_id)
            self._on_done(request_id)

    def _on_chunk(self, request_id: str, chunk: Any) -> None:
        if request_id in self._aborted_request_ids:
            return
        state = self._stream_states.setdefault(request_id, _StreamVocoderState())
        codes = chunk.data
        assert isinstance(codes, torch.Tensor)
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
        if output is not None and request_id not in self._aborted_request_ids:
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=output,
                    metadata={"modality": "audio"},
                )
            )

    def _on_done(self, request_id: str) -> None:
        if request_id in self._aborted_request_ids:
            return
        if request_id not in self._stream_states:
            return
        if request_id not in self._payloads:
            self._pending_done.add(request_id)
            return

        state = self._stream_states[request_id]
        output = flush_stream_vocoder_chunk(
            state,
            codec=self._codec,
            device=self._device,
            stream_overlap_tokens=self._stream_overlap_tokens,
            stream_crossfade_samples=self._stream_crossfade_samples,
        )
        if output is not None and request_id not in self._aborted_request_ids:
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=output,
                    metadata={"modality": "audio"},
                )
            )

        payload = self._payloads.get(request_id)
        if payload is None or request_id in self._aborted_request_ids:
            return
        result = self._vocode_payload(payload)
        if request_id in self._aborted_request_ids:
            return
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=result,
            )
        )
        self._clear_request_state(request_id)

    def _vocode_non_streaming_batch(self, batch: list[IncomingMessage]) -> None:
        batch = [
            msg for msg in batch if msg.request_id not in self._aborted_request_ids
        ]
        if not batch:
            return
        for msg in batch:
            self._pending_done.discard(msg.request_id)

        valid_messages: list[IncomingMessage] = []
        valid_payloads: list[StagePayload] = []
        for msg in batch:
            try:
                self._validate_payload_state(msg.data)
            except Exception as exc:
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id,
                        type="error",
                        data=exc,
                    )
                )
                continue
            valid_messages.append(msg)
            valid_payloads.append(msg.data)

        if not valid_messages:
            return

        try:
            results = self._vocode_payloads(valid_payloads)
        except Exception as exc:
            for msg in valid_messages:
                self.outbox.put(
                    OutgoingMessage(request_id=msg.request_id, type="error", data=exc)
                )
            return

        for msg, result in zip(valid_messages, results):
            if msg.request_id in self._aborted_request_ids:
                continue
            self.outbox.put(
                OutgoingMessage(
                    request_id=msg.request_id,
                    type="result",
                    data=result,
                )
            )

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
        usage = payload.data.get("usage") or _build_usage(state)
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
