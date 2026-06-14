# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for MOSS-TTS Local (v1.5).

The MOSS-Audio-Tokenizer-v2 decoder is natively streamable: every block is
causal and carries per-slot state under ``model.streaming()``, so chunked
decoding matches one offline decode regardless of chunk boundaries. This
scheduler holds ONE long-lived batched streaming session on the stage's
codec: streaming requests occupy a slot for their lifetime, non-streaming
requests decode through dedicated offline slots of the same session (the
codec forbids nested ``streaming()`` contexts), and every step advances all
participating slots by one uniform length with the rest masked via
``exec_mask``. Until the first streaming request arrives, non-streaming
requests take the pre-existing ``processor.decode_audio_codes`` path.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
from sglang_omni.models.tts_streaming import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

_SOURCE_HINT = "MOSS-TTS Local"


def _resolve_sample_rate(processor: Any) -> int:
    return int(
        getattr(getattr(processor, "model_config", None), "sampling_rate", 0)
        or getattr(
            getattr(getattr(processor, "audio_tokenizer", None), "config", None),
            "sampling_rate",
            0,
        )
        or 48000
    )


def _build_usage(state: MossTTSLocalState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage = {
        "prompt_tokens": int(state.prompt_tokens),
        "completion_tokens": int(state.completion_tokens),
        "total_tokens": int(state.prompt_tokens + state.completion_tokens),
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


class _CodecStreamSession:
    """A persistent batched ``codec.streaming()`` session with slot bookkeeping.

    Slots ``[0, stream_slots)`` are held by live streaming requests; slots
    ``[stream_slots, stream_slots + offline_slots)`` are reserved for
    non-streaming batch decodes so an offline request can never wait on (or
    deadlock against) a long-lived stream. All methods must be called from the
    scheduler loop thread.
    """

    def __init__(self, codec: Any, *, stream_slots: int, offline_slots: int) -> None:
        self._codec = codec
        self._stream_slots = int(stream_slots)
        self._offline_slots = int(offline_slots)
        self._batch_size = self._stream_slots + self._offline_slots
        self._device = next(codec.parameters()).device
        self._free_stream_slots = list(range(self._stream_slots))
        self._exit_stack = contextlib.ExitStack()
        self._closed = False
        with torch.no_grad():
            self._exit_stack.enter_context(codec.streaming(self._batch_size))

    def acquire(self) -> int | None:
        if not self._free_stream_slots:
            return None
        return self._free_stream_slots.pop()

    def release(self, slot: int) -> None:
        if self._closed:
            return
        self._reset_slots([slot])
        self._free_stream_slots.append(slot)

    def close(self) -> None:
        if self._closed:
            return
        with torch.no_grad():
            self._exit_stack.close()
        self._closed = True

    def _reset_slots(self, slots: list[int]) -> None:
        reset_mask = torch.zeros(
            self._batch_size, dtype=torch.bool, device=self._device
        )
        reset_mask[slots] = True

        def _reset(module: Any) -> None:
            state = getattr(module, "_streaming_state", None)
            if state is not None:
                state.reset(reset_mask.to(state.device))

        with torch.no_grad():
            self._codec.apply(_reset)

    def step(self, slot_codes: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Advance the participating slots by one uniform-length step.

        ``slot_codes`` maps slot -> ``[n_vq, T]`` codes with the SAME ``T``
        for every participant. Returns slot -> ``[channels, samples]`` float32
        CPU audio.
        """
        if not slot_codes:
            return {}
        step_lengths = {int(codes.shape[1]) for codes in slot_codes.values()}
        if len(step_lengths) != 1:
            raise ValueError(
                f"streaming step requires a uniform length, got {sorted(step_lengths)}"
            )
        (step_t,) = step_lengths
        n_vq = int(next(iter(slot_codes.values())).shape[0])
        codes_step = torch.zeros(
            n_vq, self._batch_size, step_t, dtype=torch.long, device=self._device
        )
        codes_lengths = torch.zeros(
            self._batch_size, dtype=torch.long, device=self._device
        )
        exec_mask = torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)
        for slot, codes in slot_codes.items():
            codes_step[:, slot, :] = codes.to(device=self._device, dtype=torch.long)
            codes_lengths[slot] = step_t
            exec_mask[slot] = True
        with torch.no_grad():
            self._codec._set_streaming_exec_mask(exec_mask)
            result = self._codec._decode_frame(codes_step, codes_lengths)
        # One batched D2H per step, active slots only.
        slots = list(slot_codes)
        audio_cpu = result.audio[slots].detach().to("cpu", torch.float32)
        lengths_cpu = result.audio_lengths[slots].detach().to("cpu")
        out: dict[int, torch.Tensor] = {}
        for index, slot in enumerate(slots):
            n_samples = int(lengths_cpu[index])
            out[slot] = audio_cpu[index, :, :n_samples]
        return out

    def decode_offline(
        self, codes_list: list[torch.Tensor], *, max_step_frames: int
    ) -> list[torch.Tensor]:
        """Decode complete utterances ``[n_vq, T]`` through the offline lane.

        Replays the codec's own chunked ``batch_decode`` step plan so the
        output matches the non-session ``processor.decode_audio_codes`` path.
        """
        wavs: list[torch.Tensor] = []
        for wave_start in range(0, len(codes_list), self._offline_slots):
            wave = codes_list[wave_start : wave_start + self._offline_slots]
            slots = [self._stream_slots + i for i in range(len(wave))]
            self._reset_slots(slots)
            cursors = [0] * len(wave)
            chunks: list[list[torch.Tensor]] = [[] for _ in wave]
            while True:
                remaining = [
                    int(codes.shape[1]) - cur for codes, cur in zip(wave, cursors)
                ]
                positive = [r for r in remaining if r > 0]
                if not positive:
                    break
                if any(r >= max_step_frames for r in positive):
                    step_t = max_step_frames
                else:
                    step_t = min(positive)
                plan = {
                    slots[i]: wave[i][:, cursors[i] : cursors[i] + step_t]
                    for i, rem in enumerate(remaining)
                    if rem >= step_t
                }
                decoded = self.step(plan)
                for i in range(len(wave)):
                    if slots[i] in plan:
                        chunks[i].append(decoded[slots[i]])
                        cursors[i] += step_t
            for item_chunks in chunks:
                wavs.append(torch.cat(item_chunks, dim=-1))
        return wavs


@dataclass
class _LocalStreamState:
    slot: int | None = None
    pending: list[torch.Tensor] = field(default_factory=list)
    n_vq: int | None = None
    initial_chunk_frames: int = 0
    threshold: int = 0
    emitted_any: bool = False


class MossTTSLocalStreamingVocoderScheduler(StreamingSimpleScheduler):
    """Decode MOSS-TTS Local codec rows incrementally on the v2 codec."""

    def __init__(
        self,
        processor: Any,
        *,
        stream_slots: int = 8,
        stream_chunk_frames: int = 25,
        initial_chunk_frames: int = 5,
        max_step_frames: int = 100,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
    ) -> None:
        if stream_slots < 1:
            raise ValueError(f"stream_slots must be >= 1, got {stream_slots}")
        if not 0 < stream_chunk_frames <= max_step_frames:
            raise ValueError(
                "stream_chunk_frames must be in (0, max_step_frames], got "
                f"{stream_chunk_frames} (max_step_frames={max_step_frames})"
            )
        codec = getattr(processor, "audio_tokenizer", None)
        if codec is None:
            raise RuntimeError(
                "MOSS-TTS Local vocoder requires processor.audio_tokenizer"
            )
        missing = [
            name
            for name in ("streaming", "_set_streaming_exec_mask", "_decode_frame")
            if not hasattr(codec, name)
        ]
        if missing:
            raise RuntimeError(
                f"MOSS-TTS Local streaming vocoder: codec is missing {missing}; "
                "the installed MOSS-Audio-Tokenizer-v2 version is incompatible"
            )
        self._processor = processor
        self._codec = codec
        self._stream_slots = int(stream_slots)
        self._stream_chunk_frames = int(stream_chunk_frames)
        self._default_initial_chunk_frames = max(
            0, min(int(initial_chunk_frames), int(stream_chunk_frames))
        )
        self._max_step_frames = int(max_step_frames)
        self._offline_slots = max(int(max_batch_size), 1)
        self._n_vq = int(processor.model_config.n_vq)
        self._sample_rate = _resolve_sample_rate(processor)
        self._session: _CodecStreamSession | None = None
        self._stream_states: dict[str, _LocalStreamState] = {}
        super().__init__(
            self._vocode,
            batch_compute_fn=self._vocode_batch,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def start(self) -> None:
        try:
            super().start()
        finally:
            self._close_streaming_session()

    def stop(self) -> None:
        was_running = self._running
        super().stop()
        if not was_running:
            self._close_streaming_session()

    def _close_streaming_session(self) -> None:
        with self._state_lock:
            self._stream_states.clear()
            if self._session is not None:
                self._session.close()
                self._session = None

    # ------------------------------------------------------------------
    # Streaming hooks
    # ------------------------------------------------------------------

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        params = payload.request.params
        if not isinstance(params, dict):
            raise TypeError(
                f"MOSS-TTS Local request params must be a dict, got "
                f"{type(params).__name__}"
            )
        return bool(params.get("stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        state = self._stream_states.setdefault(request_id, _LocalStreamState())
        params = (
            payload.request.params if isinstance(payload.request.params, dict) else None
        )
        self._latch_thresholds(state, params)

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        state = self._stream_states.setdefault(request_id, _LocalStreamState())
        self._latch_stream_metadata(request_id, state, item.metadata)

        row = item.data
        if not isinstance(row, torch.Tensor):
            raise TypeError(
                f"MOSS-TTS Local stream chunk for {request_id!r} must carry a "
                f"torch.Tensor, got {type(row).__name__}"
            )
        row = row.to(dtype=torch.long)
        n_vq = state.n_vq if state.n_vq is not None else self._n_vq
        if row.ndim != 1 or int(row.shape[0]) < n_vq + 1:
            raise ValueError(
                f"MOSS-TTS Local stream chunk must be a 1-D row with at least "
                f"{n_vq + 1} channels (text + codes), got {tuple(row.shape)}"
            )
        # Row layout matches output_rows: [text_token, code_0, ..., code_{n_vq-1}].
        state.pending.append(row[1 : 1 + n_vq])
        self._ensure_slot(state)
        self._pump_streams()
        return []

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        payload = self._stream_payloads[request_id]
        state = self._stream_states.setdefault(request_id, _LocalStreamState())

        audio_parts: list[torch.Tensor] = []
        if state.slot is None and state.pending:
            # Slot-starved: every frame is still buffered, decode offline.
            codes = torch.stack(state.pending, dim=1)
            state.pending = []
            audio_parts.extend(
                self._ensure_session().decode_offline(
                    [codes], max_step_frames=self._max_step_frames
                )
            )
        elif state.slot is not None:
            session = self._ensure_session()
            while state.pending:
                step_t = min(len(state.pending), self._max_step_frames)
                codes = torch.stack(state.pending[:step_t], dim=1)
                del state.pending[:step_t]
                audio_parts.append(session.step({state.slot: codes})[state.slot])
            session.release(state.slot)
            state.slot = None

        if not audio_parts and not state.emitted_any:
            fallback = self._decode_payload_codes(payload)
            if fallback is not None:
                audio_parts.append(fallback)

        messages: list[OutgoingMessage] = []
        if audio_parts:
            audio = torch.cat(audio_parts, dim=-1)
            state.emitted_any = True
            messages.append(self._chunk_message(request_id, audio))

        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = _build_usage(MossTTSLocalState.from_dict(payload.data))
        if usage is not None:
            final_data["usage"] = usage
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=final_data,
                ),
            )
        )
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        state = self._stream_states.pop(request_id, None)
        if state is not None and state.slot is not None and self._session is not None:
            self._session.release(state.slot)

    # ------------------------------------------------------------------
    # Streaming internals
    # ------------------------------------------------------------------

    def _ensure_session(self) -> _CodecStreamSession:
        if self._session is None:
            self._session = _CodecStreamSession(
                self._codec,
                stream_slots=self._stream_slots,
                offline_slots=self._offline_slots,
            )
        return self._session

    def _ensure_slot(self, state: _LocalStreamState) -> None:
        if state.slot is None:
            state.slot = self._ensure_session().acquire()

    def _latch_stream_metadata(
        self,
        request_id: str,
        state: _LocalStreamState,
        metadata: dict[str, Any] | None,
    ) -> None:
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"MOSS-TTS Local stream chunk for {request_id!r} is missing metadata"
            )
        if metadata.get("modality") not in (None, "audio_codes"):
            raise ValueError(
                f"MOSS-TTS Local stream chunk modality must be audio_codes, got "
                f"{metadata.get('modality')!r}"
            )
        if metadata.get("stream") is not True:
            raise RuntimeError(
                f"MOSS-TTS Local stream chunk for {request_id!r} must include "
                "metadata['stream'] == True"
            )
        n_vq = metadata.get("n_vq")
        if n_vq is not None:
            n_vq = int(n_vq)
            if state.n_vq is not None and state.n_vq != n_vq:
                raise ValueError(
                    f"MOSS-TTS Local stream n_vq changed for {request_id!r}: "
                    f"{state.n_vq} -> {n_vq}"
                )
            state.n_vq = n_vq
        if state.threshold == 0:
            self._latch_thresholds(state, metadata)

    def _latch_thresholds(
        self, state: _LocalStreamState, params: Mapping[str, Any] | None
    ) -> None:
        explicit = (
            isinstance(params, Mapping)
            and params.get(INITIAL_CODEC_CHUNK_FRAMES_PARAM) is not None
        )
        if explicit:
            # Explicit 0 opts out of a smaller first chunk.
            state.initial_chunk_frames = resolve_initial_codec_chunk_frames(
                params,
                steady_chunk_frames=self._stream_chunk_frames,
            )
        else:
            state.initial_chunk_frames = self._default_initial_chunk_frames
        if state.initial_chunk_frames > 0 and not state.emitted_any:
            state.threshold = state.initial_chunk_frames
        else:
            state.threshold = self._stream_chunk_frames

    def _pump_streams(self) -> None:
        """Decode every stream whose buffer crossed its threshold.

        A step costs one decoder forward over the full slot width, so a due
        stream does not step alone: peers holding at least the join floor
        ride along in the same forward. Chunks go to the outbox per
        iteration. A failed step leaves its participants' slot state
        indeterminate, so all of them are failed and aborted.
        """
        join_floor = max(
            1, min(self._default_initial_chunk_frames or 5, self._stream_chunk_frames)
        )
        while True:
            slotted = [
                (request_id, state)
                for request_id, state in self._stream_states.items()
                if state.slot is not None and state.threshold > 0
            ]
            due = [
                entry
                for entry in slotted
                if len(entry[1].pending) >= entry[1].threshold
            ]
            if not due:
                return
            floor = min(
                min(len(state.pending) for _, state in due),
                join_floor,
            )
            participants = [
                entry
                for entry in slotted
                if self._can_join_coalesced_step(entry[1], floor)
            ]
            step_t = min(
                min(len(state.pending) for _, state in participants),
                self._max_step_frames,
            )
            plan: dict[int, torch.Tensor] = {}
            for _, state in participants:
                plan[state.slot] = torch.stack(state.pending[:step_t], dim=1)
            try:
                decoded = self._ensure_session().step(plan)
            except Exception as exc:
                logger.exception(
                    "MOSS-TTS Local streaming decode step failed; aborting %d "
                    "participating request(s)",
                    len(participants),
                )
                for request_id, _ in participants:
                    self._emit_error(request_id, exc)
                    self.abort(request_id)
                return
            for request_id, state in participants:
                del state.pending[:step_t]
                state.emitted_any = True
                state.threshold = self._stream_chunk_frames
                if not self._is_aborted(request_id):
                    self.outbox.put(
                        self._chunk_message(request_id, decoded[state.slot])
                    )

    @staticmethod
    def _can_join_coalesced_step(state: _LocalStreamState, floor: int) -> bool:
        if len(state.pending) >= state.threshold:
            return True
        if not state.emitted_any:
            return False
        return len(state.pending) >= floor

    def _chunk_message(self, request_id: str, audio: torch.Tensor) -> OutgoingMessage:
        data = audio_waveform_payload(
            audio.detach().to("cpu", torch.float32),
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint=f"{_SOURCE_HINT} streaming",
            keep_channels=True,
        )
        return OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=data,
            metadata={"modality": "audio"},
        )

    def _decode_payload_codes(self, payload: StagePayload) -> torch.Tensor | None:
        state = MossTTSLocalState.from_dict(payload.data)
        if state.audio_codes is None:
            return None
        rows = torch.as_tensor(state.audio_codes, dtype=torch.long)
        if rows.numel() == 0:
            return None
        codes = rows[:, : self._n_vq].transpose(0, 1).contiguous()
        return self._ensure_session().decode_offline(
            [codes], max_step_frames=self._max_step_frames
        )[0]

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    def _prepare_codes(
        self, payload: StagePayload
    ) -> tuple[MossTTSLocalState, torch.Tensor | None]:
        state = MossTTSLocalState.from_dict(payload.data)
        if state.audio_codes is None:
            raise RuntimeError("MOSS-TTS Local vocoder requires audio_codes")
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        if codes.numel() == 0:
            # Emit no audio: only this request fails downstream, not the batch.
            return state, None
        return state, codes

    def _store_vocoder_result(
        self,
        payload: StagePayload,
        state: MossTTSLocalState,
        wav: torch.Tensor,
    ) -> StagePayload:
        # The v2 codec is natively stereo: keep [channels, samples] end to end.
        audio_payload = audio_waveform_payload(
            wav, source_hint=_SOURCE_HINT, keep_channels=True
        )
        state.audio_codes = None
        state.sample_rate = self._sample_rate
        payload.data = state.to_dict()
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = _build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _decode_codes_rows(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode ``[T, >=n_vq]`` row tensors to fp32 CPU waveforms."""
        if self._session is None:
            # Processor path opens its own streaming context; illegal once a
            # session is live.
            return [
                torch.as_tensor(wav).detach().to("cpu")
                for wav in self._processor.decode_audio_codes(codes_list)
            ]
        channels_first = [
            codes[:, : self._n_vq].transpose(0, 1).contiguous() for codes in codes_list
        ]
        # abort() resets slots under _state_lock from other threads; serialize
        # every session access on the same lock.
        with self._state_lock:
            wavs = self._session.decode_offline(
                channels_first, max_step_frames=self._max_step_frames
            )
        return [wav.detach().to("cpu", torch.float32).contiguous() for wav in wavs]

    def _vocode_batch(self, payloads: list[StagePayload]) -> list[StagePayload]:
        prepared = [self._prepare_codes(payload) for payload in payloads]
        codes_list = [codes for _, codes in prepared if codes is not None]
        decoded = iter(self._decode_codes_rows(codes_list)) if codes_list else iter(())
        results = []
        for payload, (state, codes) in zip(payloads, prepared):
            if codes is None:
                state.audio_codes = None
                payload.data = state.to_dict()
                results.append(payload)
                continue
            results.append(self._store_vocoder_result(payload, state, next(decoded)))
        return results

    def _vocode(self, payload: StagePayload) -> StagePayload:
        return self._vocode_batch([payload])[0]


__all__ = ["MossTTSLocalStreamingVocoderScheduler"]
