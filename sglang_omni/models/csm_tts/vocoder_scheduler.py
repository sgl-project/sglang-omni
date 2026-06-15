# SPDX-License-Identifier: Apache-2.0
# Follows the sglang-omni Higgs `StreamingSimpleScheduler` streaming-vocoder pattern.
"""Streaming vocoder stage for CSM — Mimi decode with overlap-trim seam
handling at 12.5 Hz.

Follows the higgs ``StreamingSimpleScheduler`` lifecycle exactly (latch
contract, ``_pending_done`` buffering, abort → ``clear_stream_state``, slim
terminal result). Key delta vs higgs: CSM has NO delay pattern,
so ``raw_frames_available = len(rows)`` — frame 0 is decodable the moment it
arrives (no ``rows - N + 1`` reversal warmup). One row = one ``[32]`` tensor
= 80 ms = 1920 samples.

Frame-math knobs at 12.5 Hz (starting points; a
seam-audibility A/B over overlap ∈ {2, 4, 6}):

- ``stream_stride=13`` (~1.04 s first decode; overridden to 1 by the
  first-chunk knob on streaming paths)
- ``stream_followup_stride=6`` (480 ms steady chunks)
- ``stream_overlap_frames=4`` (320 ms re-decoded left context — Mimi's decode
  path has no conv-state cache, edge frames differ per the HF docstring)
- ``stream_holdback_frames=2`` (160 ms; never emit codec-edge frames
  mid-stream; final flush emits all)

Emission per delta: re-decode ``[emitted - overlap : available - holdback]``,
trim ``overlap * 1920`` samples, emit exactly ``new_frames * 1920`` —
seam-free without crossfade. This STATELESS overlap-trim decode is the default;
the optional stateful ``decoder_past_key_values`` path is mutually exclusive with
overlap re-decode and stays gated on an A/B.

The three OOB fences: (1) engine never streams EOS/STOP frames;
(2) ``sanitize_for_mimi`` clamp + counter on every matrix entering Mimi;
(3) final flush / non-streaming trim at the first all-32-zero frame (HF
cutoff semantics, generation_csm.py:471-477 — mirrors HF's stop(31)/trim(32)
inconsistency exactly).


"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.csm_tts.audio_codec import (
    CsmMimiCodec,
    sanitize_for_mimi,
    trim_at_first_all_zero_frame,
)
from sglang_omni.models.csm_tts.payload_types import CsmTtsState
from sglang_omni.models.tts_streaming import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler

logger = logging.getLogger(__name__)

__all__ = ["CsmStreamState", "CsmStreamingVocoderScheduler"]

# Known CSM stream contract; used as the latch default when neither the
# payload nor the chunk metadata carries the fields (CsmTtsState has no
# num_codebooks/codebook_size fields — they're model constants).
_NUM_CODEBOOKS = 32
_CODEBOOK_VOCAB = 2051


@dataclass
class CsmStreamState:
    """Per-request streaming state (dropped in ``clear_stream_state``)."""

    rows: list[torch.Tensor] = field(default_factory=list)  # each int64 [32]
    frames_emitted: int = 0
    samples_emitted: int = 0
    num_codebooks: int | None = None  # latched: 32
    codebook_size: int | None = None  # latched: 2051
    initial_codec_chunk_frames: int | None = None
    clamp_count: int = 0  # sanitize_for_mimi telemetry (gate only at temp=0)
    past_key_values: Any | None = None  # stateful path ONLY
    done: bool = False


class CsmStreamingVocoderScheduler(StreamingSimpleScheduler):
    """Decode CSM code rows incrementally; batched final decode for the
    non-streaming path (``_vocode_payloads`` + ``decode_batch``)."""

    def __init__(
        self,
        codec: CsmMimiCodec,
        *,
        stream_stride: int = 13,
        stream_followup_stride: int = 6,
        stream_overlap_frames: int = 4,
        stream_holdback_frames: int = 2,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
    ) -> None:
        """Validate knobs, hold the shared :class:`CsmMimiCodec`, init the
        per-request ``_stream_states`` dict, then ``super().__init__(
        self._vocode_payload, batch_compute_fn=self._vocode_payloads,
        max_batch_size=..., max_batch_wait_ms=...)`` (higgs ctor shape)."""
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream_stride and stream_followup_stride must be > 0")
        if stream_overlap_frames < 0:
            raise ValueError("stream_overlap_frames must be >= 0")
        if stream_holdback_frames < 0:
            raise ValueError("stream_holdback_frames must be >= 0")

        self._codec = codec
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_overlap_frames = int(stream_overlap_frames)
        self._stream_holdback_frames = int(stream_holdback_frames)
        self._sample_rate = CsmMimiCodec.SAMPLE_RATE
        self._samples_per_frame = CsmMimiCodec.SAMPLES_PER_FRAME
        self._stream_states: dict[str, CsmStreamState] = {}
        self._clamp_count_total = 0  # process-wide fence-#2 counter

        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    # --- StreamingSimpleScheduler hooks ---------------------------------------

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        """``bool(payload.request.params.get("stream", False))``."""
        params = payload.request.params
        if not isinstance(params, dict):
            raise TypeError(
                f"CSM request params must be a dict, got {type(params).__name__}"
            )
        return bool(params.get("stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        """Latch the stream contract (``num_codebooks=32`` /
        ``codebook_size=2051``) + ``initial_codec_chunk_frames`` from the
        payload/params (higgs latch discipline; the stage runs with
        ``can_accept_stream_before_payload=True``)."""
        state = self._stream_states.setdefault(request_id, CsmStreamState())
        data = payload.data if isinstance(payload.data, dict) else {}
        self._latch_stream_contract(
            request_id,
            state,
            num_codebooks=data.get(
                "num_codebooks", state.num_codebooks or _NUM_CODEBOOKS
            ),
            codebook_size=data.get(
                "codebook_size", state.codebook_size or _CODEBOOK_VOCAB
            ),
            source="payload",
        )
        params = (
            payload.request.params if isinstance(payload.request.params, dict) else None
        )
        self._latch_initial_codec_chunk_frames_from_mapping(state, params)

    def on_stream_chunk(self, request_id: str, item: Any) -> list[OutgoingMessage]:
        """Append the ``[32]`` int64 row from ``item.data`` to the request's
        ``rows`` and run :meth:`_decode_delta`.

        Returns:
            Zero or more audio-chunk ``OutgoingMessage``s (raw PCM payloads of
            exactly ``new_frames * 1920`` samples each).
        """
        state = self._stream_states.setdefault(request_id, CsmStreamState())
        self._latch_stream_metadata(request_id, state, item.metadata)

        row = item.data
        if not isinstance(row, torch.Tensor):
            raise TypeError(
                f"CSM stream chunk for {request_id!r} must carry a torch.Tensor, "
                f"got {type(row).__name__}"
            )
        row = row.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        num_codebooks = state.num_codebooks or _NUM_CODEBOOKS
        if int(row.shape[0]) != num_codebooks:
            raise ValueError(
                f"CSM stream chunk has {int(row.shape[0])} codebooks, "
                f"expected {num_codebooks}"
            )
        state.rows.append(row)
        return self._decode_delta(request_id, state)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        """Final flush (emit all held-back frames, trimmed at the first
        all-32-zero frame — fence #3) + slim terminal result with usage
        (prompt_tokens, completion_frames, engine_time_s). Zero-chunk streams
        (frame-0 EOS) produce a well-formed empty/near-empty
        audio response matching HF ``cutoff_idx=0``."""
        payload = self._stream_payloads[request_id]
        state = self._stream_states.setdefault(request_id, CsmStreamState())
        messages = self._decode_delta(request_id, state, final=True)
        if not messages and state.frames_emitted == 0:
            # Zero-chunk stream (frame-0 EOS): fall back to the engine payload's
            # output_frames with exact HF trim parity — empty when cb31 == 0,
            # one 80 ms frame when cb31 != 0.
            output = self._audio_payload_from_stage_payload(request_id, state, payload)
            if output is not None:
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=output,
                        metadata={"modality": "audio"},
                    )
                )
        state.done = True

        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = self._build_usage(CsmTtsState.from_dict(payload.data))
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
        """Drop the per-request :class:`CsmStreamState` INCLUDING any Mimi
        ``past_key_values`` — the cross-request state-leak guard (abort path
        calls this too)."""
        self._stream_states.pop(request_id, None)

    # --- latch helpers ----------------------------------------------------------

    def _latch_stream_metadata(
        self,
        request_id: str,
        state: CsmStreamState,
        metadata: dict[str, Any] | None,
    ) -> None:
        if not isinstance(metadata, dict):
            # Engine chunks always carry stream_metadata; tolerate absence by
            # falling back to the CSM constants (latched-once discipline kept).
            self._latch_stream_contract(
                request_id,
                state,
                num_codebooks=state.num_codebooks or _NUM_CODEBOOKS,
                codebook_size=state.codebook_size or _CODEBOOK_VOCAB,
                source="stream metadata defaults",
            )
            return
        if metadata.get("modality") not in (None, "audio_codes"):
            raise ValueError(
                f"CSM stream chunk modality must be audio_codes, got "
                f"{metadata.get('modality')!r}"
            )
        if metadata.get("stream") is not True:
            raise RuntimeError(
                f"CSM stream chunk for {request_id!r} must include "
                "metadata['stream'] == True"
            )
        self._latch_stream_contract(
            request_id,
            state,
            num_codebooks=metadata.get(
                "num_codebooks", state.num_codebooks or _NUM_CODEBOOKS
            ),
            codebook_size=metadata.get(
                "codebook_size", state.codebook_size or _CODEBOOK_VOCAB
            ),
            source="stream metadata",
        )
        if INITIAL_CODEC_CHUNK_FRAMES_PARAM in metadata:
            self._latch_initial_codec_chunk_frames_from_mapping(state, metadata)

    @staticmethod
    def _latch_stream_contract(
        request_id: str,
        state: CsmStreamState,
        *,
        num_codebooks: Any,
        codebook_size: Any,
        source: str,
    ) -> None:
        try:
            num_codebooks_i = int(num_codebooks)
            codebook_size_i = int(codebook_size)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"CSM {source} for {request_id!r} must include integer "
                "num_codebooks and codebook_size"
            ) from exc
        if num_codebooks_i != _NUM_CODEBOOKS or codebook_size_i != _CODEBOOK_VOCAB:
            raise ValueError(
                f"CSM {source} for {request_id!r} has invalid contract "
                f"num_codebooks={num_codebooks_i}, codebook_size={codebook_size_i}; "
                f"expected {_NUM_CODEBOOKS}/{_CODEBOOK_VOCAB}"
            )
        if state.num_codebooks is not None and state.num_codebooks != num_codebooks_i:
            raise ValueError(
                f"CSM stream num_codebooks changed for {request_id!r}: "
                f"{state.num_codebooks} -> {num_codebooks_i}"
            )
        if state.codebook_size is not None and state.codebook_size != codebook_size_i:
            raise ValueError(
                f"CSM stream codebook_size changed for {request_id!r}: "
                f"{state.codebook_size} -> {codebook_size_i}"
            )
        state.num_codebooks = num_codebooks_i
        state.codebook_size = codebook_size_i

    def _latch_initial_codec_chunk_frames_from_mapping(
        self,
        state: CsmStreamState,
        params: Mapping[str, Any] | None,
    ) -> None:
        # No delay pattern: the steady first chunk IS stream_stride frames.
        # Only overwrite when the mapping carries the key — chunks (and their
        # metadata latch) may legally arrive BEFORE the engine payload.
        if params is not None and INITIAL_CODEC_CHUNK_FRAMES_PARAM in params:
            state.initial_codec_chunk_frames = resolve_initial_codec_chunk_frames(
                params, steady_chunk_frames=self._stream_stride
            )
        elif state.initial_codec_chunk_frames is None:
            state.initial_codec_chunk_frames = 0

    # --- decode internals -------------------------------------------------------

    def _decode_delta(
        self, request_id: str, state: CsmStreamState, *, final: bool = False
    ) -> list[OutgoingMessage]:
        """Stateless overlap-trim incremental decode:
        when enough new frames are available per stride/first-chunk knobs,
        re-decode ``rows[emitted - overlap : available - holdback]`` through
        :func:`sanitize_for_mimi` + ``codec.decode``, trim ``overlap * 1920``
        leading samples, emit exactly ``new_frames * 1920``.

        Returns:
            Audio-chunk messages (possibly empty when below stride).
        """
        available = len(state.rows)
        if available == 0:
            return []

        if final:
            # Fence #3: never decode past the first all-32-zero frame.
            rows_F32 = torch.stack(state.rows, dim=0)
            emit_until = int(trim_at_first_all_zero_frame(rows_F32).shape[0])
        else:
            initial = state.initial_codec_chunk_frames or 0
            has_emitted = state.frames_emitted > 0
            if not has_emitted:
                use_initial = 0 < initial < self._stream_stride
                need = initial if use_initial else self._stream_stride
                if available < need:
                    return []
                emit_until = (
                    initial if use_initial else available - self._stream_holdback_frames
                )
            else:
                need = (
                    state.frames_emitted
                    + self._stream_followup_stride
                    + self._stream_holdback_frames
                )
                if available < need:
                    return []
                emit_until = available - self._stream_holdback_frames
        if emit_until <= state.frames_emitted:
            return []

        window_start = max(0, state.frames_emitted - self._stream_overlap_frames)
        codes = torch.stack(state.rows[window_start:emit_until], dim=0)  # fresh copy
        codes, num_clamped = sanitize_for_mimi(codes)
        if num_clamped:
            state.clamp_count += num_clamped
            self._clamp_count_total += num_clamped
            logger.warning(
                "CSM vocoder stream %s clamped %d OOB codes (fence #2 should "
                "never fire at temp=0)",
                request_id,
                num_clamped,
            )
        audio = self._codec.decode(codes).reshape(-1)  # [decoded_frames * 1920]

        trim_samples = (state.frames_emitted - window_start) * self._samples_per_frame
        if final:
            delta = audio[trim_samples:]
        else:
            new_frames = emit_until - state.frames_emitted
            delta = audio[
                trim_samples : trim_samples + new_frames * self._samples_per_frame
            ]
        delta = delta.contiguous()
        if delta.numel() == 0:
            return []

        state.frames_emitted = emit_until
        state.samples_emitted += int(delta.numel())
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=self._audio_payload(delta, source_hint="CSM TTS streaming"),
                metadata={"modality": "audio"},
            )
        ]

    def _audio_payload_from_stage_payload(
        self, request_id: str, state: CsmStreamState, payload: StagePayload
    ) -> dict[str, Any] | None:
        """Full-payload decode fallback for streams that never emitted —
        exact HF trim semantics over ``state.output_frames`` (incl. the EOS
        frame)."""
        if not isinstance(payload.data, dict):
            return None
        codes = self._frames_to_codes(CsmTtsState.from_dict(payload.data).output_frames)
        if codes is None:
            return None
        codes, num_clamped = sanitize_for_mimi(codes)
        if num_clamped:
            state.clamp_count += num_clamped
            self._clamp_count_total += num_clamped
            logger.warning(
                "CSM vocoder fallback decode for %s clamped %d OOB codes",
                request_id,
                num_clamped,
            )
        return self._audio_payload(
            self._codec.decode(codes), source_hint="CSM TTS streaming"
        )

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        """Non-streaming single decode (delegates to :meth:`_vocode_payloads`)."""
        return self._vocode_payloads([payload])[0]

    def _vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        """Non-streaming batch decode over ``state.output_frames`` via
        ``codec.decode_batch`` (exact-frame-count buckets), trimming at the
        first all-32-zero frame (exact HF trim parity — the EOS frame IS in
        ``output_frames``)."""
        items: list[tuple[CsmTtsState, torch.Tensor | None]] = []
        for payload in payloads:
            state = CsmTtsState.from_dict(payload.data)
            codes = self._frames_to_codes(state.output_frames)
            if codes is not None:
                codes, num_clamped = sanitize_for_mimi(codes)
                if num_clamped:
                    self._clamp_count_total += num_clamped
                    logger.warning(
                        "CSM vocoder batch decode for %s clamped %d OOB codes",
                        payload.request_id,
                        num_clamped,
                    )
            items.append((state, codes))

        valid = [(i, codes) for i, (_, codes) in enumerate(items) if codes is not None]
        waveforms: list[torch.Tensor | None] = [None] * len(items)
        if valid:
            indices, codes_list = zip(*valid)
            wavs = self._codec.decode_batch(list(codes_list))
            if len(wavs) != len(valid):
                raise RuntimeError(
                    f"CSM vocoder decode_batch returned {len(wavs)} audios "
                    f"for {len(valid)} requests"
                )
            for idx, wav in zip(indices, wavs):
                waveforms[idx] = wav

        results: list[StagePayload] = []
        for payload, (state, _), waveform in zip(payloads, items, waveforms):
            data = self._audio_payload(
                waveform if waveform is not None else [],
                source_hint="CSM TTS vocoder",
            )
            usage = self._build_usage(state)
            if usage is not None:
                data["usage"] = usage
            payload.data = data
            results.append(payload)
        return results

    @staticmethod
    def _frames_to_codes(frames: list[Any] | None) -> torch.Tensor | None:
        """``output_frames`` rows → trimmed int64 ``[F, 32]`` (or None when
        nothing decodable remains — e.g. a frame-0 EOS with cb31 == 0)."""
        if not frames:
            return None
        rows = [torch.as_tensor(row, dtype=torch.long).reshape(-1) for row in frames]
        codes = torch.stack(rows, dim=0)
        codes = trim_at_first_all_zero_frame(codes)
        if codes.shape[0] == 0:
            return None
        return codes

    def _audio_payload(self, audio: Any, *, source_hint: str) -> dict[str, Any]:
        from sglang_omni.utils.audio_payload import audio_waveform_payload

        return audio_waveform_payload(
            audio,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint=source_hint,
        )

    @staticmethod
    def _build_usage(state: CsmTtsState) -> dict[str, Any] | None:
        """Usage dict for the terminal result (prompt_tokens,
        completion_frames, engine_time_s)."""
        if not (state.prompt_tokens or state.completion_frames or state.engine_time_s):
            return None
        usage: dict[str, Any] = {
            "prompt_tokens": state.prompt_tokens,
            "completion_tokens": state.completion_frames,  # 1 frame = 1 token
            "completion_frames": state.completion_frames,
            "total_tokens": state.prompt_tokens + state.completion_frames,
        }
        if state.engine_time_s:
            usage["engine_time_s"] = round(state.engine_time_s, 6)
        return usage
