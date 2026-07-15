# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Higgs TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.codec_delay import reverse_delay_pattern


@dataclass
class _HiggsStreamState:
    delayed_rows: list[torch.Tensor] = field(default_factory=list)
    emitted_raw_frames: int = 0
    next_decode_rows: int = 0
    num_codebooks: int | None = None
    codebook_size: int | None = None
    initial_codec_chunk_frames: int = 0


class HiggsStreamingVocoderScheduler(StreamingVocoderBase[_HiggsStreamState, None]):
    """Decode Higgs codec rows incrementally, with batched final decode."""

    def __init__(
        self,
        codec: HiggsAudioCodec,
        *,
        stream_stride: int = 75,
        stream_followup_stride: int = 75,
        stream_overlap_tokens: int = 8,
        stream_holdback_tokens: int = 4,
        max_batch_size: int = 4,
        max_batch_wait_ms: int = 2,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream_stride and stream_followup_stride must be > 0")
        if stream_overlap_tokens < 0:
            raise ValueError("stream_overlap_tokens must be >= 0")
        if stream_holdback_tokens < 0:
            raise ValueError("stream_holdback_tokens must be >= 0")

        self._codec = codec
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_overlap_tokens = int(stream_overlap_tokens)
        self._stream_holdback_tokens = int(stream_holdback_tokens)
        self._samples_per_frame = self._resolve_samples_per_frame(codec)

        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            sample_rate=HiggsAudioCodec.SAMPLE_RATE,
            stream_source_hint="Higgs",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def create_stream_state(self, request_id: str) -> _HiggsStreamState:
        del request_id
        return _HiggsStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: _HiggsStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            payload = source
            if not isinstance(payload.data, dict):
                raise TypeError(
                    f"Higgs streaming payload for {request_id!r} must be a dict, "
                    f"got {type(payload.data).__name__}"
                )
            missing = [
                key
                for key in ("num_codebooks", "codebook_size")
                if key not in payload.data
            ]
            if missing and (state.num_codebooks is None or state.codebook_size is None):
                raise RuntimeError(
                    f"Higgs streaming payload for {request_id!r} is missing fields: "
                    f"{', '.join(missing)}"
                )
            self._latch_contract_values(
                request_id,
                state,
                num_codebooks=payload.data.get("num_codebooks", state.num_codebooks),
                codebook_size=payload.data.get("codebook_size", state.codebook_size),
                source=origin,
            )
            self._latch_initial_codec_chunk_frames_from_mapping(
                request_id,
                state,
                (
                    payload.request.params
                    if isinstance(payload.request.params, dict)
                    else None
                ),
            )
            return
        metadata: Mapping[str, Any] = source
        missing = [
            key for key in ("num_codebooks", "codebook_size") if key not in metadata
        ]
        if missing and (state.num_codebooks is None or state.codebook_size is None):
            raise RuntimeError(
                f"Higgs stream chunk for {request_id!r} is missing metadata fields: "
                f"{', '.join(missing)}"
            )
        if "num_codebooks" in metadata and "codebook_size" in metadata:
            self._latch_contract_values(
                request_id,
                state,
                num_codebooks=metadata["num_codebooks"],
                codebook_size=metadata["codebook_size"],
                source=origin,
            )
        if INITIAL_CODEC_CHUNK_FRAMES_PARAM in metadata:
            self._latch_initial_codec_chunk_frames_from_mapping(
                request_id,
                state,
                metadata,
            )

    def validate_chunk(
        self, request_id: str, state: _HiggsStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        row = codes.to(dtype=torch.long)
        if row.ndim != 1:
            raise ValueError(
                f"Higgs stream chunk must be 1-D [N], got {tuple(row.shape)}"
            )
        num_codebooks = self._require_stream_contract(state, request_id)[0]
        if int(row.shape[0]) != num_codebooks:
            raise ValueError(
                f"Higgs stream chunk has {int(row.shape[0])} codebooks, "
                f"expected {num_codebooks}"
            )
        return row

    def ingest(
        self, request_id: str, state: _HiggsStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        state.delayed_rows.append(codes)

    def decode_delta(
        self, request_id: str, state: _HiggsStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        delayed_count = len(state.delayed_rows)
        if delayed_count == 0:
            return None
        num_codebooks, codebook_size = self._require_stream_contract(state, "<stream>")
        if delayed_count < num_codebooks:
            return None
        raw_total = delayed_count - num_codebooks + 1

        steady_codec_frames = max(1, self._stream_stride - num_codebooks + 1)
        use_initial_chunk = (
            state.initial_codec_chunk_frames > 0
            and state.initial_codec_chunk_frames < steady_codec_frames
            and not self._stream_has_emitted(request_id)
        )
        first_decode_rows = max(
            num_codebooks,
            state.initial_codec_chunk_frames + num_codebooks - 1,
        )
        next_decode_rows = state.next_decode_rows or (
            first_decode_rows
            if use_initial_chunk and not is_final
            else max(num_codebooks, self._stream_stride)
        )
        if not is_final and delayed_count < next_decode_rows:
            state.next_decode_rows = next_decode_rows
            return None

        emit_until_raw = raw_total
        if use_initial_chunk and not is_final:
            emit_until_raw = min(raw_total, state.initial_codec_chunk_frames)
        elif not is_final and self._stream_holdback_tokens:
            emit_until_raw = max(0, raw_total - self._stream_holdback_tokens)
        can_flush_codec_tail = is_final and self._samples_per_frame is not None
        if emit_until_raw < state.emitted_raw_frames or (
            emit_until_raw == state.emitted_raw_frames and not can_flush_codec_tail
        ):
            state.next_decode_rows = delayed_count + self._stream_followup_stride
            return None

        window_start_raw = max(
            0, state.emitted_raw_frames - self._stream_overlap_tokens
        )
        rows_end = emit_until_raw + num_codebooks - 1
        rows = state.delayed_rows[window_start_raw:rows_end]
        audio = self._decode_delayed_rows(
            rows,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
        )

        decoded_raw_frames = emit_until_raw - window_start_raw
        samples_per_frame = self._samples_per_frame or max(
            int(audio.shape[-1]) // max(decoded_raw_frames, 1), 1
        )
        trim_frames = state.emitted_raw_frames - window_start_raw
        trim_samples = min(int(trim_frames * samples_per_frame), int(audio.shape[-1]))
        if not is_final and self._samples_per_frame is not None:
            new_frames = emit_until_raw - state.emitted_raw_frames
            emit_samples = int(new_frames * samples_per_frame)
            delta = audio[trim_samples : trim_samples + emit_samples].contiguous()
        else:
            delta = audio[trim_samples:].contiguous()
        if delta.numel() == 0:
            state.next_decode_rows = delayed_count + self._stream_followup_stride
            return None

        state.emitted_raw_frames = emit_until_raw
        state.next_decode_rows = self._next_decode_rows_after_emit(
            delayed_count,
            num_codebooks=num_codebooks,
            emitted_initial_chunk=use_initial_chunk and not is_final,
        )
        return delta

    def stream_payload(self, request_id: str, waveform: torch.Tensor) -> dict[str, Any]:
        del request_id
        return audio_waveform_payload(
            waveform,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Higgs TTS streaming",
        )

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: _HiggsStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        return self._decode_state_to_audio(HiggsTtsState.from_dict(payload.data))

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: _HiggsStreamState
    ) -> dict[str, Any]:
        del request_id, state
        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        final_state = HiggsTtsState.from_dict(payload.data)
        usage = build_usage(final_state)
        if usage is not None:
            final_data["usage"] = usage
        if final_state.omni_rollout is not None:
            final_data["omni_rollout"] = final_state.omni_rollout
        return final_data

    @staticmethod
    def _latch_contract_values(
        request_id: str,
        state: _HiggsStreamState,
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
                f"Higgs {source} for {request_id!r} must include integer "
                "num_codebooks and codebook_size"
            ) from exc
        if num_codebooks_i <= 0 or codebook_size_i <= 2:
            raise ValueError(
                f"Higgs {source} for {request_id!r} has invalid "
                f"num_codebooks={num_codebooks_i}, codebook_size={codebook_size_i}"
            )
        if state.num_codebooks is not None and state.num_codebooks != num_codebooks_i:
            raise ValueError(
                f"Higgs stream num_codebooks changed for {request_id!r}: "
                f"{state.num_codebooks} -> {num_codebooks_i}"
            )
        if state.codebook_size is not None and state.codebook_size != codebook_size_i:
            raise ValueError(
                f"Higgs stream codebook_size changed for {request_id!r}: "
                f"{state.codebook_size} -> {codebook_size_i}"
            )
        state.num_codebooks = num_codebooks_i
        state.codebook_size = codebook_size_i

    def _latch_initial_codec_chunk_frames_from_mapping(
        self,
        request_id: str,
        state: _HiggsStreamState,
        params: Mapping[str, Any] | None,
    ) -> None:
        num_codebooks, _ = self._require_stream_contract(state, request_id)
        steady_codec_frames = max(1, self._stream_stride - num_codebooks + 1)
        state.initial_codec_chunk_frames = resolve_initial_codec_chunk_frames(
            params,
            steady_chunk_frames=steady_codec_frames,
        )

    @staticmethod
    def _require_stream_contract(
        state: _HiggsStreamState,
        request_id: str,
    ) -> tuple[int, int]:
        if state.num_codebooks is None or state.codebook_size is None:
            raise RuntimeError(
                f"Higgs stream contract for {request_id!r} is missing "
                "num_codebooks or codebook_size"
            )
        return state.num_codebooks, state.codebook_size

    def _next_decode_rows_after_emit(
        self,
        delayed_count: int,
        *,
        num_codebooks: int,
        emitted_initial_chunk: bool,
    ) -> int:
        if emitted_initial_chunk:
            return (
                max(num_codebooks, self._stream_stride) + self._stream_followup_stride
            )
        return delayed_count + self._stream_followup_stride

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        return self._vocode_payloads([payload])[0]

    def _vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        items = [self._prepare_vocoder_item(payload) for payload in payloads]
        valid = [(i, codes) for i, (_, codes) in enumerate(items) if codes is not None]
        waveforms: list[torch.Tensor | None] = [None] * len(items)
        if valid:
            indices, codes_list = zip(*valid)
            wavs = self._codec.decode_batch(list(codes_list))
            if len(wavs) != len(valid):
                raise RuntimeError(
                    f"Higgs vocoder decode_batch returned {len(wavs)} audios "
                    f"for {len(valid)} requests"
                )
            for idx, wav in zip(indices, wavs):
                waveforms[idx] = wav
        return [
            self._store_vocoder_result(payload, state, wav)
            for payload, (state, _), wav in zip(payloads, items, waveforms)
        ]

    def _prepare_vocoder_item(
        self,
        payload: StagePayload,
    ) -> tuple[HiggsTtsState, torch.Tensor | None]:
        state = HiggsTtsState.from_dict(payload.data)
        delayed_rows = state.output_codes_delayed
        if not delayed_rows:
            return state, None
        delayed_LN = torch.tensor(delayed_rows, dtype=torch.long)
        if delayed_LN.shape[0] < state.num_codebooks:
            return state, None
        codes_TN = reverse_delay_pattern(delayed_LN)
        codec_vocab = int(state.codebook_size) - 2
        return state, torch.where(
            codes_TN >= codec_vocab, torch.zeros_like(codes_TN), codes_TN
        )

    def _store_vocoder_result(
        self,
        payload: StagePayload,
        state: HiggsTtsState,
        waveform: torch.Tensor | None,
    ) -> StagePayload:
        data = audio_waveform_payload(
            waveform if waveform is not None else [],
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Higgs TTS vocoder",
        )
        usage = build_usage(state)
        if usage is not None:
            data["usage"] = usage
        if state.omni_rollout is not None:
            data["omni_rollout"] = state.omni_rollout
        payload.data = data
        return payload

    def _decode_state_to_audio(self, state: HiggsTtsState) -> torch.Tensor | None:
        delayed_rows = state.output_codes_delayed
        if not delayed_rows:
            return None
        rows = [torch.tensor(row, dtype=torch.long) for row in delayed_rows]
        if len(rows) < int(state.num_codebooks):
            return None
        return self._decode_delayed_rows(
            rows,
            num_codebooks=int(state.num_codebooks),
            codebook_size=int(state.codebook_size),
        )

    def _decode_delayed_rows(
        self,
        rows: list[torch.Tensor],
        *,
        num_codebooks: int,
        codebook_size: int,
    ) -> torch.Tensor:
        if len(rows) < int(num_codebooks):
            raise ValueError(
                f"Higgs delayed rows must include at least {num_codebooks} rows, "
                f"got {len(rows)}"
            )
        delayed_LN = torch.stack(rows, dim=0).to(torch.long)
        codes_TN = reverse_delay_pattern(delayed_LN)
        codec_vocab = int(codebook_size) - 2
        codes_TN = torch.where(
            codes_TN >= codec_vocab, torch.zeros_like(codes_TN), codes_TN
        )
        return self._codec.decode(codes_TN).detach().to(torch.float32)

    @staticmethod
    def _resolve_samples_per_frame(codec: HiggsAudioCodec) -> int | None:
        hop_length = getattr(getattr(codec, "model", None), "config", None)
        hop_length = getattr(hop_length, "hop_length", None)
        if hop_length is None:
            return None
        hop_length_i = int(hop_length)
        return hop_length_i if hop_length_i > 0 else None


__all__ = ["HiggsStreamingVocoderScheduler"]
