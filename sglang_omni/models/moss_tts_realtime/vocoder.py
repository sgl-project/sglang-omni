# SPDX-License-Identifier: Apache-2.0
"""Streaming MOSS-Audio-Tokenizer decoder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.moss_tts_realtime.payload_types import (
    N_CODEBOOKS,
    SAMPLE_RATE,
    MossTTSRealtimeState,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import (
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload


@dataclass
class _StreamState:
    rows: list[torch.Tensor] = field(default_factory=list)
    initial_chunk_frames: int = 1
    threshold: int = 1
    context: Any = None
    latched: bool = False


class MossTTSRealtimeVocoder(StreamingVocoderBase[_StreamState, None]):
    """Decode one causal codec stream while retaining decoder KV state."""

    def __init__(
        self,
        codec: Any,
        *,
        stream_chunk_frames: int = 6,
        initial_chunk_frames: int = 1,
    ) -> None:
        self._codec = codec
        self._device = next(codec.parameters()).device
        self._stream_chunk_frames = int(stream_chunk_frames)
        self._default_initial_chunk_frames = int(initial_chunk_frames)
        self._active_request_id: str | None = None
        super().__init__(
            self._vocode,
            sample_rate=SAMPLE_RATE,
            stream_source_hint="MOSS-TTS-Realtime",
            max_batch_size=1,
        )

    def create_stream_state(self, request_id: str) -> _StreamState:
        if (
            self._active_request_id is not None
            and self._active_request_id != request_id
        ):
            raise RuntimeError(
                "MOSS-TTS-Realtime codec supports one active stream at a time"
            )
        context = self._codec.streaming(batch_size=1)
        context.__enter__()
        self._active_request_id = request_id
        return _StreamState(
            initial_chunk_frames=self._default_initial_chunk_frames,
            threshold=self._default_initial_chunk_frames,
            context=context,
        )

    def latch_stream_contract(
        self,
        request_id: str,
        state: _StreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        del request_id
        if state.latched:
            return
        params = source.request.params if origin == "payload" else source
        if isinstance(params, Mapping):
            state.initial_chunk_frames = (
                resolve_initial_codec_chunk_frames(
                    params,
                    steady_chunk_frames=self._stream_chunk_frames,
                )
                or self._default_initial_chunk_frames
            )
            state.threshold = state.initial_chunk_frames
        state.latched = True

    def validate_chunk(
        self, request_id: str, state: _StreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id, state
        row = codes.to(dtype=torch.long, device="cpu").reshape(-1)
        if int(row.numel()) < N_CODEBOOKS + 1:
            raise ValueError(
                "MOSS-TTS-Realtime stream rows require text plus 16 codebooks"
            )
        return row[1 : N_CODEBOOKS + 1]

    def ingest(self, request_id: str, state: _StreamState, codes: torch.Tensor) -> None:
        del request_id
        state.rows.append(codes)

    def should_decode(self, state: _StreamState, *, is_final: bool) -> bool:
        return is_final or len(state.rows) >= state.threshold

    def decode_delta(
        self, request_id: str, state: _StreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        del request_id
        if not state.rows:
            return None
        count = len(state.rows) if is_final else state.threshold
        if len(state.rows) < count:
            return None
        rows = torch.stack(state.rows[:count], dim=0)
        del state.rows[:count]
        codes = rows.transpose(0, 1).contiguous().to(self._device)
        decoded = self._codec.decode(codes, return_dict=True)
        waveform = getattr(decoded, "audio", None)
        if waveform is None and isinstance(decoded, dict):
            waveform = decoded.get("audio")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("MOSS audio tokenizer returned no waveform")
        state.threshold = self._stream_chunk_frames
        return waveform[0].detach().to(device="cpu", dtype=torch.float32).contiguous()

    def release_stream_resources(self, request_id: str, state: _StreamState) -> None:
        if state.context is not None:
            state.context.__exit__(None, None, None)
            state.context = None
        if self._active_request_id == request_id:
            self._active_request_id = None

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: _StreamState
    ) -> torch.Tensor | None:
        del request_id, state
        realtime_state = MossTTSRealtimeState.from_dict(payload.data)
        return self._decode_codes(realtime_state.audio_codes)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: _StreamState
    ) -> dict[str, Any]:
        del request_id, state
        realtime_state = MossTTSRealtimeState.from_dict(payload.data)
        data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": SAMPLE_RATE,
        }
        usage = build_usage(realtime_state)
        if usage is not None:
            data["usage"] = usage
        return data

    def _decode_codes(self, raw_codes: Any) -> torch.Tensor | None:
        if raw_codes is None:
            return None
        codes = torch.as_tensor(raw_codes, dtype=torch.long)
        if codes.numel() == 0:
            return None
        decoded = self._codec.decode(
            codes[:, :N_CODEBOOKS].transpose(0, 1).contiguous().to(self._device),
            return_dict=True,
            chunk_duration=8,
        )
        waveform = getattr(decoded, "audio", None)
        if waveform is None and isinstance(decoded, dict):
            waveform = decoded.get("audio")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("MOSS audio tokenizer returned no waveform")
        return waveform[0].detach().to(device="cpu", dtype=torch.float32).contiguous()

    def _vocode(self, payload: StagePayload) -> StagePayload:
        state = MossTTSRealtimeState.from_dict(payload.data)
        waveform = self._decode_codes(state.audio_codes)
        state.audio_codes = None
        payload.data = state.to_dict()
        if waveform is None:
            return payload
        payload.data.update(
            audio_waveform_payload(
                waveform,
                sample_rate=SAMPLE_RATE,
                modality="audio",
                source_hint="MOSS-TTS-Realtime",
            )
        )
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload


__all__ = ["MossTTSRealtimeVocoder"]
