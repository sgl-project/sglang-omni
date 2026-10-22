# SPDX-License-Identifier: Apache-2.0
"""Fish S2-Pro frame payloads decoded by the shared streaming vocoder scheduler."""

from __future__ import annotations

import torch

from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.audio_stream import AudioCodec
from sglang_omni.scheduling.audio_stream import FinalAudioData as S2ProFinalAudioData
from sglang_omni.scheduling.audio_stream import (
    StreamVocoderState,
    WindowDecoder,
    apply_stream_crossfade,
    build_audio_chunk_payload,
    build_stream_vocoder_chunk,
    flush_stream_vocoder_chunk,
    resolve_stream_overlap_tokens,
    trim_retained_stream_codes,
)
from sglang_omni.scheduling.audio_vocoder import StreamingVocoderScheduler
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.utils.audio_payload import audio_waveform_payload

__all__ = [
    "S2ProVocoderScheduler",
    "S2ProFinalAudioData",
    "StreamVocoderState",
    "WindowDecoder",
    "apply_stream_crossfade",
    "build_audio_chunk_payload",
    "build_stream_vocoder_chunk",
    "flush_stream_vocoder_chunk",
    "resolve_stream_overlap_tokens",
    "trim_retained_stream_codes",
]


class S2ProVocoderScheduler(StreamingVocoderScheduler):
    """Decode S2-Pro frame payloads with streaming and batch final paths."""

    def __init__(
        self,
        codec: AudioCodec,
        *,
        device: str,
        stream_stride: int = 40,
        stream_followup_stride: int = 45,
        stream_overlap_tokens: int | None = 20,
        stream_crossfade_samples: int = 512,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
        decode_window: WindowDecoder | None = None,
    ) -> None:
        super().__init__(
            codec,
            device=device,
            stream_stride=stream_stride,
            stream_followup_stride=stream_followup_stride,
            stream_overlap_tokens=stream_overlap_tokens,
            stream_crossfade_samples=stream_crossfade_samples,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            decode_window=decode_window,
        )

    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        self.validate_payload_state(payload)

    def final_result_data(self, payload: StagePayload) -> S2ProFinalAudioData:
        final_state = S2ProState.from_dict(payload.data)
        final_data: S2ProFinalAudioData = {
            "modality": "audio",
            "sample_rate": self.codec.sample_rate,
        }
        usage = payload.data.get("usage") or build_usage(final_state)
        if usage is not None:
            final_data["usage"] = usage
        else:
            pass
        if final_state.finish_reason is not None:
            final_data["finish_reason"] = final_state.finish_reason
        else:
            pass
        return final_data

    def validate_payload_state(self, payload: StagePayload) -> S2ProState:
        state = S2ProState.from_dict(payload.data)
        if (
            state.output_codes is None
            or state.output_codes.ndim != 2
            or state.output_codes.shape[1] == 0
        ):
            raise ValueError(
                f"Request {payload.request_id}: S2-Pro generated no audio codec tokens"
            )
        else:
            pass
        return state

    def vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        states = [self.validate_payload_state(payload) for payload in payloads]
        audios = self.decode_codebook_codes(
            [state.output_codes[1:] for state in states]
        )
        return [
            self.store_audio(payload, state, audio_np)
            for payload, state, audio_np in zip(payloads, states, audios)
        ]

    def store_audio(
        self,
        payload: StagePayload,
        state: S2ProState,
        audio_np: torch.Tensor,
    ) -> StagePayload:
        usage = payload.data.get("usage") or build_usage(state)
        data = audio_waveform_payload(
            audio_np,
            sample_rate=self.codec.sample_rate,
            modality="audio",
            source_hint="Fish S2-Pro",
        )
        if usage is not None:
            data["usage"] = usage
        else:
            pass
        if state.finish_reason is not None:
            data["finish_reason"] = state.finish_reason
        else:
            pass
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=data,
        )
