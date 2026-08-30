# SPDX-License-Identifier: Apache-2.0
"""Stage factories for NVIDIA NemotronLabs VoiceChat 11B."""

from __future__ import annotations

import time
from typing import Any

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.device import place_device_spec

from .payload_types import OUTPUT_SAMPLE_RATE, VoiceChatFrameState


def _output(payload: StagePayload, state: VoiceChatFrameState) -> StagePayload:
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def create_perception_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "float32",
    use_cudagraph: bool = True,
    max_sessions: int = 1,
) -> SimpleScheduler:
    if dtype != "float32":
        raise ValueError("VoiceChat perception requires dtype='float32'")
    from .audio_runtime import VoiceChatPerceptionRuntime

    runtime = VoiceChatPerceptionRuntime(
        model_path,
        device=place_device_spec(device, gpu_id),
        use_cudagraph=use_cudagraph,
        max_sessions=max_sessions,
    )

    def compute(payload: StagePayload) -> StagePayload:
        state = VoiceChatFrameState.from_data(payload.data)
        if state.event == "session_close":
            runtime.close(state.session_id)
            return _output(payload, state)
        if not isinstance(state.pcm16, str):
            raise TypeError("VoiceChat perception requires base64 PCM16 data")
        started = time.perf_counter()
        state.acoustic_embedding = runtime.step(
            state.session_id, int(state.frame_index), state.pcm16
        )
        state.timings_ms["perception"] = (time.perf_counter() - started) * 1000
        # Raw PCM is no longer needed after perception and would otherwise be
        # copied through every downstream process.
        state.pcm16 = None
        return _output(payload, state)

    return SimpleScheduler(compute)


def create_thinker_executor(model_path: str, **kwargs: Any) -> Any:
    from .engine_builder import VoiceChatThinkerEngineBuilder

    context_length = int(kwargs.pop("context_length", 8192))
    max_sessions = int(kwargs.pop("max_sessions", 1))
    total_gpu_memory_fraction = kwargs.pop("total_gpu_memory_fraction", None)
    duplex_model_path = kwargs.pop("duplex_model_path", None)
    return VoiceChatThinkerEngineBuilder(
        context_length=context_length,
        max_sessions=max_sessions,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        duplex_model_path=duplex_model_path,
    ).build(model_path, **kwargs)


def create_talker_executor(model_path: str, **kwargs: Any) -> Any:
    from .engine_builder import VoiceChatTalkerEngineBuilder

    context_length = int(kwargs.pop("context_length", 8192))
    max_sessions = int(kwargs.pop("max_sessions", 1))
    total_gpu_memory_fraction = kwargs.pop("total_gpu_memory_fraction", None)
    eartts_model_path = kwargs.pop("eartts_model_path", None)
    speaker_latent_path = kwargs.pop("speaker_latent_path", None)
    return VoiceChatTalkerEngineBuilder(
        context_length=context_length,
        max_sessions=max_sessions,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        eartts_model_path=eartts_model_path,
        speaker_latent_path=speaker_latent_path,
    ).build(model_path, **kwargs)


def create_code2wav_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "float32",
    max_sessions: int = 1,
) -> SimpleScheduler:
    if dtype != "float32":
        raise ValueError("VoiceChat code2wav requires dtype='float32'")
    from .audio_runtime import VoiceChatCodecRuntime

    runtime = VoiceChatCodecRuntime(
        model_path,
        device=place_device_spec(device, gpu_id),
        max_sessions=max_sessions,
    )

    def compute(payload: StagePayload) -> StagePayload:
        state = VoiceChatFrameState.from_data(payload.data)
        if state.event == "session_close":
            runtime.close(state.session_id)
            return StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data={"closed": True, "session_id": state.session_id},
            )
        if state.audio_codes is None:
            raise ValueError("VoiceChat code2wav requires audio_codes")
        started = time.perf_counter()
        audio = runtime.step(
            state.session_id, int(state.frame_index), state.audio_codes
        )
        state.timings_ms["code2wav"] = (time.perf_counter() - started) * 1000
        audio_payload = audio_waveform_payload(
            audio,
            sample_rate=OUTPUT_SAMPLE_RATE,
            modality="audio",
            source_hint="Nemotron VoiceChat code2wav",
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                **audio_payload,
                "text": state.text_delta or "",
                "token_ids": [state.text_token] if state.text_token is not None else [],
                "omni_rollout": {
                    "frame_index": state.frame_index,
                    "text_token": state.text_token,
                    "function_token": state.function_token,
                    "timings_ms": state.timings_ms,
                },
            },
        )

    return SimpleScheduler(compute)


__all__ = [
    "create_code2wav_executor",
    "create_perception_executor",
    "create_talker_executor",
    "create_thinker_executor",
]
