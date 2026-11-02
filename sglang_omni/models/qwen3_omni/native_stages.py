# SPDX-License-Identifier: Apache-2.0
"""Qwen model factories using the shared persistent stage scheduler."""

from typing import Literal

from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.models.qwen3_omni.merge import build_thinker_inputs
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.session_input import QwenInputHooks
from sglang_omni.models.qwen3_omni.session_preprocessing import QwenPreprocessingHooks
from sglang_omni.models.qwen3_omni.stages import run_single_encoder_payload
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import (
    DEFAULT_MAX_CONCURRENCY,
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from sglang_omni.utils.cpu import automatic_video_decode_threads
from sglang_omni.utils.device import resolve_concrete_device


class QwenEncoderHooks(SessionHooks):
    def __init__(
        self,
        model: Qwen3OmniAudioEncoder | Qwen3OmniImageEncoder,
        stage_name: Literal["image_encoder", "audio_encoder"],
    ) -> None:
        self.model: Qwen3OmniAudioEncoder | Qwen3OmniImageEncoder = model
        self.stage_name: Literal["image_encoder", "audio_encoder"] = stage_name

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        pass

    def close(self, session_identity: SessionIdentity) -> None:
        pass

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if context.cancelled.is_set():
            return payload
        else:
            session = payload.data["session"]
        # Note (Junnan Li): A session never encodes the same media twice, so no output cache.
        payload = run_single_encoder_payload(
            payload, stage_name=self.stage_name, model=self.model
        )
        if self.stage_name == "audio_encoder":
            state = Qwen3OmniPipelineState.from_dict(payload.data)
            state.thinker_inputs = build_thinker_inputs(state, state.encoder_outs)
            payload.data = state.to_dict()
        else:
            pass
        payload.data["session"] = session
        return payload


def create_input_scheduler(
    *,
    max_open_sessions: int,
    max_utterance_seconds: float,
    max_queued_turns: int,
    max_utterance_frames: int,
) -> SessionScheduler:
    return SessionScheduler(
        QwenInputHooks(
            max_utterance_seconds=max_utterance_seconds,
            max_queued_turns=max_queued_turns,
            max_utterance_frames=max_utterance_frames,
        ),
        max_open_sessions=max_open_sessions,
    )


def create_preprocessing_scheduler(
    model_path: str,
    *,
    max_open_sessions: int,
    max_seq_len: int,
) -> SessionScheduler:
    return SessionScheduler(
        QwenPreprocessingHooks(
            Qwen3OmniPreprocessor(
                model_path,
                max_seq_len=max_seq_len,
                video_decode_threads=automatic_video_decode_threads(
                    DEFAULT_MAX_CONCURRENCY
                ),
            )
        ),
        max_open_sessions=max_open_sessions,
    )


def create_encoder_scheduler(
    model_path: str,
    *,
    stage_name: Literal["image_encoder", "audio_encoder"],
    max_open_sessions: int,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | None = None,
    enable_layer_cuda_graph: bool = False,
) -> SessionScheduler:
    resolved_device = str(resolve_concrete_device(device, gpu_id))
    if stage_name == "image_encoder":
        model = Qwen3OmniImageEncoder(
            model_path=model_path, device=resolved_device, dtype=dtype
        )
    else:
        model = Qwen3OmniAudioEncoder(
            model_path=model_path,
            device=resolved_device,
            dtype=dtype,
            enable_layer_cuda_graph=enable_layer_cuda_graph,
        )
    # note (db-ol): The audio encoder is not thread safe, so overlapping units could return another session's embeddings.
    return SessionScheduler(
        QwenEncoderHooks(model, stage_name),
        max_open_sessions=max_open_sessions,
        max_concurrency=1,
    )
