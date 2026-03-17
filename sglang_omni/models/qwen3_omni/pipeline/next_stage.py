# SPDX-License-Identifier: Apache-2.0
"""Stage routing helpers for Qwen3-Omni pipelines."""

from __future__ import annotations

from typing import Any, Callable

from sglang_omni.models.qwen3_omni.io import PipelineState
from sglang_omni.proto import StagePayload

PREPROCESSING_STAGE = "preprocessing"
IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"
AGGREGATE_STAGE = "mm_aggregate"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"
TALKER_AR_STAGE = "talker_ar"
CODE_PREDICTOR_STAGE = "code_predictor"
CODE2WAV_STAGE = "code2wav"


def preprocessing_next(request_id: str, output: Any) -> list[str]:
    del request_id
    if not isinstance(output, StagePayload):
        return [AGGREGATE_STAGE]
    state = PipelineState.from_dict(output.data)
    encoder_inputs = state.encoder_inputs
    if not isinstance(encoder_inputs, dict):
        return [AGGREGATE_STAGE]
    stages = [stage for stage in encoder_inputs.keys() if stage != AGGREGATE_STAGE]
    stages = sorted(stages)
    stages.append(AGGREGATE_STAGE)
    return stages


def encoder_next(request_id: str, output: Any) -> str:
    del request_id, output
    return AGGREGATE_STAGE


def aggregate_next(request_id: str, output: Any) -> str:
    del request_id, output
    return THINKER_STAGE


def thinker_next(request_id: str, output: Any) -> str:
    del request_id, output
    return DECODE_STAGE


def make_thinker_next(
    speech_enabled: bool = False,
) -> Callable[[str, Any], str | list[str]]:
    """Create a thinker_next function with optional speech fan-out.

    When speech_enabled=True, returns [DECODE_STAGE, TALKER_AR_STAGE] (fan-out).
    When speech_enabled=False, returns DECODE_STAGE (existing behavior).
    """

    def _thinker_next(request_id: str, output: Any) -> str | list[str]:
        del request_id, output
        if speech_enabled:
            return [DECODE_STAGE, TALKER_AR_STAGE]
        return DECODE_STAGE

    return _thinker_next


def thinker_next_speech(request_id: str, output: Any) -> list[str]:
    """Thinker fan-out to both Decode and Talker AR (speech pipeline)."""
    del request_id, output
    return [DECODE_STAGE, TALKER_AR_STAGE]


def talker_ar_next(request_id: str, output: Any) -> str:
    del request_id, output
    return CODE_PREDICTOR_STAGE


def code_predictor_next(request_id: str, output: Any) -> str:
    del request_id, output
    return CODE2WAV_STAGE


def code2wav_next(request_id: str, output: Any) -> None:
    del request_id, output
    return None


def decode_next(request_id: str, output: Any) -> None:
    del request_id, output
    return None
