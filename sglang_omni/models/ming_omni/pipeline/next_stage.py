# SPDX-License-Identifier: Apache-2.0
"""Stage routing helpers for Ming-Omni pipelines."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.ming_omni.io import PipelineState
from sglang_omni.proto import StagePayload

PREPROCESSING_STAGE = "preprocessing"
AUDIO_STAGE = "audio_encoder"
IMAGE_STAGE = "image_encoder"
AGGREGATE_STAGE = "mm_aggregate"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"
TALKER_STAGE = "talker"


def preprocessing_next(request_id: str, output: Any) -> list[str]:
    """Route from preprocessing to encoder stages then aggregate."""
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
    """Audio encoder always routes to aggregate."""
    del request_id, output
    return AGGREGATE_STAGE


def aggregate_next(request_id: str, output: Any) -> str:
    """Aggregate always routes to thinker."""
    del request_id, output
    return THINKER_STAGE


def thinker_next(request_id: str, output: Any) -> str:
    """Text-only: thinker routes to decode."""
    del request_id, output
    return DECODE_STAGE


def thinker_next_speech(request_id: str, output: Any) -> list[str]:
    """Speech pipeline: thinker fan-out to decode and talker."""
    del request_id, output
    return [DECODE_STAGE, TALKER_STAGE]


def decode_next(request_id: str, output: Any) -> None:
    """Decode is terminal."""
    del request_id, output
    return None


def talker_next(request_id: str, output: Any) -> None:
    """Talker is terminal."""
    del request_id, output
    return None
