# SPDX-License-Identifier: Apache-2.0
"""Stage routing helpers for Qwen3-Omni pipelines."""

from __future__ import annotations

from typing import Any

from sglang_omni.proto import StagePayload

FRONTEND_STAGE = "frontend"
IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"
AGGREGATE_STAGE = "mm_aggregate"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"


def frontend_next(request_id: str, output: Any) -> list[str]:
    del request_id
    if not isinstance(output, StagePayload):
        return [AGGREGATE_STAGE]
    data = output.data if isinstance(output.data, dict) else {}
    encoder_inputs = data.get("encoder_inputs")
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


def decode_next(request_id: str, output: Any) -> None:
    del request_id, output
    return None
