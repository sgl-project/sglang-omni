# SPDX-License-Identifier: Apache-2.0
"""Routing logic for Qwen3-ASR pipeline stages."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.qwen3_asr.io import PipelineState
from sglang_omni.proto import StagePayload

PREPROCESSING_STAGE = "preprocess"
AUDIO_STAGE = "audio"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"


def preprocessing_next(request_id: str, payload: Any) -> list[str]:
    """Route from preprocessing to the next stage."""
    del request_id
    if not isinstance(payload, StagePayload):
        return [THINKER_STAGE]

    state = PipelineState.from_dict(payload.data)
    if state.raw_inputs and "audio" in state.raw_inputs:
        return [AUDIO_STAGE]
    return [THINKER_STAGE]


def encoder_next(request_id: str, payload: Any) -> str:
    """Route from encoders to the thinker."""
    del request_id, payload
    return THINKER_STAGE


def thinker_next(request_id: str, payload: Any) -> str:
    """Route from thinker to decode."""
    del request_id, payload
    return DECODE_STAGE


def decode_next(request_id: str, payload: Any) -> None:
    """Route from decode to end."""
    del request_id, payload
    return None
