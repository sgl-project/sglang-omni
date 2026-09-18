# SPDX-License-Identifier: Apache-2.0
"""Routing for the generation pipeline's optional thinking pass."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.llada2_uni.config import (
    DECODE_STAGE,
    IMAGE_DECODE_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState


def thinker_next(request_id: str, output: Any) -> str | list[str]:
    state = LLaDA2UniPipelineState.from_dict(getattr(output, "data", None))
    if state.stream_state.get("thinking_needs_reentry"):
        return THINKER_STAGE
    return [DECODE_STAGE, IMAGE_DECODE_STAGE]
