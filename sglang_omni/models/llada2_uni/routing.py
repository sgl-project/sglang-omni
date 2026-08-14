# SPDX-License-Identifier: Apache-2.0
"""Task-aware terminal routing for LLaDA2-Uni."""

from __future__ import annotations

import copy
from typing import Any

from sglang_omni.models.llada2_uni.config import (
    DECODE_STAGE,
    IMAGE_DECODE_STAGE,
    INTERLEAVED_COLLECT_STAGE,
    THINKER_STAGE,
)
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState


def project_interleaved_payload(payload: Any) -> Any:
    """Isolate mutable model state for each asynchronous fan-out branch."""
    return copy.deepcopy(payload)


def resolve_terminal_stages(request: Any) -> list[str]:
    """Select the one terminal reached by task-aware thinker routing."""
    metadata = getattr(request, "metadata", None)
    if isinstance(metadata, dict):
        image_generation = metadata.get("image_generation")
        if (
            isinstance(image_generation, dict)
            and image_generation.get("mode") == "interleaved"
        ):
            return [INTERLEAVED_COLLECT_STAGE]
        modalities = metadata.get("output_modalities")
        if isinstance(modalities, str):
            modalities = (modalities,)
        if isinstance(modalities, (list, tuple, set)) and "image" in {
            str(modality).lower() for modality in modalities
        }:
            return [IMAGE_DECODE_STAGE]
    return [DECODE_STAGE]


def thinker_next(request_id: str, output: Any) -> str | list[str]:
    data = output.data if hasattr(output, "data") else output
    state = LLaDA2UniPipelineState.from_dict(data)
    interleaved = state.generation_state.get("interleaved")
    if state.task_kind == "interleaved" and isinstance(interleaved, dict):
        emit_frame = bool(interleaved.get("emit_frame"))
        done = bool(interleaved.get("done"))
        if emit_frame and done:
            return [IMAGE_DECODE_STAGE, INTERLEAVED_COLLECT_STAGE]
        if emit_frame:
            return [IMAGE_DECODE_STAGE, THINKER_STAGE]
        if interleaved.get("needs_reentry"):
            return THINKER_STAGE
        if done:
            return INTERLEAVED_COLLECT_STAGE
        raise ValueError(f"interleaved request {request_id} has no valid route")
    thinking = state.generation_state.get("thinking")
    if isinstance(thinking, dict) and thinking.get("needs_reentry"):
        return THINKER_STAGE
    if state.task_kind in {"t2i", "edit"}:
        return IMAGE_DECODE_STAGE
    return DECODE_STAGE
