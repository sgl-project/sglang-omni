# SPDX-License-Identifier: Apache-2.0
"""Task-aware terminal routing for LLaDA2-Uni."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.llada2_uni.config import DECODE_STAGE, IMAGE_DECODE_STAGE
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState


def resolve_terminal_stages(request: Any) -> list[str]:
    """Select the one terminal reached by task-aware thinker routing."""
    metadata = getattr(request, "metadata", None)
    if isinstance(metadata, dict):
        modalities = metadata.get("output_modalities")
        if isinstance(modalities, str):
            modalities = (modalities,)
        if isinstance(modalities, (list, tuple, set)) and "image" in {
            str(modality).lower() for modality in modalities
        }:
            return [IMAGE_DECODE_STAGE]
    return [DECODE_STAGE]


def thinker_next(request_id: str, output: Any) -> str:
    del request_id
    data = output.data if hasattr(output, "data") else output
    state = LLaDA2UniPipelineState.from_dict(data)
    if state.task_kind in {"t2i", "edit"}:
        return IMAGE_DECODE_STAGE
    return DECODE_STAGE
