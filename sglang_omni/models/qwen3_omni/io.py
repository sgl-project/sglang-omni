# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni payload schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict


class PromptInputs(TypedDict):
    """Tokenized prompt inputs for the thinker."""

    input_ids: Any
    attention_mask: Any
    prompt_text: str


class FrontendData(TypedDict, total=False):
    """Frontend outputs stored on StagePayload.data."""

    raw_inputs: Any
    prompt: PromptInputs
    mm_inputs: dict[str, Any]
    encoder_inputs: dict[str, dict[str, Any]]
    stream_state: dict[str, Any]


class ThinkerOutput(TypedDict, total=False):
    """Normalized thinker output used for decoding and streaming."""

    output_ids: list[int]
    step: int
    is_final: bool
    extra_model_outputs: dict[str, Any]


OmniEventType = Literal[
    "text_delta",
    "text_final",
    "audio_chunk",
    "audio_final",
    "image",
    "video_chunk",
    "video_final",
    "debug",
    "final",
]


@dataclass
class OmniEvent:
    """Streaming-friendly event emitted by decode logic."""

    type: OmniEventType
    modality: str
    payload: dict[str, Any]
    is_final: bool = False
