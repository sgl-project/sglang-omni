# SPDX-License-Identifier: Apache-2.0
"""Shared payload schemas for omni pipelines.

Type aliases kept here so model code can import them without pulling in the
full ``proto`` package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict


class PromptInputs(TypedDict):
    """Tokenized prompt inputs for the thinker."""

    input_ids: Any
    attention_mask: Any
    prompt_text: str


class ThinkerOutput(TypedDict, total=False):
    """Normalized thinker output used for decoding and streaming."""

    output_ids: list[int]
    step: int
    is_final: bool
    extra_model_outputs: dict[str, Any]


OmniEventType = Literal[
    # Currently used:
    "text_delta",
    "text_final",
    "final",
    # Reserved for future use:
    "audio_chunk",
    "audio_final",
    "image",
    "video_chunk",
    "video_final",
    "debug",
]


@dataclass
class OmniEvent:
    """Streaming-friendly event emitted by decode logic."""

    type: OmniEventType
    modality: str
    payload: dict[str, Any]
    is_final: bool = False
