# SPDX-License-Identifier: Apache-2.0
"""Adapter interface for staged omni pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Protocol, TypedDict


class PromptInputs(TypedDict):
    """Tokenized prompt inputs for the thinker."""

    input_ids: Any
    attention_mask: Any
    prompt_text: str


class FrontendOutput(TypedDict, total=False):
    """Normalized frontend output shared across stages."""

    prompt: PromptInputs
    mm_inputs: dict[str, Any]
    adapter_state: dict[str, Any]


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


class OmniAdapter(Protocol):
    """Model-specific policy surface for staged pipelines."""

    name: str
    model_id: str
    modalities: tuple[str, ...]

    def build_frontend(
        self,
        *,
        request_inputs: Any,
        request_params: dict[str, Any],
    ) -> FrontendOutput:
        """Return normalized frontend output."""

    def build_encoder_inputs(
        self,
        *,
        frontend_out: FrontendOutput,
    ) -> dict[str, dict[str, Any]]:
        """Return per-stage encoder inputs."""

    def merge_for_thinker(
        self,
        *,
        frontend_out: FrontendOutput,
        encoder_outs: dict[str, Any],
    ) -> dict[str, Any]:
        """Produce thinker inputs for the AR engine."""

    def decode_events(
        self,
        *,
        thinker_out: ThinkerOutput,
        frontend_out: FrontendOutput,
        step: int,
    ) -> Iterable[OmniEvent]:
        """Convert thinker outputs for a single step into events."""
