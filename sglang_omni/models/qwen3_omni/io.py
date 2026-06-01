# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni payload schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


class PromptInputs(TypedDict):
    """Tokenized prompt inputs for the thinker."""

    input_ids: Any
    attention_mask: Any
    prompt_text: str


class PreprocessingData(TypedDict, total=False):
    """Preprocessing outputs stored on StagePayload.data."""

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
    finish_reason: str | None
    extra_model_outputs: dict[str, Any]


@dataclass
class PipelineState:
    """Typed view of the per-request pipeline state.

    This stays msgpack-safe by converting back to plain dicts before crossing
    process boundaries.
    """

    raw_inputs: Any | None = None
    prompt: PromptInputs | None = None
    mm_inputs: dict[str, Any] = field(default_factory=dict)
    encoder_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    encoder_outs: dict[str, Any] = field(default_factory=dict)
    thinker_inputs: dict[str, Any] = field(default_factory=dict)
    thinker_out: ThinkerOutput | None = None
    engine_outputs: dict[str, Any] = field(default_factory=dict)
    stream_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "PipelineState":
        if not isinstance(data, dict):
            data = {}
        mm_inputs = data.get("mm_inputs")
        encoder_inputs = data.get("encoder_inputs")
        encoder_outs = data.get("encoder_outs")
        thinker_inputs = data.get("thinker_inputs")
        engine_outputs = data.get("engine_outputs")
        stream_state = data.get("stream_state")
        thinker_out = data.get("thinker_out")
        return cls(
            raw_inputs=data.get("raw_inputs"),
            prompt=data.get("prompt"),
            mm_inputs=mm_inputs if isinstance(mm_inputs, dict) else {},
            encoder_inputs=encoder_inputs if isinstance(encoder_inputs, dict) else {},
            encoder_outs=encoder_outs if isinstance(encoder_outs, dict) else {},
            thinker_inputs=thinker_inputs if isinstance(thinker_inputs, dict) else {},
            thinker_out=thinker_out if isinstance(thinker_out, dict) else None,
            engine_outputs=engine_outputs if isinstance(engine_outputs, dict) else {},
            stream_state=stream_state if isinstance(stream_state, dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.raw_inputs is not None:
            data["raw_inputs"] = self.raw_inputs
        if self.prompt is not None:
            data["prompt"] = self.prompt
        if self.mm_inputs:
            data["mm_inputs"] = self.mm_inputs
        if self.encoder_inputs:
            data["encoder_inputs"] = self.encoder_inputs
        if self.encoder_outs:
            data["encoder_outs"] = self.encoder_outs
        if self.thinker_inputs:
            data["thinker_inputs"] = self.thinker_inputs
        if self.thinker_out is not None:
            data["thinker_out"] = self.thinker_out
        if self.engine_outputs:
            data["engine_outputs"] = self.engine_outputs
        if self.stream_state:
            data["stream_state"] = self.stream_state
        return data


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
