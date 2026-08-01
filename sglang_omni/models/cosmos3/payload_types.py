# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 payload schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class PromptInputs(TypedDict):
    """Tokenized prompt passed to the Cosmos3 AR stage."""

    input_ids: Any
    attention_mask: Any
    prompt_text: str
    mm_token_type_ids: Any


class TextOutput(TypedDict, total=False):
    """Normalized text output populated by the later AR stage."""

    output_ids: list[int]
    finish_reason: str | None
    is_final: bool
    output_token_logprobs: Any
    weight_version: Any


@dataclass
class Cosmos3PipelineState:
    """Typed, process-safe state shared by Cosmos3 pipeline stages."""

    prompt: PromptInputs | None = None
    mm_inputs: dict[str, Any] = field(default_factory=dict)
    encoder_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    encoder_outs: dict[str, Any] = field(default_factory=dict)
    thinker_inputs: dict[str, Any] = field(default_factory=dict)
    text_out: TextOutput | None = None
    engine_outputs: dict[str, Any] = field(default_factory=dict)
    stream_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> Cosmos3PipelineState:
        if not isinstance(data, dict):
            data = {}
        text_out = data.get("text_out")
        mm_inputs = data.get("mm_inputs")
        encoder_inputs = data.get("encoder_inputs")
        encoder_outs = data.get("encoder_outs")
        thinker_inputs = data.get("thinker_inputs")
        engine_outputs = data.get("engine_outputs")
        stream_state = data.get("stream_state")
        return cls(
            prompt=data.get("prompt") if isinstance(data.get("prompt"), dict) else None,
            mm_inputs=mm_inputs if isinstance(mm_inputs, dict) else {},
            encoder_inputs=(encoder_inputs if isinstance(encoder_inputs, dict) else {}),
            encoder_outs=encoder_outs if isinstance(encoder_outs, dict) else {},
            thinker_inputs=(thinker_inputs if isinstance(thinker_inputs, dict) else {}),
            text_out=text_out if isinstance(text_out, dict) else None,
            engine_outputs=(engine_outputs if isinstance(engine_outputs, dict) else {}),
            stream_state=stream_state if isinstance(stream_state, dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
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
        if self.text_out is not None:
            data["text_out"] = self.text_out
        if self.engine_outputs:
            data["engine_outputs"] = self.engine_outputs
        if self.stream_state:
            data["stream_state"] = self.stream_state
        return data


__all__ = ["Cosmos3PipelineState", "PromptInputs", "TextOutput"]
