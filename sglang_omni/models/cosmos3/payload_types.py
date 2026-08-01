# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 payload schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class PromptInputs(TypedDict):
    """Tokenized text prompt passed to the Cosmos3 AR stage."""

    input_ids: Any
    attention_mask: Any
    prompt_text: str


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
    text_out: TextOutput | None = None
    engine_outputs: dict[str, Any] = field(default_factory=dict)
    stream_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> Cosmos3PipelineState:
        if not isinstance(data, dict):
            data = {}
        text_out = data.get("text_out")
        engine_outputs = data.get("engine_outputs")
        stream_state = data.get("stream_state")
        return cls(
            prompt=data.get("prompt") if isinstance(data.get("prompt"), dict) else None,
            text_out=text_out if isinstance(text_out, dict) else None,
            engine_outputs=(engine_outputs if isinstance(engine_outputs, dict) else {}),
            stream_state=stream_state if isinstance(stream_state, dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.prompt is not None:
            data["prompt"] = self.prompt
        if self.text_out is not None:
            data["text_out"] = self.text_out
        if self.engine_outputs:
            data["engine_outputs"] = self.engine_outputs
        if self.stream_state:
            data["stream_state"] = self.stream_state
        return data


__all__ = ["Cosmos3PipelineState", "PromptInputs", "TextOutput"]
