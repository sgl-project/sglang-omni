# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 payload schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

import torch
from typing_extensions import NotRequired


class PromptInputs(TypedDict):
    """Tokenized text prompt passed to the Cosmos3 AR stage."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_text: str


class TextOutput(TypedDict):
    """Normalized text output populated by the later AR stage."""

    output_ids: list[int]
    is_final: bool
    finish_reason: NotRequired[str | None]
    matched_stop: NotRequired[int | str | None]
    output_token_logprobs: NotRequired[list[Any]]
    weight_version: NotRequired[str | None]


@dataclass
class Cosmos3PipelineState:
    """Typed, process-safe state shared by Cosmos3 pipeline stages."""

    prompt: PromptInputs | None = None
    text_out: TextOutput | None = None
    engine_outputs: dict[str, Any] = field(default_factory=dict)
    stream_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cosmos3PipelineState:
        return cls(**data)

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
