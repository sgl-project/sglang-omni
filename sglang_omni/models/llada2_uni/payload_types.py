# SPDX-License-Identifier: Apache-2.0
"""LLaDA2-Uni payload schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class ThinkerOutput(TypedDict, total=False):
    """Normalized thinker output used for decoding."""

    output_ids: list[int]
    is_final: bool
    finish_reason: str | None


@dataclass
class LLaDA2UniPipelineState:
    """Typed view of the per-request pipeline state."""

    prompt: dict[str, Any] | None = None
    encoder_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    encoder_outs: dict[str, Any] = field(default_factory=dict)
    thinker_out: ThinkerOutput | None = None
    engine_outputs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "LLaDA2UniPipelineState":
        if not isinstance(data, dict):
            data = {}
        encoder_inputs = data.get("encoder_inputs")
        encoder_outs = data.get("encoder_outs")
        engine_outputs = data.get("engine_outputs")
        thinker_out = data.get("thinker_out")
        return cls(
            prompt=data.get("prompt"),
            encoder_inputs=encoder_inputs if isinstance(encoder_inputs, dict) else {},
            encoder_outs=encoder_outs if isinstance(encoder_outs, dict) else {},
            thinker_out=thinker_out if isinstance(thinker_out, dict) else None,
            engine_outputs=engine_outputs if isinstance(engine_outputs, dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.prompt is not None:
            data["prompt"] = self.prompt
        if self.encoder_inputs:
            data["encoder_inputs"] = self.encoder_inputs
        if self.encoder_outs:
            data["encoder_outs"] = self.encoder_outs
        if self.thinker_out is not None:
            data["thinker_out"] = self.thinker_out
        if self.engine_outputs:
            data["engine_outputs"] = self.engine_outputs
        return data


@dataclass
class LLaDA2UniEvent:
    """Streaming-friendly event emitted by decode logic."""

    type: str
    modality: str
    payload: dict[str, Any]
    is_final: bool = False
