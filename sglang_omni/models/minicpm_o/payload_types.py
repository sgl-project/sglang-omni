# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o payload schemas.

The state keys deliberately mirror ``qwen3_omni.payload_types`` so the shared
streaming detokenizer (which reads ``thinker_out`` / ``stream_state`` from the
payload dict) works unchanged for this pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


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


@dataclass
class MiniCPMOPipelineState:
    """Typed view of the per-request pipeline state.

    Stays msgpack-safe by converting back to plain dicts before crossing
    process boundaries.
    """

    prompt: PromptInputs | None = None
    mm_inputs: dict[str, Any] = field(default_factory=dict)
    encoder_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    encoder_outs: dict[str, Any] = field(default_factory=dict)
    thinker_inputs: dict[str, Any] = field(default_factory=dict)
    thinker_out: ThinkerOutput | None = None
    engine_outputs: dict[str, Any] = field(default_factory=dict)
    stream_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "MiniCPMOPipelineState":
        if not isinstance(data, dict):
            data = {}

        def _dict(key: str) -> dict[str, Any]:
            value = data.get(key)
            return value if isinstance(value, dict) else {}

        thinker_out = data.get("thinker_out")
        return cls(
            prompt=data.get("prompt"),
            mm_inputs=_dict("mm_inputs"),
            encoder_inputs=_dict("encoder_inputs"),
            encoder_outs=_dict("encoder_outs"),
            thinker_inputs=_dict("thinker_inputs"),
            thinker_out=thinker_out if isinstance(thinker_out, dict) else None,
            engine_outputs=_dict("engine_outputs"),
            stream_state=_dict("stream_state"),
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
        if self.thinker_out is not None:
            data["thinker_out"] = self.thinker_out
        if self.engine_outputs:
            data["engine_outputs"] = self.engine_outputs
        if self.stream_state:
            data["stream_state"] = self.stream_state
        return data
