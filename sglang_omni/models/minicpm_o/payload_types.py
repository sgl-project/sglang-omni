# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o state schemas compatible with the shared streaming detokenizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import torch
else:
    pass


class PromptInputs(TypedDict):
    """Tokenized prompt inputs for the thinker."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_text: str


class ThinkerOutput(TypedDict, total=False):
    """Normalized thinker output used for decoding and streaming."""

    output_ids: list[int]
    step: int
    is_final: bool
    extra_model_outputs: dict[str, object]
    finish_reason: str
    weight_version: str
    output_token_logprobs: list[list[float | int]]


@dataclass(kw_only=True)
class MiniCPMOPipelineState:
    """Per-request state serialized as plain dictionaries across processes."""

    prompt: PromptInputs | None = None
    mm_inputs: dict[str, object] = field(default_factory=dict)
    encoder_inputs: dict[str, object] = field(default_factory=dict)
    encoder_outs: dict[str, object] = field(default_factory=dict)
    thinker_inputs: dict[str, object] = field(default_factory=dict)
    thinker_out: ThinkerOutput | None = None
    engine_outputs: dict[str, object] = field(default_factory=dict)
    stream_state: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: object) -> "MiniCPMOPipelineState":
        if not isinstance(data, dict):
            data = {}
        else:
            pass

        def _dict(key: str) -> dict[str, object]:
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

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {}
        if self.prompt is not None:
            data["prompt"] = self.prompt
        else:
            pass
        if self.mm_inputs:
            data["mm_inputs"] = self.mm_inputs
        else:
            pass
        if self.encoder_inputs:
            data["encoder_inputs"] = self.encoder_inputs
        else:
            pass
        if self.encoder_outs:
            data["encoder_outs"] = self.encoder_outs
        else:
            pass
        if self.thinker_inputs:
            data["thinker_inputs"] = self.thinker_inputs
        else:
            pass
        if self.thinker_out is not None:
            data["thinker_out"] = self.thinker_out
        else:
            pass
        if self.engine_outputs:
            data["engine_outputs"] = self.engine_outputs
        else:
            pass
        if self.stream_state:
            data["stream_state"] = self.stream_state
        else:
            pass
        return data
