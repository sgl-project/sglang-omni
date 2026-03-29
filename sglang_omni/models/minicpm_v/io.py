# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-V/MiniCPM-o payload schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


class PromptInputs(TypedDict):
    """Tokenized prompt inputs for the LLM."""

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


class LLMOutput(TypedDict, total=False):
    """Normalized LLM output used for decoding and streaming."""

    output_ids: list[int]
    step: int
    is_final: bool
    extra_model_outputs: dict[str, Any]


@dataclass
class PipelineState:
    """Typed view of the per-request pipeline state.

    This stays msgpack-safe by converting back to plain dicts before crossing
    process boundaries.

    Key differences from Qwen3-Omni:
    - mm_inputs["image"] contains `tgt_sizes` and `slice_lengths` instead of `image_grid_thw`
    - Audio input support via mm_inputs["audio"] (MiniCPM-o only)
    - Audio output support via vocoder_out (MiniCPM-o only)
    """

    raw_inputs: Any | None = None
    prompt: PromptInputs | None = None
    mm_inputs: dict[str, Any] = field(default_factory=dict)
    encoder_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    encoder_outs: dict[str, Any] = field(default_factory=dict)
    llm_inputs: dict[str, Any] = field(default_factory=dict)
    llm_out: LLMOutput | None = None
    engine_outputs: dict[str, Any] = field(default_factory=dict)
    stream_state: dict[str, Any] = field(default_factory=dict)
    # Audio output (MiniCPM-o only)
    vocoder_out: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "PipelineState":
        if not isinstance(data, dict):
            data = {}
        mm_inputs = data.get("mm_inputs")
        encoder_inputs = data.get("encoder_inputs")
        encoder_outs = data.get("encoder_outs")
        llm_inputs = data.get("llm_inputs")
        engine_outputs = data.get("engine_outputs")
        stream_state = data.get("stream_state")
        llm_out = data.get("llm_out")
        vocoder_out = data.get("vocoder_out")
        return cls(
            raw_inputs=data.get("raw_inputs"),
            prompt=data.get("prompt"),
            mm_inputs=mm_inputs if isinstance(mm_inputs, dict) else {},
            encoder_inputs=encoder_inputs if isinstance(encoder_inputs, dict) else {},
            encoder_outs=encoder_outs if isinstance(encoder_outs, dict) else {},
            llm_inputs=llm_inputs if isinstance(llm_inputs, dict) else {},
            llm_out=llm_out if isinstance(llm_out, dict) else None,
            engine_outputs=engine_outputs if isinstance(engine_outputs, dict) else {},
            stream_state=stream_state if isinstance(stream_state, dict) else {},
            vocoder_out=vocoder_out if isinstance(vocoder_out, dict) else {},
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
        if self.llm_inputs:
            data["llm_inputs"] = self.llm_inputs
        if self.llm_out is not None:
            data["llm_out"] = self.llm_out
        if self.engine_outputs:
            data["engine_outputs"] = self.engine_outputs
        if self.stream_state:
            data["stream_state"] = self.stream_state
        if self.vocoder_out:
            data["vocoder_out"] = self.vocoder_out
        return data


OmniEventType = Literal[
    "text_delta",
    "text_final",
    "audio_delta",
    "audio_final",
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
