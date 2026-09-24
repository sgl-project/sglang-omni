# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o state schemas compatible with the shared streaming detokenizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    import torch
else:
    pass


class PromptInputs(TypedDict):
    """Tokenized prompt inputs for the thinker."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_text: str


class SpeakerPromptInputs(TypedDict):
    """CPU reference tokens, lengths, speaker embedding, and mel conditioning."""

    speech_tokens: torch.Tensor
    speech_token_len: torch.Tensor
    speaker_embedding: torch.Tensor
    prompt_mel: torch.Tensor


class ThinkerOutput(TypedDict, total=False):
    """Normalized thinker output used for decoding and streaming."""

    output_ids: list[int]
    step: int
    is_final: bool
    extra_model_outputs: dict[str, Any]


@dataclass(kw_only=True)
class MiniCPMOPipelineState:
    """Per-request state serialized as plain dictionaries across processes."""

    prompt: PromptInputs | None = None
    mm_inputs: dict[str, Any] = field(default_factory=dict)
    encoder_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    encoder_outs: dict[str, Any] = field(default_factory=dict)
    thinker_inputs: dict[str, Any] = field(default_factory=dict)
    thinker_out: ThinkerOutput | None = None
    engine_outputs: dict[str, Any] = field(default_factory=dict)
    stream_state: dict[str, Any] = field(default_factory=dict)
    speaker_prompt: SpeakerPromptInputs | None = None

    @classmethod
    def from_dict(cls, data: Any) -> "MiniCPMOPipelineState":
        if not isinstance(data, dict):
            data = {}
        else:
            pass

        def _dict(key: str) -> dict[str, Any]:
            value = data.get(key)
            return value if isinstance(value, dict) else {}

        thinker_out = data.get("thinker_out")
        speaker_prompt = data.get("speaker_prompt")
        if isinstance(speaker_prompt, dict):
            speaker_prompt_inputs = speaker_prompt
        else:
            speaker_prompt_inputs = None
        return cls(
            prompt=data.get("prompt"),
            mm_inputs=_dict("mm_inputs"),
            encoder_inputs=_dict("encoder_inputs"),
            encoder_outs=_dict("encoder_outs"),
            thinker_inputs=_dict("thinker_inputs"),
            thinker_out=thinker_out if isinstance(thinker_out, dict) else None,
            engine_outputs=_dict("engine_outputs"),
            stream_state=_dict("stream_state"),
            speaker_prompt=speaker_prompt_inputs,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
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
        if self.speaker_prompt is not None:
            data["speaker_prompt"] = self.speaker_prompt
        else:
            pass
        return data
