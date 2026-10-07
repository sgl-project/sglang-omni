# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni payload schemas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import torch


class PromptInputs(TypedDict):
    """Tokenized prompt inputs for the thinker."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_text: str


class UsagePromptInputs(TypedDict):
    input_ids: torch.Tensor


class EncoderCacheKeyInputs(TypedDict, total=False):
    cache_key: str


class AudioEncoderInputs(EncoderCacheKeyInputs):
    audio_feats: torch.Tensor
    audio_feats_lengths: torch.Tensor
    audio_placeholder_loc_lens: torch.Tensor


class ImageEncoderInputs(EncoderCacheKeyInputs, total=False):
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    pixel_values_videos: torch.Tensor | None
    video_grid_thw: torch.Tensor | None


class SkippedEncoderInputs(EncoderCacheKeyInputs):
    _skip: bool
    _result: dict[str, object]


class ThinkerEmbeddingInputs(TypedDict, total=False):
    audio_embeds: torch.Tensor | None
    image_embeds: torch.Tensor | None
    video_embeds: torch.Tensor | None


class ThinkerOutput(TypedDict, total=False):
    """Normalized thinker output used for decoding and streaming."""

    output_ids: list[int]
    step: int
    is_final: bool
    extra_model_outputs: dict[str, torch.Tensor | list[torch.Tensor] | list[int]]
    finish_reason: str


class StreamState(TypedDict, total=False):
    token_ids: list[int]
    text: str
    emitted_text: str
    emitted_ids: list[int]
    accumulated_text: str


@dataclass
class MingOmniPipelineState:
    """Typed view of the per-request pipeline state.

    Stays msgpack-safe by converting back to plain dicts before crossing
    process boundaries.
    """

    raw_inputs: object | None = None
    prompt: PromptInputs | UsagePromptInputs | None = None
    mm_inputs: dict[str, object] = field(default_factory=dict)
    encoder_inputs: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    encoder_outs: dict[str, object] = field(default_factory=dict)
    thinker_inputs: dict[str, Mapping[str, torch.Tensor | str]] = field(
        default_factory=dict
    )
    thinker_out: ThinkerOutput | None = None
    engine_outputs: dict[str, object] = field(default_factory=dict)
    stream_state: StreamState = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: object) -> "MingOmniPipelineState":
        if not isinstance(data, dict):
            data = {}
        else:
            pass
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

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {}
        if self.raw_inputs is not None:
            data["raw_inputs"] = self.raw_inputs
        else:
            pass
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


MingOmniEventType = Literal[
    "text_delta",
    "text_final",
    "audio_chunk",
    "audio_final",
    "debug",
    "final",
]


@dataclass
class MingOmniEvent:
    """Streaming-friendly event emitted by decode logic."""

    type: MingOmniEventType
    modality: str
    payload: dict[str, str | list[str]]
    is_final: bool = False
