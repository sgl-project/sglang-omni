# SPDX-License-Identifier: Apache-2.0
# Inter-stage state follows the sglang-omni TTS pipeline pattern; the EOS-frame semantics
# mirror HuggingFace Transformers CSM (transformers/models/csm, Apache-2.0).
"""Per-request pipeline state for CSM TTS — THE inter-stage wire schema.

Carried between stages via :class:`sglang_omni.proto.StagePayload.data`.
Fields populate lazily so a deserialised state is valid at any stage boundary.


"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CsmTtsState:
    """State threaded through preprocessing → audio_encoder → tts_engine →
    vocoder.

    ``context`` entries are ``{speaker_id, text, codes_F32 | waveform}``;
    ``context_codes`` holds one CPU int32 ``[F, 32]`` matrix per context
    segment; ``num_ctx_codes_consumed`` is the chunked-prefill overlay cursor; ``max_new_tokens`` counts FRAMES (1 frame = 1 backbone
    position, exactly like HF); ``output_frames`` is a list of ``[32]``
    int rows and INCLUDES the EOS frame (HF ``sequences`` parity — the
    vocoder never receives it).
    """

    # preprocessing
    text: str | None = None
    speaker_id: int = 0
    context: list[dict[str, Any]] = field(default_factory=list)

    # preprocessing / audio_encoder
    prompt_ids: list[int] = field(default_factory=list)
    context_codes: list[Any] | None = None  # per segment: CPU int32 [F, 32]
    num_ctx_codes_consumed: int = 0

    # generation params (dual sampling sets; HF defaults T=0.9 / top_k=50)
    max_new_tokens: int = 125  # frames (DEFAULT_MAX_FRAMES)
    temperature: float = 0.9
    top_k: int = 50
    top_p: float | None = None
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    seed: int | None = None
    stream: bool = False

    # tts_engine outputs
    output_frames: list[Any] | None = None  # list of [32] int rows
    prompt_tokens: int = 0
    completion_frames: int = 0
    engine_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for ``StagePayload.data`` (sparse: omit None/empty;
        higgs pattern)."""
        data: dict[str, Any] = {
            "speaker_id": self.speaker_id,
            "prompt_ids": list(self.prompt_ids),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "depth_temperature": self.depth_temperature,
            "depth_top_k": self.depth_top_k,
        }
        if self.text is not None:
            data["text"] = self.text
        if self.context:
            data["context"] = self.context
        if self.context_codes is not None:
            data["context_codes"] = self.context_codes
        if self.num_ctx_codes_consumed:
            data["num_ctx_codes_consumed"] = self.num_ctx_codes_consumed
        for key in ("top_p", "seed"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.stream:
            data["stream"] = True
        if self.output_frames is not None:
            data["output_frames"] = self.output_frames
        for key in ("prompt_tokens", "completion_frames", "engine_time_s"):
            value = getattr(self, key)
            if value:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CsmTtsState":
        """Rebuild from ``StagePayload.data`` produced by :meth:`to_dict`."""
        return cls(
            text=data.get("text"),
            speaker_id=data.get("speaker_id", 0),
            context=list(data.get("context", [])),
            prompt_ids=list(data.get("prompt_ids", [])),
            context_codes=data.get("context_codes"),
            num_ctx_codes_consumed=data.get("num_ctx_codes_consumed", 0),
            max_new_tokens=data.get("max_new_tokens", 125),
            temperature=data.get("temperature", 0.9),
            top_k=data.get("top_k", 50),
            top_p=data.get("top_p"),
            depth_temperature=data.get("depth_temperature", 0.9),
            depth_top_k=data.get("depth_top_k", 50),
            seed=data.get("seed"),
            stream=data.get("stream", False),
            output_frames=data.get("output_frames"),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_frames=data.get("completion_frames", 0),
            engine_time_s=data.get("engine_time_s", 0.0),
        )


__all__ = ["CsmTtsState"]
