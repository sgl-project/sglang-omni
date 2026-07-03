# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS pipeline state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sglang_omni.scheduling.pipeline_state import PipelineStateBase


def moss_tts_special_token_defaults(
    audio_vocab_size: int = 1024,
) -> tuple[tuple[str, int], ...]:
    """Default MOSS-TTS special-token ids, shared by model- and processor-side
    config normalization. ``audio_pad_code`` follows ``audio_vocab_size``."""
    return (
        ("audio_start_token_id", 151652),
        ("audio_end_token_id", 151653),
        ("audio_assistant_gen_slot_token_id", 151656),
        ("audio_assistant_delay_slot_token_id", 151662),
        ("audio_pad_code", int(audio_vocab_size)),
        ("im_start_token_id", 151644),
        ("im_end_token_id", 151645),
        ("pad_token_id", 151643),
    )


@dataclass
class MossTTSState(PipelineStateBase):
    """Per-request state for MOSS-TTS Delay generation."""

    sample_rate: int = 24000
    text: str = ""
    ref_audio: Any | None = None
    ref_text: str | None = None
    language: str | None = None
    instructions: str | None = None
    token_count: int | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    delayed_audio_codes: Any | None = None
    assistant_start_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "generation_kwargs": dict(self.generation_kwargs),
            "sample_rate": int(self.sample_rate),
        }
        if self.ref_audio is not None:
            data["ref_audio"] = self.ref_audio
        if self.ref_text is not None:
            data["ref_text"] = self.ref_text
        if self.language is not None:
            data["language"] = self.language
        if self.instructions is not None:
            data["instructions"] = self.instructions
        if self.token_count is not None:
            data["token_count"] = int(self.token_count)
        if self.delayed_audio_codes is not None:
            data["delayed_audio_codes"] = self.serialize_value(self.delayed_audio_codes)
        if self.assistant_start_length:
            data["assistant_start_length"] = int(self.assistant_start_length)
        self.append_usage_fields(data)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "MossTTSState":
        if not isinstance(data, dict):
            data = {}
        generation_kwargs = data.get("generation_kwargs")
        return cls(
            text=str(data.get("text", "")),
            ref_audio=data.get("ref_audio"),
            ref_text=data.get("ref_text"),
            language=data.get("language"),
            instructions=data.get("instructions"),
            token_count=(
                int(data["token_count"])
                if data.get("token_count") is not None
                else None
            ),
            generation_kwargs=(
                dict(generation_kwargs) if isinstance(generation_kwargs, dict) else {}
            ),
            delayed_audio_codes=data.get("delayed_audio_codes"),
            assistant_start_length=int(data.get("assistant_start_length", 0) or 0),
            sample_rate=int(data.get("sample_rate", 24000) or 24000),
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            engine_time_s=float(data.get("engine_time_s", 0.0) or 0.0),
        )
