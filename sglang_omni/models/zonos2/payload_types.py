# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 TTS pipeline state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ZONOS2State:
    """Per-request state placeholder for ZONOS2 generation."""

    text: str = ""
    language: str = "auto"
    voice: str | None = None
    ref_audio: Any | None = None
    ref_text: str | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    frame_width: int = 9
    codebook_size: int = 1024
    sample_rate: int = 44100
    audio_codes: Any | None = None
    audio_samples: Any | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "language": self.language,
            "generation_kwargs": dict(self.generation_kwargs),
            "frame_width": self.frame_width,
            "codebook_size": self.codebook_size,
            "sample_rate": self.sample_rate,
        }
        if self.voice is not None:
            data["voice"] = self.voice
        if self.ref_audio is not None:
            data["ref_audio"] = self.ref_audio
        if self.ref_text is not None:
            data["ref_text"] = self.ref_text
        if self.audio_codes is not None:
            data["audio_codes"] = self.audio_codes
        if self.audio_samples is not None:
            data["audio_samples"] = self.audio_samples
        if self.prompt_tokens:
            data["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens:
            data["completion_tokens"] = self.completion_tokens
        if self.engine_time_s:
            data["engine_time_s"] = self.engine_time_s
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "ZONOS2State":
        if not isinstance(data, dict):
            data = {}
        generation_kwargs = data.get("generation_kwargs")
        return cls(
            text=str(data.get("text", "")),
            language=str(data.get("language") or "auto"),
            voice=data.get("voice"),
            ref_audio=data.get("ref_audio"),
            ref_text=data.get("ref_text"),
            generation_kwargs=(
                dict(generation_kwargs) if isinstance(generation_kwargs, dict) else {}
            ),
            frame_width=int(data.get("frame_width", 9) or 9),
            codebook_size=int(data.get("codebook_size", 1024) or 1024),
            sample_rate=int(data.get("sample_rate", 44100) or 44100),
            audio_codes=data.get("audio_codes"),
            audio_samples=data.get("audio_samples"),
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            engine_time_s=float(data.get("engine_time_s", 0.0) or 0.0),
        )
