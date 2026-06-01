# SPDX-License-Identifier: Apache-2.0
"""Shared request/result models for the S2-Pro TTS playground."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_RESPONSE_FORMAT = "wav"
DEFAULT_VOICE = "default"


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int


@dataclass(frozen=True)
class SpeechSynthesisRequest:
    text: str
    reference_audio_path: str | None
    reference_text: str
    settings: GenerationSettings
    voice: str = DEFAULT_VOICE
    response_format: str = DEFAULT_RESPONSE_FORMAT

    def validate(self) -> None:
        if not self.text.strip():
            raise ValueError("Please enter some text to synthesize.")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input": self.text,
            "voice": self.voice,
            "response_format": self.response_format,
            "max_new_tokens": self.settings.max_new_tokens,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "top_k": self.settings.top_k,
        }
        if self.reference_audio_path is not None:
            payload["ref_audio"] = self.reference_audio_path
            if self.reference_text.strip():
                payload["ref_text"] = self.reference_text.strip()
        return payload

    def build_history_user_content(self) -> list[Any]:
        content: list[Any] = [self.text]
        if self.reference_audio_path is not None:
            content.append(
                {
                    "path": self.reference_audio_path,
                    "mime_type": "audio/wav",
                }
            )
        return content


@dataclass(frozen=True)
class NonStreamingSpeechResult:
    audio_bytes: bytes
    elapsed_s: float

    @property
    def size_bytes(self) -> int:
        return len(self.audio_bytes)
