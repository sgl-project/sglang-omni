# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS pipeline state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Qwen3TTSState:
    """Per-request state for Qwen3-TTS generation."""

    text: str = ""
    task_type: str = "Base"
    task_type_explicit: bool = False
    language: str = "auto"
    voice: str | None = None
    instructions: str | None = None
    ref_audio: Any | None = None
    ref_text: str | None = None
    uploaded_voice_name: str | None = None
    uploaded_voice_created_at: int | None = None
    x_vector_only_mode: bool = False
    non_streaming_mode: bool = False
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    audio_codes: Any | None = None
    ref_code_len: int = 0
    audio_samples: Any | None = None
    sample_rate: int = 24000
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    @staticmethod
    def _tensor_to_list(value: Any) -> Any:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return value

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "task_type": self.task_type,
            "task_type_explicit": self.task_type_explicit,
            "language": self.language,
            "x_vector_only_mode": self.x_vector_only_mode,
            "non_streaming_mode": self.non_streaming_mode,
            "generation_kwargs": dict(self.generation_kwargs),
            "sample_rate": self.sample_rate,
        }
        if self.voice is not None:
            data["voice"] = self.voice
        if self.instructions is not None:
            data["instructions"] = self.instructions
        if self.ref_audio is not None:
            data["ref_audio"] = self.ref_audio
        if self.ref_text is not None:
            data["ref_text"] = self.ref_text
        if self.uploaded_voice_name is not None:
            data["uploaded_voice_name"] = self.uploaded_voice_name
        if self.uploaded_voice_created_at is not None:
            data["uploaded_voice_created_at"] = self.uploaded_voice_created_at
        if self.seed is not None:
            data["seed"] = self.seed
        if self.audio_codes is not None:
            data["audio_codes"] = self._tensor_to_list(self.audio_codes)
        if self.ref_code_len:
            data["ref_code_len"] = self.ref_code_len
        if self.audio_samples is not None:
            data["audio_samples"] = self._tensor_to_list(self.audio_samples)
        if self.prompt_tokens:
            data["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens:
            data["completion_tokens"] = self.completion_tokens
        if self.engine_time_s:
            data["engine_time_s"] = self.engine_time_s
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "Qwen3TTSState":
        if not isinstance(data, dict):
            data = {}
        generation_kwargs = data.get("generation_kwargs")
        return cls(
            text=str(data.get("text", "")),
            task_type=str(data.get("task_type") or "Base"),
            task_type_explicit=bool(data.get("task_type_explicit", False)),
            language=str(data.get("language") or "auto"),
            voice=data.get("voice"),
            instructions=data.get("instructions"),
            ref_audio=data.get("ref_audio"),
            ref_text=data.get("ref_text"),
            uploaded_voice_name=data.get("uploaded_voice_name"),
            uploaded_voice_created_at=data.get("uploaded_voice_created_at"),
            x_vector_only_mode=bool(data.get("x_vector_only_mode", False)),
            non_streaming_mode=bool(data.get("non_streaming_mode", False)),
            generation_kwargs=(
                dict(generation_kwargs) if isinstance(generation_kwargs, dict) else {}
            ),
            seed=data.get("seed"),
            audio_codes=data.get("audio_codes"),
            ref_code_len=int(data.get("ref_code_len", 0) or 0),
            audio_samples=data.get("audio_samples"),
            sample_rate=int(data.get("sample_rate", 24000)),
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            engine_time_s=float(data.get("engine_time_s", 0.0) or 0.0),
        )
