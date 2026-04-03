"""Pipeline state definition for Voxtral TTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VoxtralTTSState:
    """Per-request pipeline state for Voxtral TTS."""

    input_ids: list[int] | None = None
    voice: str | None = None

    max_new_tokens: int = 4096

    # Generation output: list of [num_codebooks] tensors, one per frame
    audio_codes: Any | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Vocoder output
    audio_samples: Any | None = None
    sample_rate: int = 24000

    @staticmethod
    def _tensor_to_list(t: Any) -> Any:
        if isinstance(t, torch.Tensor):
            return t.tolist()
        return t

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.input_ids is not None:
            data["input_ids"] = self.input_ids
        if self.voice is not None:
            data["voice"] = self.voice
        data["max_new_tokens"] = self.max_new_tokens
        if self.audio_codes is not None:
            data["audio_codes"] = self._tensor_to_list(self.audio_codes)
        if self.prompt_tokens:
            data["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens:
            data["completion_tokens"] = self.completion_tokens
        if self.audio_samples is not None:
            data["audio_samples"] = self._tensor_to_list(self.audio_samples)
        data["sample_rate"] = self.sample_rate
        return data

    @classmethod
    def from_dict(cls, data: dict) -> VoxtralTTSState:
        audio_codes = data.get("audio_codes")
        if audio_codes is not None and isinstance(audio_codes, list):
            audio_codes = torch.tensor(audio_codes)
        return cls(
            input_ids=data.get("input_ids"),
            voice=data.get("voice"),
            max_new_tokens=data.get("max_new_tokens", 4096),
            audio_codes=audio_codes,
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            audio_samples=data.get("audio_samples"),
            sample_rate=data.get("sample_rate", 24000),
        )
