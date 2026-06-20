# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local (v1.5) pipeline state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def moss_tts_local_special_token_defaults(
    audio_vocab_size: int = 1024,
) -> tuple[tuple[str, int], ...]:
    """Default special-token ids for MOSS-TTS-Local-Transformer-v1.5.

    These differ from the MOSS Delay family: the Local release introduces
    dedicated ``<|audio_start|>``/``<|audio_end|>`` tokens and reuses the
    Qwen vision/video pad ids as the user/assistant audio slot tokens.
    """
    return (
        ("audio_start_token_id", 151669),
        ("audio_end_token_id", 151670),
        ("audio_user_slot_token_id", 151654),
        ("audio_assistant_slot_token_id", 151656),
        ("audio_assistant_gen_slot_token_id", 151656),
        ("audio_pad_token_id", int(audio_vocab_size)),
        ("audio_pad_code", int(audio_vocab_size)),
        ("im_start_token_id", 151644),
        ("im_end_token_id", 151645),
        ("pad_token_id", 151643),
    )


@dataclass
class MossTTSLocalState:
    """Per-request state for MOSS-TTS Local generation."""

    text: str = ""
    ref_audio: Any | None = None
    ref_text: str | None = None
    language: str | None = None
    instructions: str | None = None
    token_count: int | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    audio_codes: Any | None = None
    sample_rate: int = 48000
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    @staticmethod
    def _tensor_to_payload(value: Any) -> Any:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return value

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
        if self.audio_codes is not None:
            data["audio_codes"] = self._tensor_to_payload(self.audio_codes)
        if self.prompt_tokens:
            data["prompt_tokens"] = int(self.prompt_tokens)
        if self.completion_tokens:
            data["completion_tokens"] = int(self.completion_tokens)
        if self.engine_time_s:
            data["engine_time_s"] = float(self.engine_time_s)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "MossTTSLocalState":
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
            audio_codes=data.get("audio_codes"),
            sample_rate=int(data.get("sample_rate", 48000) or 48000),
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            engine_time_s=float(data.get("engine_time_s", 0.0) or 0.0),
        )
