# SPDX-License-Identifier: Apache-2.0
"""Per-request pipeline state for Higgs TTS.

Carried between stages via :class:`sglang_omni.proto.StagePayload.data`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sglang_omni.scheduling.pipeline_state import PipelineStateBase


@dataclass
class HiggsTtsState(PipelineStateBase):
    """Per-request state threaded through preprocessing → audio_encoder →
    tts_engine → vocoder. Fields populate lazily so a deserialised state is
    valid at any stage boundary."""

    sample_rate: int = 24000

    # preprocessing / audio_encoder
    prompt_token_ids: list[int] = field(default_factory=list)
    reference_codes_delayed: list[list[int]] | None = None
    target_text: str | None = None
    reference_text: str | None = None
    reference_waveform: Any | None = None  # mono 24 kHz [1, 1, L] torch.Tensor
    reference_code_cache_key: str | None = None
    uploaded_voice_name: str | None = None
    uploaded_voice_created_at: int | None = None

    num_codebooks: int = 8
    codebook_size: int = 1026  # 1024 data + <|boc|> + <|eoc|>

    # generation params
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None

    # RL rollout controls
    return_logprob: bool = False
    return_omni_rollout: bool = False

    # tts_engine
    output_codes_delayed: list[list[int]] | None = None
    omni_rollout: dict[str, Any] | None = None

    # vocoder
    audio_samples: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "prompt_token_ids": list(self.prompt_token_ids),
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        if self.reference_codes_delayed is not None:
            data["reference_codes_delayed"] = self.reference_codes_delayed
        if self.target_text is not None:
            data["target_text"] = self.target_text
        if self.reference_text is not None:
            data["reference_text"] = self.reference_text
        if self.reference_waveform is not None:
            data["reference_waveform"] = self.reference_waveform
        if self.reference_code_cache_key is not None:
            data["reference_code_cache_key"] = self.reference_code_cache_key
        if self.uploaded_voice_name is not None:
            data["uploaded_voice_name"] = self.uploaded_voice_name
        if self.uploaded_voice_created_at is not None:
            data["uploaded_voice_created_at"] = self.uploaded_voice_created_at
        for key in ("top_p", "top_k", "seed"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        for key in ("return_logprob", "return_omni_rollout"):
            if getattr(self, key):
                data[key] = True
        if self.output_codes_delayed is not None:
            data["output_codes_delayed"] = self.output_codes_delayed
        if self.omni_rollout is not None:
            data["omni_rollout"] = self.omni_rollout
        self.append_usage_fields(data)
        if self.audio_samples is not None:
            data["audio_samples"] = self.audio_samples
            data["sample_rate"] = self.sample_rate
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HiggsTtsState:
        return cls(
            prompt_token_ids=list(data.get("prompt_token_ids", [])),
            reference_codes_delayed=data.get("reference_codes_delayed"),
            target_text=data.get("target_text"),
            reference_text=data.get("reference_text"),
            reference_waveform=data.get("reference_waveform"),
            reference_code_cache_key=data.get("reference_code_cache_key"),
            uploaded_voice_name=data.get("uploaded_voice_name"),
            uploaded_voice_created_at=data.get("uploaded_voice_created_at"),
            num_codebooks=data.get("num_codebooks", 8),
            codebook_size=data.get("codebook_size", 1026),
            max_new_tokens=data.get("max_new_tokens", 2048),
            temperature=data.get("temperature", 1.0),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            seed=data.get("seed"),
            return_logprob=data.get("return_logprob", False),
            return_omni_rollout=data.get("return_omni_rollout", False),
            output_codes_delayed=data.get("output_codes_delayed"),
            omni_rollout=data.get("omni_rollout"),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            engine_time_s=data.get("engine_time_s", 0.0),
            audio_samples=data.get("audio_samples"),
            sample_rate=data.get("sample_rate", 24000),
        )


__all__ = ["HiggsTtsState"]
