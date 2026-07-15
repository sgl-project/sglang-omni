# SPDX-License-Identifier: Apache-2.0
"""CosyVoice3 pipeline state definition (passed between stages)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CosyVoice3State:
    """Per-request pipeline state for CosyVoice3 zero-shot TTS."""

    # -- From preprocessing -------------------------------------------------
    # The projected LM prompt embeds are large tensors stashed CPU-side in a
    # module-global context keyed by request_id (see request_builders; moved to the
    # engine device at prefill); the State carries only a marker.
    prepared: bool = False
    # Reference artifacts consumed by the flow in the vocoder stage:
    prompt_speech_token: Any | None = None  # [T_ref] speech tokens (list)
    prompt_feat: Any | None = None  # [1, T_mel, 80] reference mel (nested list)
    flow_embedding: Any | None = None  # [1, 192] speaker embedding (nested list)
    # Generation bounds (CosyVoice min/max token-to-text ratio applied in preprocessing):
    min_len: int = 2
    max_len: int = 2048

    # -- Generation params (per-request; defaults come from cosyvoice3.yaml) ---
    top_k: int = 25
    top_p: float = 0.8
    temperature: float = 1.0
    # RAS stand-in; NOTE the shared runner + native sglang penalizer both apply this,
    # so the effective factor is repetition_penalty ** 2 (see request_builders).
    repetition_penalty: float = 1.5
    seed: int | None = None

    # -- From TTS engine ----------------------------------------------------
    speech_tokens: Any | None = None  # [T] generated speech tokens (list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0
    finish_reason: str | None = None

    # -- From vocoder -------------------------------------------------------
    sample_rate: int = 24000

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _to_list(t: Any) -> Any:
        if isinstance(t, torch.Tensor):
            return t.tolist()
        return t

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"prepared": self.prepared}
        if self.prompt_speech_token is not None:
            data["prompt_speech_token"] = self._to_list(self.prompt_speech_token)
        if self.prompt_feat is not None:
            data["prompt_feat"] = self._to_list(self.prompt_feat)
        if self.flow_embedding is not None:
            data["flow_embedding"] = self._to_list(self.flow_embedding)
        data["min_len"] = self.min_len
        data["max_len"] = self.max_len
        data["top_k"] = self.top_k
        data["top_p"] = self.top_p
        data["temperature"] = self.temperature
        data["repetition_penalty"] = self.repetition_penalty
        if self.seed is not None:
            data["seed"] = self.seed
        if self.speech_tokens is not None:
            data["speech_tokens"] = self._to_list(self.speech_tokens)
        if self.prompt_tokens:
            data["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens:
            data["completion_tokens"] = self.completion_tokens
        if self.engine_time_s:
            data["engine_time_s"] = self.engine_time_s
        if self.finish_reason is not None:
            data["finish_reason"] = self.finish_reason
        data["sample_rate"] = self.sample_rate
        return data

    @classmethod
    def from_dict(cls, data: dict) -> CosyVoice3State:
        def _tensor(key: str) -> Any:
            v = data.get(key)
            return torch.tensor(v) if isinstance(v, list) else v

        return cls(
            prepared=data.get("prepared", False),
            prompt_speech_token=_tensor("prompt_speech_token"),
            prompt_feat=_tensor("prompt_feat"),
            flow_embedding=_tensor("flow_embedding"),
            min_len=data.get("min_len", 2),
            max_len=data.get("max_len", 2048),
            top_k=data.get("top_k", 25),
            top_p=data.get("top_p", 0.8),
            temperature=data.get("temperature", 1.0),
            repetition_penalty=data.get("repetition_penalty", 1.5),
            seed=data.get("seed"),
            speech_tokens=data.get("speech_tokens"),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            engine_time_s=data.get("engine_time_s", 0.0),
            finish_reason=data.get("finish_reason"),
            sample_rate=data.get("sample_rate", 24000),
        )
