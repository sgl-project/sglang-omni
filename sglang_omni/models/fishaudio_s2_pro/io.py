# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro pipeline state definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class S2ProState:
    """Per-request pipeline state for FishAudio S2-Pro TTS.

    Unlike the S1 state (which uses ``input_values`` as a multi-codebook
    tensor), S2-Pro stores the tokenized prompt as ``input_ids`` plus
    separate VQ-related tensors for embedding, matching the
    ``FishQwen3OmniForCausalLM`` interface.
    """

    # -- From preprocessing ------------------------------------------------
    input_ids: Any = None  # [seq_len] token IDs (1D)
    vq_parts: Any = None  # [num_vq_tokens, num_codebooks] VQ codebook values
    vq_mask_tokens: Any = None  # [seq_len] boolean mask

    # -- Generation params -------------------------------------------------
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 30
    seed: int | None = None

    # -- From TTS engine ---------------------------------------------------
    output_codes: Any = None  # [num_semantic, num_codebooks] VQ codes
    num_semantic_tokens: int = 0

    # -- From vocoder ------------------------------------------------------
    audio_samples: Any = None
    sample_rate: int = 44100

    @staticmethod
    def _tensor_to_list(t: Any) -> Any:
        if isinstance(t, torch.Tensor):
            return t.tolist()
        return t

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.input_ids is not None:
            data["input_ids"] = self._tensor_to_list(self.input_ids)
        if self.vq_parts is not None:
            data["vq_parts"] = self._tensor_to_list(self.vq_parts)
        if self.vq_mask_tokens is not None:
            data["vq_mask_tokens"] = self._tensor_to_list(self.vq_mask_tokens)
        data["max_new_tokens"] = self.max_new_tokens
        data["temperature"] = self.temperature
        data["top_p"] = self.top_p
        data["top_k"] = self.top_k
        if self.seed is not None:
            data["seed"] = self.seed
        if self.output_codes is not None:
            data["output_codes"] = self._tensor_to_list(self.output_codes)
        data["num_semantic_tokens"] = self.num_semantic_tokens
        if self.audio_samples is not None:
            data["audio_samples"] = self._tensor_to_list(self.audio_samples)
        data["sample_rate"] = self.sample_rate
        return data

    @classmethod
    def from_dict(cls, data: Any) -> S2ProState:
        if not isinstance(data, dict):
            data = {}
        return cls(
            input_ids=data.get("input_ids"),
            vq_parts=data.get("vq_parts"),
            vq_mask_tokens=data.get("vq_mask_tokens"),
            max_new_tokens=data.get("max_new_tokens", 2048),
            temperature=data.get("temperature", 1.0),
            top_p=data.get("top_p", 0.9),
            top_k=data.get("top_k", 30),
            seed=data.get("seed"),
            output_codes=data.get("output_codes"),
            num_semantic_tokens=data.get("num_semantic_tokens", 0),
            audio_samples=data.get("audio_samples"),
            sample_rate=data.get("sample_rate", 44100),
        )
