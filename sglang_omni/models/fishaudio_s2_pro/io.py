# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro pipeline state definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class S2ProState:
    """Per-request pipeline state for S2-Pro TTS."""

    # -- From preprocessing ------------------------------------------------
    input_ids: Any = None  # [seq_len] as list
    vq_mask_tokens: Any | None = None  # [seq_len] bool as list
    vq_parts: Any | None = None  # list of [num_codebooks, T_i] as nested lists
    num_codebooks: int = 10
    codebook_size: int = 4096

    # -- Generation params -------------------------------------------------
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    repetition_penalty: float = 1.1

    # -- From TTS engine ---------------------------------------------------
    output_codes: Any | None = None  # [num_codebooks+1, T] as nested list

    # -- From vocoder ------------------------------------------------------
    audio_samples: Any | None = None
    sample_rate: int = 44100

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _tensor_to_list(t: Any) -> Any:
        if isinstance(t, torch.Tensor):
            return t.tolist()
        return t

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.input_ids is not None:
            data["input_ids"] = self._tensor_to_list(self.input_ids)
        if self.vq_mask_tokens is not None:
            data["vq_mask_tokens"] = self._tensor_to_list(self.vq_mask_tokens)
        if self.vq_parts is not None:
            data["vq_parts"] = [self._tensor_to_list(p) for p in self.vq_parts]
        data["num_codebooks"] = self.num_codebooks
        data["codebook_size"] = self.codebook_size
        data["max_new_tokens"] = self.max_new_tokens
        data["temperature"] = self.temperature
        data["top_p"] = self.top_p
        data["top_k"] = self.top_k
        data["repetition_penalty"] = self.repetition_penalty
        if self.output_codes is not None:
            data["output_codes"] = self._tensor_to_list(self.output_codes)
        if self.audio_samples is not None:
            data["audio_samples"] = self._tensor_to_list(self.audio_samples)
        data["sample_rate"] = self.sample_rate
        return data

    @classmethod
    def from_dict(cls, data: dict) -> S2ProState:
        vq_parts = data.get("vq_parts")
        if vq_parts is not None:
            vq_parts = [torch.tensor(p) if isinstance(p, list) else p for p in vq_parts]
        return cls(
            input_ids=data.get("input_ids"),
            vq_mask_tokens=data.get("vq_mask_tokens"),
            vq_parts=vq_parts,
            num_codebooks=data.get("num_codebooks", 10),
            codebook_size=data.get("codebook_size", 4096),
            max_new_tokens=data.get("max_new_tokens", 1024),
            temperature=data.get("temperature", 0.8),
            top_p=data.get("top_p", 0.8),
            top_k=data.get("top_k", 30),
            repetition_penalty=data.get("repetition_penalty", 1.1),
            output_codes=(
                torch.tensor(data["output_codes"]) if "output_codes" in data else None
            ),
            audio_samples=data.get("audio_samples"),
            sample_rate=data.get("sample_rate", 44100),
        )
