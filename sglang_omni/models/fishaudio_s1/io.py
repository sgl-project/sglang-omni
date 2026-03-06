# SPDX-License-Identifier: Apache-2.0
"""FishAudio-S1 pipeline state definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FishAudioState:
    """Typed view of the per-request pipeline state for FishAudio TTS.

    Tensors are stored as nested Python lists when crossing process boundaries
    (msgpack-safe).  Use :meth:`to_dict` / :meth:`from_dict` for conversion.
    """

    # -- From preprocessing ------------------------------------------------
    input_values: Any = None  # [num_codebooks+1, seq_len] as nested list
    audio_masks: Any | None = None
    audio_parts: Any | None = None
    num_codebooks: int = 10
    codebook_size: int = 4096

    # -- Generation params -------------------------------------------------
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.8
    repetition_penalty: float = 1.1

    # -- From TTS engine ---------------------------------------------------
    output_codes: Any | None = None  # [num_codebooks+1, T] as nested list

    # -- From vocoder ------------------------------------------------------
    audio_samples: Any | None = None  # [samples] as list of floats
    sample_rate: int = 44100

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _tensor_to_list(t: Any) -> Any:
        if isinstance(t, torch.Tensor):
            return t.tolist()
        return t

    @staticmethod
    def _list_to_tensor(v: Any) -> Any:
        if isinstance(v, list):
            return torch.tensor(v)
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict with tensors serialised as nested lists."""
        data: dict[str, Any] = {}
        if self.input_values is not None:
            data["input_values"] = self._tensor_to_list(self.input_values)
        if self.audio_masks is not None:
            data["audio_masks"] = self._tensor_to_list(self.audio_masks)
        if self.audio_parts is not None:
            data["audio_parts"] = self._tensor_to_list(self.audio_parts)
        data["num_codebooks"] = self.num_codebooks
        data["codebook_size"] = self.codebook_size
        data["max_new_tokens"] = self.max_new_tokens
        data["temperature"] = self.temperature
        data["top_p"] = self.top_p
        data["repetition_penalty"] = self.repetition_penalty
        if self.output_codes is not None:
            data["output_codes"] = self._tensor_to_list(self.output_codes)
        if self.audio_samples is not None:
            data["audio_samples"] = self._tensor_to_list(self.audio_samples)
        data["sample_rate"] = self.sample_rate
        return data

    @classmethod
    def from_dict(cls, data: Any) -> FishAudioState:
        """Reconstruct from a plain dict (inverse of :meth:`to_dict`)."""
        if not isinstance(data, dict):
            data = {}
        return cls(
            input_values=data.get("input_values"),
            audio_masks=data.get("audio_masks"),
            audio_parts=data.get("audio_parts"),
            num_codebooks=data.get("num_codebooks", 10),
            codebook_size=data.get("codebook_size", 4096),
            max_new_tokens=data.get("max_new_tokens", 1024),
            temperature=data.get("temperature", 0.8),
            top_p=data.get("top_p", 0.8),
            repetition_penalty=data.get("repetition_penalty", 1.1),
            output_codes=data.get("output_codes"),
            audio_samples=data.get("audio_samples"),
            sample_rate=data.get("sample_rate", 44100),
        )
