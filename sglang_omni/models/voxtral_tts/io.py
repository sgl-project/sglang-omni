"""Pipeline state definition for Voxtral TTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _encode_audio_codes(codes: Any) -> dict[str, Any]:
    if isinstance(codes, torch.Tensor):
        codes = codes.detach().cpu().numpy()
    array = np.asarray(codes)
    if array.size == 0:
        array = array.astype(np.uint16, copy=False)
    elif int(array.min()) >= 0 and int(array.max()) <= np.iinfo(np.uint16).max:
        array = array.astype(np.uint16, copy=False)
    else:
        array = array.astype(np.int32, copy=False)
    contiguous = np.ascontiguousarray(array)
    return {
        "audio_codes_bytes": contiguous.tobytes(),
        "audio_codes_shape": list(contiguous.shape),
        "audio_codes_dtype": str(contiguous.dtype),
    }


def _decode_audio_codes(data: dict[str, Any]) -> Any | None:
    legacy = data.get("audio_codes")
    if legacy is not None:
        if isinstance(legacy, list):
            return torch.tensor(legacy)
        return legacy

    raw = data.get("audio_codes_bytes")
    shape = data.get("audio_codes_shape")
    if raw is None or shape is None:
        return None
    dtype = np.dtype(data.get("audio_codes_dtype", "uint16"))
    array = np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.int64)
    return torch.from_numpy(array)


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
            data.update(_encode_audio_codes(self.audio_codes))
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
        return cls(
            input_ids=data.get("input_ids"),
            voice=data.get("voice"),
            max_new_tokens=data.get("max_new_tokens", 4096),
            audio_codes=_decode_audio_codes(data),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            audio_samples=data.get("audio_samples"),
            sample_rate=data.get("sample_rate", 24000),
        )
