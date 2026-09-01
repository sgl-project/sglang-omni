"""Small torchaudio compatibility layer for MUSA smoke deployments.

This is not a full torchaudio replacement. It covers the subset exercised by
the current sglang-omni TTS pipelines when a native MUSA torchaudio wheel is
not available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from . import compliance, functional, transforms

__version__ = "2.9.0+musa-shim"


@dataclass(frozen=True)
class AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int = 1
    bits_per_sample: int = 16
    encoding: str = "PCM_S"


def load(uri, *_, **__) -> tuple[torch.Tensor, int]:
    from sglang_omni.preprocessing.audio import _decode_audio_bytes_av
    from sglang_omni.preprocessing.audio import _parse_wav_bytes

    with open(Path(uri).expanduser(), "rb") as f:
        data = f.read()
    try:
        audio, sample_rate = _parse_wav_bytes(data, source=str(uri))
    except ValueError:
        audio, sample_rate = _decode_audio_bytes_av(data)
    return torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1), int(sample_rate)


def info(uri, *_, **__) -> AudioMetaData:
    audio, sample_rate = load(uri)
    return AudioMetaData(
        sample_rate=int(sample_rate),
        num_frames=int(audio.shape[-1]),
        num_channels=int(audio.shape[0]),
    )


__all__ = ["AudioMetaData", "compliance", "functional", "info", "load", "transforms"]
