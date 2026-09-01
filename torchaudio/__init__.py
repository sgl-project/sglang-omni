"""Small torchaudio compatibility layer for MUSA smoke deployments.

This is not a full torchaudio replacement. It covers the subset exercised by
the current sglang-omni TTS pipelines when a native MUSA torchaudio wheel is
not available.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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


def _read_audio_source(uri: Any) -> bytes:
    if isinstance(uri, (bytes, bytearray, memoryview)):
        return bytes(uri)
    if hasattr(uri, "read"):
        data = uri.read()
        return bytes(data)
    with open(Path(uri).expanduser(), "rb") as f:
        return f.read()


def _load_wav_bytes(data: bytes) -> tuple[torch.Tensor, int]:
    with wave.open(io.BytesIO(data), "rb") as wav_file:
        sample_rate = int(wav_file.getframerate())
        num_channels = int(wav_file.getnchannels())
        sample_width = int(wav_file.getsampwidth())
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        audio = torch.from_numpy(np.frombuffer(frames, dtype=np.uint8).copy())
        audio = audio.to(torch.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = torch.from_numpy(np.frombuffer(frames, dtype="<i2").copy())
        audio = audio.to(torch.float32)
        audio = audio / 32768.0
    else:
        raise ValueError("unsupported WAV sample width")

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).transpose(0, 1).contiguous()
    else:
        audio = audio.reshape(1, -1)
    return audio, sample_rate


def load(uri, *_, **__) -> tuple[torch.Tensor, int]:
    data = _read_audio_source(uri)
    try:
        audio, sample_rate = _load_wav_bytes(data)
    except Exception:
        try:
            import soundfile as sf
        except Exception as exc:
            raise ValueError("torchaudio shim only supports WAV input without soundfile") from exc
        audio_np, sample_rate = sf.read(io.BytesIO(data), always_2d=True, dtype="float32")
        audio = torch.from_numpy(audio_np).transpose(0, 1).contiguous()
    return audio.to(dtype=torch.float32), int(sample_rate)


def info(uri, *_, **__) -> AudioMetaData:
    audio, sample_rate = load(uri)
    return AudioMetaData(
        sample_rate=int(sample_rate),
        num_frames=int(audio.shape[-1]),
        num_channels=int(audio.shape[0]),
    )


__all__ = ["AudioMetaData", "compliance", "functional", "info", "load", "transforms"]
