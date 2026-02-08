# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic audio frontend utilities."""

from __future__ import annotations

import base64
import struct
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from .base import MediaIO, _is_url
from .cache_key import compute_media_cache_key


def _parse_wav_bytes(data: bytes, source: str = "bytes") -> tuple[np.ndarray, int]:
    """Parse PCM/IEEE-float WAV from bytes without external deps."""
    if len(data) < 12:
        raise ValueError(f"Invalid WAV header: {source}")

    header = data[:12]
    riff, _, wave = struct.unpack("<4sI4s", header)
    if riff != b"RIFF" or wave != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {source}")

    fmt_tag = None
    channels = None
    sample_rate = None
    bits_per_sample = None
    data_bytes = b""

    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            break
        chunk_header = data[offset : offset + 8]
        chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
        offset += 8

        if offset + chunk_size > len(data):
            break
        chunk_data = data[offset : offset + chunk_size]
        offset += chunk_size
        if chunk_size % 2 == 1:
            offset += 1

        if chunk_id == b"fmt ":
            if len(chunk_data) >= 16:
                fmt_tag, channels, sample_rate, _, _, bits_per_sample = struct.unpack(
                    "<HHIIHH", chunk_data[:16]
                )
        elif chunk_id == b"data":
            data_bytes = chunk_data

    if fmt_tag is None or sample_rate is None or bits_per_sample is None:
        raise ValueError(f"Missing fmt chunk in WAV: {source}")
    if not data_bytes:
        raise ValueError(f"Missing data chunk in WAV: {source}")

    if fmt_tag == 3:  # IEEE float
        if bits_per_sample == 32:
            audio = np.frombuffer(data_bytes, dtype="<f4")
        elif bits_per_sample == 64:
            audio = np.frombuffer(data_bytes, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(f"Unsupported float WAV bit depth: {bits_per_sample}")
    elif fmt_tag == 1:  # PCM
        if bits_per_sample == 16:
            audio_i16 = np.frombuffer(data_bytes, dtype="<i2")
            audio = (audio_i16.astype(np.float32) / 32768.0).astype(np.float32)
        elif bits_per_sample == 32:
            audio_i32 = np.frombuffer(data_bytes, dtype="<i4")
            audio = (audio_i32.astype(np.float32) / 2147483648.0).astype(np.float32)
        elif bits_per_sample == 8:
            audio_u8 = np.frombuffer(data_bytes, dtype="u1")
            audio = ((audio_u8.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported PCM WAV bit depth: {bits_per_sample}")
    else:
        raise ValueError(f"Unsupported WAV format tag: {fmt_tag}")

    if channels and channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio.astype(np.float32, copy=False), int(sample_rate)


def _read_wav_bytes(path: str) -> tuple[np.ndarray, int]:
    """Read PCM/IEEE-float WAV from file path without external deps."""
    with open(path, "rb") as f:
        data = f.read()
    return _parse_wav_bytes(data, source=path)


def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.shape[0] / float(orig_sr)
    new_len = max(int(round(duration * target_sr)), 1)
    old_idx = np.arange(audio.shape[0], dtype=np.float64)
    new_idx = np.linspace(0.0, audio.shape[0] - 1, num=new_len, dtype=np.float64)
    return np.interp(new_idx, old_idx, audio).astype(np.float32)


def load_audio_path(path: str | Path, *, target_sr: int = 16000) -> np.ndarray:
    audio, sr = _read_wav_bytes(str(path))
    return _resample_linear(audio, sr, target_sr)


class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):
    """MediaIO implementation for audio files."""

    def __init__(self, *, target_sr: int = 16000, **kwargs) -> None:
        """Initialize AudioMediaIO.

        Args:
            target_sr: Target sample rate for resampling.
            **kwargs: Additional arguments (for compatibility with MediaConnector).
        """
        super().__init__()
        self.target_sr = target_sr
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        """Load audio from raw bytes."""
        audio, sr = _parse_wav_bytes(data, source="bytes")
        resampled = _resample_linear(audio, sr, self.target_sr)
        return resampled, float(self.target_sr)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        """Load audio from base64-encoded data."""
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        """Load audio from a local file path."""
        audio, sr = _read_wav_bytes(str(filepath))
        resampled = _resample_linear(audio, sr, self.target_sr)
        return resampled, float(self.target_sr)


def ensure_audio_list(
    audios: Any,
    *,
    target_sr: int = 16000,
    media_connector: Any | None = None,
) -> list[Any]:
    """Normalize audio inputs into a list.

    Supports local file paths, URLs (HTTP/HTTPS, data URLs, file URLs),
    and numpy arrays.

    Args:
        audios: Audio input(s) - can be a path, URL, numpy array, or list.
        target_sr: Target sample rate for resampling.
        media_connector: Optional MediaConnector instance. If None, uses
                        the global connector.

    Returns:
        List of normalized audio arrays.
    """
    if audios is None:
        return []
    items = audios if isinstance(audios, list) else [audios]
    normalized: list[Any] = []

    # Import here to avoid circular dependency
    if media_connector is None:
        from .media_connector import get_global_media_connector

        media_connector = get_global_media_connector()

    for item in items:
        if isinstance(item, (str, Path)):
            if _is_url(item):
                # Load from URL
                audio, _ = media_connector.fetch_audio(str(item), target_sr=target_sr)
                normalized.append(audio)
            else:
                # Load from local path
                normalized.append(load_audio_path(item, target_sr=target_sr))
        else:
            # Already processed (numpy array, etc.)
            normalized.append(item)
    return normalized


def build_audio_mm_inputs(hf_inputs: dict[str, Any]) -> dict[str, Any]:
    """Extract standard audio tensors from HF processor outputs."""
    feature_attention_mask = hf_inputs.get("feature_attention_mask")
    audio_feature_lengths = hf_inputs.get("audio_feature_lengths")
    if audio_feature_lengths is None and isinstance(
        feature_attention_mask, torch.Tensor
    ):
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1).to(
            dtype=torch.long
        )
    return {
        "input_features": hf_inputs.get("input_features"),
        "feature_attention_mask": feature_attention_mask,
        "audio_feature_lengths": audio_feature_lengths,
    }


def compute_audio_cache_key(audios: Any) -> str | None:
    """Compute cache key from raw audio inputs (paths, numpy arrays).

    This should be called BEFORE ensure_audio_list() to capture original
    paths which are much cheaper to hash than audio data.
    """
    return compute_media_cache_key(audios, prefix="audio")
