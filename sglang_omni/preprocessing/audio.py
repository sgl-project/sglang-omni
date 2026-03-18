# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic audio preprocessing utilities."""

from __future__ import annotations

import asyncio
import base64
import struct
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from .base import MediaIO, _is_url
from .cache_key import compute_media_cache_key


def _decode_audio_bytes_av(data: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes using PyAV (supports WebM/Opus, MP3, OGG, FLAC, etc.)."""
    import io

    import av

    container = av.open(io.BytesIO(data))
    try:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise ValueError("No audio stream found in data")

        sample_rate = audio_stream.rate
        frames = []
        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()  # shape varies by format
            if arr.ndim == 2:
                # Planar formats (fltp, s16p, etc.): shape is (channels, samples)
                # Average channels to mono
                arr = arr.mean(axis=0)
            frames.append(arr.flatten().astype(np.float32))
    finally:
        container.close()

    if not frames:
        raise ValueError("No audio frames decoded")

    audio = np.concatenate(frames)
    # Normalize integer formats to [-1, 1] float range
    if audio.max() > 1.0 or audio.min() < -1.0:
        peak = max(abs(audio.max()), abs(audio.min()))
        if peak > 0:
            audio = audio / peak
    return audio, int(sample_rate)


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
    with open(path, "rb") as f:
        data = f.read()
    try:
        audio, sr = _parse_wav_bytes(data, source=str(path))
    except ValueError:
        audio, sr = _decode_audio_bytes_av(data)
    return _resample_linear(audio, sr, target_sr)


class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):
    """MediaIO implementation for audio files."""

    def __init__(self, *, target_sr: int = 16000, **kwargs) -> None:
        """Initialize AudioMediaIO.

        Args:
            target_sr: Target sample rate for resampling.
            **kwargs: Additional arguments (for compatibility with MultiModalResourceConnector).
        """
        super().__init__()
        self.target_sr = target_sr
        self.kwargs = kwargs

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        """Load audio from raw bytes (WAV, WebM/Opus, MP3, OGG, FLAC, etc.)."""
        try:
            audio, sr = _parse_wav_bytes(data, source="bytes")
        except ValueError:
            audio, sr = _decode_audio_bytes_av(data)
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
        """Load audio from a local file path (WAV, WebM/Opus, MP3, OGG, FLAC, etc.)."""
        with open(filepath, "rb") as f:
            data = f.read()
        try:
            audio, sr = _parse_wav_bytes(data, source=str(filepath))
        except ValueError:
            audio, sr = _decode_audio_bytes_av(data)
        resampled = _resample_linear(audio, sr, self.target_sr)
        return resampled, float(self.target_sr)


async def ensure_audio_list_async(
    audios: Any,
    *,
    target_sr: int = 16000,
    resource_connector: Any | None = None,
) -> list[Any]:
    """Asynchronously normalize audio inputs into a list.

    Args:
        audios: Audio input(s) - can be a path, URL, numpy array, or list.
        target_sr: Target sample rate for resampling.
        media_connector: Optional MultiModalResourceConnector instance. If None, uses
                        the global connector.

    Returns:
        List of normalized audio arrays.
    """
    if audios is None:
        return []
    items = audios if isinstance(audios, list) else [audios]

    # Import here to avoid circular dependency
    if resource_connector is None:
        from .resource_connector import get_global_resource_connector

        resource_connector = get_global_resource_connector()

    # Collect coroutines for URL items
    coroutines: list[asyncio.Task[tuple[npt.NDArray, float]] | None] = []
    url_indices: list[int] = []
    normalized: list[Any] = []

    # First pass: identify URL items and create coroutines
    for idx, item in enumerate(items):
        if isinstance(item, (str, Path)):
            if _is_url(item):
                # Create coroutine for async URL fetching
                coro = resource_connector.fetch_audio_async(
                    str(item), target_sr=target_sr
                )
                task = asyncio.create_task(coro)
                coroutines.append(task)
                url_indices.append(idx)
                normalized.append(None)  # Placeholder
            else:
                # Local path - can be loaded synchronously
                normalized.append(load_audio_path(item, target_sr=target_sr))
        else:
            # Already processed (numpy array, etc.)
            normalized.append(item)

    # Wait for all URL fetches to complete
    if coroutines:
        results = await asyncio.gather(*coroutines)
        # Fill in the results at the correct indices (extract audio array, ignore sample rate)
        for url_idx, (audio, _) in zip(url_indices, results):
            normalized[url_idx] = audio

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
