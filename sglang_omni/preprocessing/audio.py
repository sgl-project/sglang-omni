# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic audio preprocessing utilities."""

from __future__ import annotations

import asyncio
import base64
import io
import struct
from collections.abc import Mapping
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt
import soundfile
import torch

from sglang_omni.preprocessing.resource_connector import MultiModalResourceConnector

from .base import MediaIO, is_url
from .resource_connector import await_media_cleanup


def decode_audio_bytes_av(data: bytes) -> tuple[npt.NDArray[np.float32], int]:
    """Decode audio bytes with PyAV (WebM/Opus, MP3, OGG, FLAC, etc.) to mono float at full scale."""
    container = av.open(io.BytesIO(data))
    try:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise ValueError("No audio stream found in data")
        else:
            pass

        sample_rate = audio_stream.rate
        frames = []
        for frame in container.decode(audio_stream):
            samples = frame.to_ndarray()
            if np.issubdtype(samples.dtype, np.unsignedinteger):
                midpoint = (np.iinfo(samples.dtype).max + 1) / 2
                samples = (samples.astype(np.float32) - midpoint) / midpoint
            elif np.issubdtype(samples.dtype, np.signedinteger):
                full_scale = -float(np.iinfo(samples.dtype).min)
                samples = samples.astype(np.float32) / full_scale
            else:
                samples = samples.astype(np.float32, copy=False)
            channels = len(frame.layout.channels)
            # note (ratish): a packed frame interleaves its channels in one row.
            if frame.format.is_planar:
                samples = samples.reshape(channels, -1).mean(axis=0)
            else:
                samples = samples.reshape(-1, channels).mean(axis=1)
            frames.append(samples)
    finally:
        container.close()

    if not frames:
        raise ValueError("No audio frames decoded")
    else:
        pass

    return np.concatenate(frames), int(sample_rate)


def decode_audio_bytes(data: bytes) -> tuple[npt.NDArray[np.float32], int]:
    """Decode audio bytes to mono float as the reference loader does: libsndfile, then
    PyAV for the containers libsndfile does not read (MP4, WebM, AAC)."""
    try:
        audio, sample_rate = soundfile.read(
            io.BytesIO(data), dtype="float32", always_2d=True
        )
    except soundfile.LibsndfileError:
        return decode_audio_bytes_av(data)
    if len(audio) > 0:
        return audio.mean(axis=1), int(sample_rate)
    else:
        # note (ratish): libsndfile reads the frames a WAV header states, which a writer
        # that could not seek back leaves at 0; FFmpeg reads to the end of the data.
        return decode_audio_bytes_av(data)


def parse_wav_bytes(
    data: bytes, source: str = "bytes"
) -> tuple[npt.NDArray[np.float32], int]:
    """Parse PCM/IEEE-float WAV from bytes without external deps."""
    if len(data) < 12:
        raise ValueError(f"Invalid WAV header: {source}")
    else:
        pass

    header = data[:12]
    riff, _, wave = struct.unpack("<4sI4s", header)
    if riff != b"RIFF" or wave != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {source}")
    else:
        pass

    fmt_tag = None
    channels = None
    sample_rate = None
    bits_per_sample = None
    data_bytes = b""

    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            break
        else:
            pass
        chunk_header = data[offset : offset + 8]
        chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
        offset += 8

        if offset + chunk_size > len(data):
            break
        else:
            pass
        chunk_data = data[offset : offset + chunk_size]
        offset += chunk_size
        if chunk_size % 2 == 1:
            offset += 1
        else:
            pass

        if chunk_id == b"fmt ":
            if len(chunk_data) >= 16:
                fmt_tag, channels, sample_rate, _, _, bits_per_sample = struct.unpack(
                    "<HHIIHH", chunk_data[:16]
                )
            else:
                pass
        elif chunk_id == b"data":
            data_bytes = chunk_data
        else:
            pass

    if fmt_tag is None or sample_rate is None or bits_per_sample is None:
        raise ValueError(f"Missing fmt chunk in WAV: {source}")
    else:
        pass
    if not data_bytes:
        raise ValueError(f"Missing data chunk in WAV: {source}")
    else:
        pass

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
    else:
        pass

    return audio.astype(np.float32, copy=False), int(sample_rate)


def resample_linear(
    audio: np.ndarray, orig_sr: int, target_sr: int
) -> npt.NDArray[np.float32]:
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    else:
        pass
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    else:
        pass
    duration = audio.shape[0] / float(orig_sr)
    new_len = max(int(round(duration * target_sr)), 1)
    old_idx = np.arange(audio.shape[0], dtype=np.float64)
    new_idx = np.linspace(0.0, audio.shape[0] - 1, num=new_len, dtype=np.float64)
    return np.interp(new_idx, old_idx, audio).astype(np.float32)


def load_audio_path(
    path: str | Path, *, target_sr: int = 16000
) -> npt.NDArray[np.float32]:
    with open(path, "rb") as f:
        data = f.read()
    audio, sample_rate = decode_audio_bytes(data)
    return resample_linear(audio, sample_rate, target_sr)


class AudioMediaIO(MediaIO[tuple[npt.NDArray[np.float32], float]]):
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

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray[np.float32], float]:
        """Load audio from raw bytes (WAV, WebM/Opus, MP3, OGG, FLAC, etc.)."""
        audio, sample_rate = decode_audio_bytes(data)
        resampled = resample_linear(audio, sample_rate, self.target_sr)
        return resampled, float(self.target_sr)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray[np.float32], float]:
        """Load audio from base64-encoded data."""
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray[np.float32], float]:
        """Load audio from a local file path (WAV, WebM/Opus, MP3, OGG, FLAC, etc.)."""
        with open(filepath, "rb") as f:
            data = f.read()
        audio, sample_rate = decode_audio_bytes(data)
        resampled = resample_linear(audio, sample_rate, self.target_sr)
        return resampled, float(self.target_sr)


async def ensure_audio_list_async(
    audios: object,
    *,
    target_sr: int = 16000,
    resource_connector: MultiModalResourceConnector | None = None,
) -> list[object]:
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
    else:
        pass
    items = audios if isinstance(audios, list) else [audios]

    # Import here to avoid circular dependency
    if resource_connector is None:
        from .resource_connector import get_global_resource_connector

        resource_connector = get_global_resource_connector()
    else:
        pass

    # Collect coroutines for URL items
    coroutines: list[asyncio.Task[tuple[npt.NDArray[np.float32], float]]] = []
    url_indices: list[int] = []
    normalized: list[object] = []

    try:
        # note (Teery): First pass: identify URL items and create coroutines
        for idx, item in enumerate(items):
            if isinstance(item, (str, Path)):
                if is_url(item):
                    # note (Teery): Create coroutine for async URL fetching
                    coro = resource_connector.fetch_audio_async(
                        str(item), target_sr=target_sr
                    )
                    task = asyncio.create_task(coro)
                    coroutines.append(task)
                    url_indices.append(idx)
                    normalized.append(None)  # note (Teery): Placeholder
                else:
                    # note (Teery): Local path - can be loaded synchronously
                    path = resource_connector.local_media_path(item)
                    normalized.append(load_audio_path(path, target_sr=target_sr))
            else:
                # note (Teery): Already processed (numpy array, etc.)
                normalized.append(item)

        # note (Teery): Wait for all URL fetches to complete
        if coroutines:
            results = await asyncio.gather(*coroutines)
            # note (Teery): Fill in the results at the correct indices (extract audio array, ignore sample rate)
            for url_idx, (audio, _) in zip(url_indices, results):
                normalized[url_idx] = audio
        else:
            pass
    finally:
        for task in coroutines:
            if not task.done():
                task.cancel()
            else:
                pass

        async def cleanup_loaders() -> None:
            await asyncio.gather(*coroutines, return_exceptions=True)

        await await_media_cleanup(cleanup_loaders())

    return normalized


def build_audio_mm_inputs(
    hf_inputs: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor | None]:
    """Extract standard audio tensors from HF processor outputs."""
    feature_attention_mask = hf_inputs.get("feature_attention_mask")
    audio_feature_lengths = hf_inputs.get("audio_feature_lengths")
    if audio_feature_lengths is None and isinstance(
        feature_attention_mask, torch.Tensor
    ):
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1).to(
            dtype=torch.long
        )
    else:
        pass
    return {
        "input_features": hf_inputs.get("input_features"),
        "feature_attention_mask": feature_attention_mask,
        "audio_feature_lengths": audio_feature_lengths,
    }


def compute_audio_cache_key(audios: object) -> str | None:
    """Compute a cache key from loaded audio waveforms.

    Pass decoded waveforms, such as the output of ensure_audio_list_async. A URL
    or path can name different samples over time, so it is not a stable key.
    """
    from .cache_key import compute_media_cache_key

    return compute_media_cache_key(audios, prefix="audio")
