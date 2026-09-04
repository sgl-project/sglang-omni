# SPDX-License-Identifier: Apache-2.0
"""Shared audio utilities."""

from __future__ import annotations

import functools
import io
import math
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
import numpy as np
import pybase64
import torch
import torchaudio
import xxhash

from sglang_omni.platforms import current_platform

_DEFAULT_REQUEST_TIMEOUT = 5


class AudioDecodeError(ValueError):
    """Raised when supplied encoded audio cannot be decoded."""


def _ensure_torchaudio_decoder_ready() -> None:
    from torchcodec.decoders import AudioDecoder

    del AudioDecoder


def _has_operational_decoder_cause(exc: BaseException) -> bool:
    current = exc.__cause__ or exc.__context__
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, torch.OutOfMemoryError):
            return True
        if not isinstance(current, (RuntimeError, ValueError)):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_invalid_audio_source(source: bytes | str) -> bool:
    try:
        import av
    except (ImportError, RuntimeError):
        return False

    candidate = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        container = av.open(candidate)
    except (av.error.InvalidDataError, av.error.EOFError, ValueError):
        return True
    except Exception:
        return False

    try:
        audio_stream = next(
            (stream for stream in container.streams if stream.type == "audio"),
            None,
        )
        if audio_stream is None:
            return True
        try:
            decoded_frame = False
            for _frame in container.decode(audio_stream):
                decoded_frame = True
            return not decoded_frame
        except (av.error.InvalidDataError, av.error.EOFError, ValueError):
            return True
        except Exception:
            return False
    finally:
        container.close()


def _load_with_torchaudio(
    source: bytes | str, *, source_name: str
) -> tuple[torch.Tensor, int]:
    _ensure_torchaudio_decoder_ready()
    decoder_source = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        return torchaudio.load(decoder_source)
    except (ImportError, MemoryError, torch.OutOfMemoryError):
        raise
    except RuntimeError as exc:
        if _has_operational_decoder_cause(exc) or not _is_invalid_audio_source(source):
            raise
        raise AudioDecodeError(f"Could not decode {source_name} audio input") from exc


def _is_riff_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _resample_with_scipy(
    audio_np: np.ndarray, sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    import scipy.signal

    orig_freq = int(sample_rate)
    new_freq = target_sample_rate
    gcd = math.gcd(orig_freq, new_freq)
    up = new_freq // gcd
    down = orig_freq // gcd
    resampled_np = scipy.signal.resample_poly(audio_np, up, down, axis=-1)
    return resampled_np.astype(np.float32)


def _try_fast_wav_decode(
    data: bytes,
    target_sample_rate: int,
    resample_kwargs: Mapping[str, Any] | None = None,
) -> np.ndarray | None:
    # Note (akazaakane): Keep unsupported WAV encodings on torchaudio so the fast
    # path never narrows existing format coverage.
    from sglang_omni.preprocessing.audio import _parse_wav_bytes

    try:
        audio, sample_rate = _parse_wav_bytes(data)
    except ValueError:
        return None
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not audio.flags.writeable:
        audio = audio.copy()
    if sample_rate == target_sample_rate:
        return audio

    if current_platform.supports_torchaudio_resample():
        resampled = _cached_resample(
            torch.from_numpy(audio),
            sample_rate,
            target_sample_rate,
            resample_kwargs,
        )
        return resampled.numpy()
    else:
        return _resample_with_scipy(audio, sample_rate, target_sample_rate)


@functools.lru_cache(maxsize=32)
def _resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kwargs_items: tuple[tuple[str, Any], ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    # Note (Jiaxin Deng): torchaudio rebuilds this per call even though it
    # depends only on the rate pair, the options and the tensor type.
    return torchaudio.functional.functional._get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        device=device,
        dtype=dtype,
        **dict(kwargs_items),
    )


def _cached_resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
    resample_kwargs: Mapping[str, Any] | None,
) -> torch.Tensor:
    kwargs = dict(resample_kwargs or {})
    orig_freq, new_freq = int(orig_freq), int(new_freq)
    try:
        gcd = math.gcd(orig_freq, new_freq)
        kernel, width = _resample_kernel(
            orig_freq,
            new_freq,
            gcd,
            tuple(sorted(kwargs.items())),
            waveform.device,
            waveform.dtype,
        )
        return torchaudio.functional.functional._apply_sinc_resample_kernel(
            waveform, orig_freq, new_freq, gcd, kernel, width
        )
    except (AttributeError, TypeError, RuntimeError):
        return torchaudio.functional.resample(waveform, orig_freq, new_freq, **kwargs)


def decode_audio_data_uri(value: str) -> bytes | None:
    if not value.startswith("data:"):
        return None
    header, separator, payload = value.partition(",")
    if not separator or ";base64" not in header.lower() or not payload:
        raise AudioDecodeError("Invalid base64 audio data URI")
    try:
        return pybase64.b64decode(payload, validate=True)
    except Exception as exc:
        raise AudioDecodeError("Invalid base64 audio data URI") from exc


def load_audio(
    source: Any,
    source_name: str = "audio",
    target_sample_rate: int = 16000,
    mono: bool = True,
    trim_top_db: float | None = None,
    resample_kwargs: Mapping[str, Any] | None = None,
) -> np.ndarray:
    if isinstance(source, memoryview):
        source = source.tobytes()
    if isinstance(source, bytearray):
        source = bytes(source)
    if isinstance(source, str):
        decoded = decode_audio_data_uri(source)
        if decoded is not None:
            source = decoded
        elif source.startswith(("http://", "https://")):
            try:
                timeout = int(
                    os.getenv("REQUEST_TIMEOUT", str(_DEFAULT_REQUEST_TIMEOUT))
                )
                if timeout <= 0:
                    timeout = _DEFAULT_REQUEST_TIMEOUT
            except ValueError:
                timeout = _DEFAULT_REQUEST_TIMEOUT
            response = httpx.get(source, timeout=timeout, follow_redirects=True)
            response.raise_for_status()
            source = response.content
        elif source.startswith("file://"):
            source = unquote(urlparse(source).path)

    if isinstance(source, bytes):
        # Note (akazaakane): The direct WAV/NumPy path avoids torchaudio decoder
        # startup when mono=True without changing channel-preserving loads.
        if mono and trim_top_db is None and _is_riff_wav(source):
            fast = _try_fast_wav_decode(
                source, target_sample_rate, resample_kwargs=resample_kwargs
            )
            if fast is not None:
                return fast
        audio, sample_rate = _load_with_torchaudio(source, source_name=source_name)
    elif isinstance(source, str):
        audio, sample_rate = _load_with_torchaudio(source, source_name=source_name)
    else:
        raise ValueError(
            f"Unsupported {source_name} audio input: {type(source).__name__}"
        )

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if mono and audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.to(torch.float32)
    if trim_top_db is not None:
        import librosa

        trimmed, _ = librosa.effects.trim(audio.numpy(), top_db=trim_top_db)
        audio = torch.from_numpy(trimmed)
    if sample_rate != target_sample_rate:
        if current_platform.supports_torchaudio_resample():
            audio = torchaudio.functional.resample(
                audio,
                int(sample_rate),
                target_sample_rate,
                **dict(resample_kwargs or {}),
            )
        else:
            waveform_np = audio.cpu().numpy()
            resampled_np = _resample_with_scipy(
                waveform_np, int(sample_rate), target_sample_rate
            )
            audio = torch.from_numpy(resampled_np).float()
    if mono:
        audio = audio.squeeze(0)
    return audio.cpu().numpy()


def audio_fingerprint(audio: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(audio, dtype=np.float32)
    return xxhash.xxh3_128_hexdigest(contiguous)


def audio_fingerprint_int(fingerprint: str) -> int:
    return int(fingerprint[:16], 16)
