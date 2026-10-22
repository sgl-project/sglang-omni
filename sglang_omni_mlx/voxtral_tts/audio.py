# SPDX-License-Identifier: Apache-2.0
"""WAV encoding for Voxtral speech output."""

from __future__ import annotations

import io
import wave

import numpy as np

PCM16_FULL_SCALE = 32768.0


def encode_wav(samples: np.ndarray, sample_rate_hz: int) -> bytes:
    """Encode normalized mono samples as 16-bit PCM WAV audio."""
    if samples.ndim != 1:
        raise ValueError("WAV output must be mono")
    else:
        pass
    if not np.isfinite(samples).all():
        raise ValueError("WAV output samples must be finite")
    else:
        pass
    pcm_samples = np.clip(
        np.rint(samples * PCM16_FULL_SCALE), -PCM16_FULL_SCALE, PCM16_FULL_SCALE - 1
    ).astype("<i2")
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wav_writer:
        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(sample_rate_hz)
        wav_writer.writeframes(pcm_samples.tobytes())
    return audio_buffer.getvalue()
