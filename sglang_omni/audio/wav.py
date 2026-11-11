# SPDX-License-Identifier: Apache-2.0
"""WAV containers for raw PCM sent to services that take audio files."""

import io
import wave


def pcm16_wav(pcm: bytes, sample_rate_hz: int) -> bytes:
    """A mono WAV file holding signed 16-bit little-endian PCM."""
    container = io.BytesIO()
    with wave.open(container, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(pcm)
    return container.getvalue()
