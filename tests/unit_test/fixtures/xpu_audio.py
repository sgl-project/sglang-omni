# SPDX-License-Identifier: Apache-2.0
"""Audio bodies for the XPU lane's and baseline's speech checks."""

from __future__ import annotations

import io
import wave
from array import array


def loud_wav(frames: int, rate: int = 16000) -> bytes:
    """A mono 16-bit WAV of frames samples at half scale, so it passes every loudness check."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(array("h", [16384, -16384] * (frames // 2)).tobytes())
    return buffer.getvalue()
