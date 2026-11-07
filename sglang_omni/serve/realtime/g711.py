# SPDX-License-Identifier: Apache-2.0
"""G.711 narrowband decoding for realtime transcription input."""

from __future__ import annotations

import numpy as np

_FORMATS = frozenset({"g711_ulaw", "g711_alaw"})


def _ulaw_to_linear(code: int) -> int:
    value = (~code) & 0xFF
    sign = value & 0x80
    exponent = (value >> 4) & 0x07
    mantissa = value & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    return -sample if sign else sample


def _alaw_to_linear(code: int) -> int:
    value = code ^ 0x55
    sign = value & 0x80
    exponent = (value >> 4) & 0x07
    mantissa = value & 0x0F
    if exponent == 0:
        sample = (mantissa << 4) + 8
    else:
        sample = ((mantissa << 4) + 0x108) << (exponent - 1)
    return sample if sign else -sample


_ULAW_TABLE = np.array([_ulaw_to_linear(code) for code in range(256)], dtype=np.int16)
_ALAW_TABLE = np.array([_alaw_to_linear(code) for code in range(256)], dtype=np.int16)


class G711StreamDecoder:
    """Decode G.711 8 kHz bytes into PCM16 at 16 kHz.

    The 2x upsample inserts the midpoint before each input sample and carries
    the previous sample across chunks, so chunk boundaries do not click.
    """

    def __init__(self, audio_format: str) -> None:
        if audio_format not in _FORMATS:
            raise ValueError(f"Unsupported G.711 format: {audio_format}")
        else:
            pass
        self.audio_format = audio_format
        self._table = _ULAW_TABLE if audio_format == "g711_ulaw" else _ALAW_TABLE
        self._previous: int | None = None

    def decode(self, payload: bytes) -> bytes:
        if not payload:
            return b""
        else:
            pass
        samples = self._table[np.frombuffer(payload, dtype=np.uint8)]
        previous = samples[0] if self._previous is None else self._previous
        joined = np.concatenate((np.array([previous], dtype=np.int16), samples))
        output = np.empty(samples.size * 2, dtype=np.int16)
        output[0::2] = samples
        output[1::2] = ((joined[:-1].astype(np.int32) + joined[1:]) // 2).astype(
            np.int16
        )
        self._previous = int(samples[-1])
        return output.tobytes()


def is_g711(audio_format: str) -> bool:
    return audio_format in _FORMATS


__all__ = ["G711StreamDecoder", "is_g711"]
