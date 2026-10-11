# SPDX-License-Identifier: Apache-2.0
"""WAV request bodies to float32 samples, for every model served here."""

from __future__ import annotations

import struct

import numpy as np

SAMPLE_RATE = 16000
WAV_FORMAT_PCM = 1
WAV_FORMAT_FLOAT = 3
PCM16_FULL_SCALE = 32768.0


def decode_wav(wav_bytes: bytes) -> np.ndarray:
    """16 kHz mono PCM16 or float32 WAV to float32 samples."""
    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("audio must be a RIFF/WAVE file")
    else:
        pass
    offset = 12
    audio_format = channel_count = sample_rate = bits_per_sample = None
    while offset + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[offset : offset + 4]
        chunk_size = struct.unpack("<I", wav_bytes[offset + 4 : offset + 8])[0]
        body = wav_bytes[offset + 8 : offset + 8 + chunk_size]
        if chunk_id == b"fmt ":
            audio_format, channel_count, sample_rate = struct.unpack("<HHI", body[:8])
            bits_per_sample = struct.unpack("<H", body[14:16])[0]
        elif chunk_id == b"data":
            if (channel_count, sample_rate) != (1, SAMPLE_RATE):
                raise ValueError("audio must be 16 kHz mono")
            elif (audio_format, bits_per_sample) == (WAV_FORMAT_PCM, 16):
                return (
                    np.frombuffer(body, dtype="<i2").astype(np.float32)
                    / PCM16_FULL_SCALE
                )
            elif (audio_format, bits_per_sample) == (WAV_FORMAT_FLOAT, 32):
                return np.frombuffer(body, dtype="<f4").astype(np.float32)
            else:
                raise ValueError("audio must be PCM16 or float32 WAV")
        else:
            pass
        offset += 8 + chunk_size + (chunk_size & 1)
    raise ValueError("WAV file has no data chunk")
