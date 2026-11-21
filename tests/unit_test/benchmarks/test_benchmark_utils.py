# SPDX-License-Identifier: Apache-2.0
"""WAV duration of benchmark audio responses."""

import io
import struct
import wave

import pytest

from benchmarks.benchmarker.utils import get_wav_duration

SAMPLE_RATE = 24000
PCM = struct.pack("<h", 1000) * (SAMPLE_RATE * 2)
LIST_CHUNK = (
    b"LIST" + struct.pack("<I", 16) + b"INFOISFT" + struct.pack("<I", 4) + b"test"
)


def build_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(PCM)
    return buf.getvalue()


def insert_chunk(wav_bytes: bytes, offset: int, chunk: bytes) -> bytes:
    body = wav_bytes[8:offset] + chunk + wav_bytes[offset:]
    return b"RIFF" + struct.pack("<I", len(body)) + body


WAV = build_wav()


@pytest.mark.parametrize(
    "wav_bytes",
    [
        WAV,
        insert_chunk(WAV, 36, LIST_CHUNK),
        insert_chunk(WAV, len(WAV), LIST_CHUNK),
        WAV[:40] + struct.pack("<I", 0xFFFFFFFF) + WAV[44:],
    ],
    ids=["plain", "list_before_data", "list_after_data", "placeholder_data_size"],
)
def test_duration_counts_only_the_data_chunk(wav_bytes: bytes) -> None:
    assert get_wav_duration(wav_bytes) == 2.0


@pytest.mark.parametrize("audio_bytes", [PCM, b"fLaC" + PCM], ids=["raw_pcm", "flac"])
def test_bytes_that_are_not_wav_have_no_duration(audio_bytes: bytes) -> None:
    assert get_wav_duration(audio_bytes) == 0.0
