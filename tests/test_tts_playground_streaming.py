# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the S2-Pro TTS playground streaming helpers."""

from __future__ import annotations

import base64
import json
import wave

import numpy as np

from playground.tts.audio_stream import (
    WavChunkAccumulator,
    decode_wav_chunk,
    parse_speech_stream_data,
)
from sglang_omni.client.audio import encode_wav


def _make_event(audio: np.ndarray, *, index: int = 0, sample_rate: int = 24000) -> str:
    payload = {
        "id": "speech-test",
        "object": "audio.speech.chunk",
        "index": index,
        "audio": {
            "data": base64.b64encode(encode_wav(audio, sample_rate)).decode("ascii"),
            "format": "wav",
            "mime_type": "audio/wav",
            "sample_rate": sample_rate,
        },
        "finish_reason": None,
    }
    return json.dumps(payload)


def test_parse_speech_stream_data_decodes_audio_event() -> None:
    event = parse_speech_stream_data(_make_event(np.array([0.0, 0.1], dtype=np.float32)))

    assert event is not None
    assert event.audio_bytes is not None
    assert event.sample_rate == 24000
    assert event.audio_format == "wav"


def test_parse_speech_stream_data_handles_done_marker() -> None:
    event = parse_speech_stream_data("[DONE]")

    assert event is not None
    assert event.is_done is True


def test_decode_wav_chunk_returns_audio_tuple() -> None:
    sample_rate, audio = decode_wav_chunk(
        encode_wav(np.array([0.0, 0.1, -0.1], dtype=np.float32), 24000)
    )

    assert sample_rate == 24000
    assert audio.shape == (3,)
    assert audio.dtype == np.float32


def test_wav_chunk_accumulator_writes_combined_audio() -> None:
    accumulator = WavChunkAccumulator()
    first = encode_wav(np.array([0.0, 0.1], dtype=np.float32), 24000)
    second = encode_wav(np.array([-0.1, 0.0], dtype=np.float32), 24000)

    assert accumulator.add_wav_chunk(first) == first
    assert accumulator.add_wav_chunk(second) == second

    output_path = accumulator.write_temp_wav()

    assert output_path is not None
    with wave.open(output_path, "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 4
