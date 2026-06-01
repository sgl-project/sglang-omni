# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the S2-Pro TTS playground streaming helpers."""

from __future__ import annotations

import base64
import io
import json
import wave

import numpy as np

from playground.tts.audio_stream import (
    BufferedWavChunkEmitter,
    WavChunkAccumulator,
    decode_wav_chunk,
    parse_speech_stream_data,
    wav_duration_seconds,
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
    event = parse_speech_stream_data(
        _make_event(np.array([0.0, 0.1], dtype=np.float32))
    )

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

    output_bytes = accumulator.to_wav_bytes()

    assert output_bytes is not None
    with wave.open(io.BytesIO(output_bytes), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 4


def test_wav_duration_seconds_reads_wav_length() -> None:
    audio_bytes = encode_wav(np.zeros(2400, dtype=np.float32), 24000)

    assert wav_duration_seconds(audio_bytes) == 0.1


def test_buffered_wav_chunk_emitter_combines_short_live_chunks() -> None:
    emitter = BufferedWavChunkEmitter(
        min_emit_duration_s=1.0,
        max_buffered_chunks=3,
    )
    chunk = encode_wav(np.zeros(2400, dtype=np.float32), 24000)

    assert emitter.add_wav_chunk(chunk) is None
    assert emitter.add_wav_chunk(chunk) is None

    emitted = emitter.add_wav_chunk(chunk)

    assert emitted is not None
    with wave.open(io.BytesIO(emitted), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 7200


def test_buffered_wav_chunk_emitter_flushes_remaining_audio() -> None:
    emitter = BufferedWavChunkEmitter(
        min_emit_duration_s=1.0,
        max_buffered_chunks=3,
    )
    chunk = encode_wav(np.zeros(2400, dtype=np.float32), 24000)

    assert emitter.add_wav_chunk(chunk) is None

    emitted = emitter.flush()

    assert emitted is not None
    with wave.open(io.BytesIO(emitted), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 2400
