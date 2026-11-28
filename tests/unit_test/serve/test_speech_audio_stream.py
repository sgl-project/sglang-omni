# SPDX-License-Identifier: Apache-2.0
"""Speech audio streams keep a producer's channels only when both sides ask."""

from __future__ import annotations

import struct

import numpy as np

from sglang_omni.client.audio import encode_wav
from sglang_omni.client.types import GenerateChunk
from sglang_omni.serve.speech_streaming import (
    SpeechAudioStream,
    create_streaming_wav_header,
)


def stereo_pieces() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [
        (rng.random((2, samples)) * 2 - 1).astype(np.float32)
        for samples in (6400, 6720, 320, 9601)
    ]


def stream_bytes(stream: SpeechAudioStream, pieces, *, keep: bool) -> bytes:
    body = b""
    for piece in pieces:
        chunk = GenerateChunk(
            request_id="song",
            audio_data=piece,
            sample_rate=32000,
            modality="audio",
            keep_audio_channels=keep,
        )
        body += stream.push(chunk) or b""
    return body + (stream.finish() or b"")


def test_marked_stereo_stream_carries_the_full_wav_samples() -> None:
    pieces = stereo_pieces()
    stream = SpeechAudioStream(
        output_sample_rate=None, speed=1.0, allow_native_channels=True
    )

    body = stream_bytes(stream, pieces, keep=True)

    full_wav = encode_wav(np.concatenate(pieces, axis=-1), 32000)
    assert stream.channels == 2
    assert body == full_wav[44:]
    header = create_streaming_wav_header(32000, stream.channels)
    assert header[:4] == full_wav[:4] and header[8:40] == full_wav[8:40]
    assert struct.unpack("<H", header[22:24]) == (2,)


def test_unmarked_or_pcm_streams_stay_mono() -> None:
    pieces = stereo_pieces()
    mono = SpeechAudioStream(output_sample_rate=None, speed=1.0)
    expected = stream_bytes(mono, pieces, keep=False)

    unmarked = SpeechAudioStream(
        output_sample_rate=None, speed=1.0, allow_native_channels=True
    )
    pcm_request = SpeechAudioStream(output_sample_rate=None, speed=1.0)

    assert stream_bytes(unmarked, pieces, keep=False) == expected
    assert stream_bytes(pcm_request, pieces, keep=True) == expected
    assert unmarked.channels == pcm_request.channels == mono.channels == 1


def test_marked_stereo_stream_resamples_both_channels() -> None:
    pieces = stereo_pieces()
    stream = SpeechAudioStream(
        output_sample_rate=16000, speed=1.0, allow_native_channels=True
    )

    body = stream_bytes(stream, pieces, keep=True)

    frames = np.frombuffer(body, dtype=np.int16).reshape(-1, 2)
    assert stream.channels == 2 and stream.sample_rate == 16000
    assert abs(frames.shape[0] - sum(p.shape[-1] for p in pieces) // 2) <= 32
