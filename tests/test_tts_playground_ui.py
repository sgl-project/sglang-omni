# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the S2-Pro TTS playground UI handlers."""

from __future__ import annotations

import wave

import numpy as np

from playground.tts.api_client import SpeechDemoClientError
from playground.tts.audio_stream import SpeechStreamEvent
from playground.tts.ui import make_streaming_handler
from sglang_omni.client.audio import encode_wav


def test_streaming_handler_builds_expected_final_wav(monkeypatch) -> None:
    first = encode_wav(np.array([0.0, 0.1], dtype=np.float32), 24000)
    second = encode_wav(np.array([-0.1, 0.0], dtype=np.float32), 24000)

    def _stream(self, request):
        del self, request
        yield SpeechStreamEvent(index=0, audio_bytes=first, sample_rate=24000)
        yield SpeechStreamEvent(index=1, audio_bytes=second, sample_rate=24000)
        yield SpeechStreamEvent(index=2, finish_reason="stop")

    monkeypatch.setattr("playground.tts.ui.SpeechDemoClient.stream_synthesize", _stream)

    handler = make_streaming_handler("http://localhost:8000")
    outputs = list(
        handler(
            "hello world",
            None,
            "",
            0.8,
            0.8,
            30,
            256,
            [],
            [],
        )
    )

    assert len(outputs) == 4
    _, _, live_audio, _, live_status, _ = outputs[1]
    assert isinstance(live_audio, bytes)
    assert "Streaming | chunk 1" in live_status

    final_history, _, _, final_path, final_status, artifact_paths = outputs[-1]
    assert final_path is not None
    assert final_path in artifact_paths
    assert "chunks" in final_status
    assert final_history[-1]["content"][0]["path"] == final_path

    with wave.open(final_path, "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 4
        frames = wav_file.readframes(wav_file.getnframes())

    expected = (np.array([0.0, 0.1, -0.1, 0.0], dtype=np.float32) * 32767.0).astype(
        np.int16
    )
    assert frames == expected.tobytes()


def test_streaming_handler_reports_truncated_stream(monkeypatch) -> None:
    def _stream(self, request):
        del self, request
        raise SpeechDemoClientError("stream closed early")
        yield

    monkeypatch.setattr("playground.tts.ui.SpeechDemoClient.stream_synthesize", _stream)

    handler = make_streaming_handler("http://localhost:8000")
    outputs = list(
        handler(
            "hello world",
            None,
            "",
            0.8,
            0.8,
            30,
            256,
            [],
            [],
        )
    )

    assert len(outputs) == 2
    failed_history, _, _, _, status, artifact_paths = outputs[-1]
    assert artifact_paths == []
    assert "Request failed" in status
    assert "stream closed early" in failed_history[-1]["content"]
