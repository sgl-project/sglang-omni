# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the S2-Pro TTS playground UI handlers."""

from __future__ import annotations

import wave

import gradio as gr
import numpy as np

from playground.tts.api_client import SpeechDemoClientError
from playground.tts.audio_stream import SpeechStreamEvent
from playground.tts.models import NonStreamingSpeechResult
from playground.tts.ui import make_non_streaming_handler, make_streaming_handler
from sglang_omni.client.audio import encode_wav


def _assert_reset_update(value) -> None:
    assert isinstance(value, dict)
    assert value.get("value") is None
    assert value.get("__type__") == "update"


def _assert_not_reset_update(value) -> None:
    assert value != gr.update(value=None)


def test_streaming_handler_builds_expected_final_wav(monkeypatch) -> None:
    first = encode_wav(np.array([0.0, 0.1], dtype=np.float32), 24000)
    second = encode_wav(np.array([-0.1, 0.0], dtype=np.float32), 24000)
    third = encode_wav(np.array([0.05, -0.05], dtype=np.float32), 24000)

    def _stream(self, request):
        del self, request
        yield SpeechStreamEvent(index=0, audio_bytes=first, sample_rate=24000)
        yield SpeechStreamEvent(index=1, audio_bytes=second, sample_rate=24000)
        yield SpeechStreamEvent(index=2, audio_bytes=third, sample_rate=24000)
        yield SpeechStreamEvent(index=3, finish_reason="stop")

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

    assert len(outputs) == 5
    (
        _,
        _,
        reset_live_audio,
        reset_final_audio,
        reset_status,
        _,
        reset_pending,
        reset_synth_button,
        reset_stream_button,
    ) = outputs[0]
    _assert_reset_update(reset_live_audio)
    _assert_reset_update(reset_final_audio)
    assert reset_status == "Connecting to speech stream..."
    assert reset_pending is None
    assert reset_synth_button["interactive"] is False
    assert reset_stream_button["interactive"] is False

    (
        _,
        _,
        buffering_live_audio,
        buffering_final_audio,
        buffering_status,
        _,
        buffering_pending,
        _,
        _,
    ) = outputs[1]
    assert buffering_live_audio == gr.skip()
    assert buffering_final_audio == gr.skip()
    assert buffering_status == "Buffering live playback | chunk 1"
    assert buffering_pending is None

    (
        _,
        _,
        live_audio,
        chunk_final_audio,
        live_status,
        _,
        live_pending,
        _,
        _,
    ) = outputs[3]
    assert isinstance(live_audio, bytes)
    assert chunk_final_audio == gr.skip()
    assert "Streaming | chunk 3" in live_status
    assert live_pending is None

    (
        final_history,
        final_text,
        final_live_audio,
        final_path,
        final_status,
        artifact_paths,
        pending_result,
        _,
        _,
    ) = outputs[-1]
    assert final_history == gr.skip()
    assert final_text == gr.skip()
    assert final_live_audio == gr.skip()
    assert final_path == gr.skip()
    assert final_status == gr.skip()
    assert pending_result is not None

    final_path = pending_result["final_audio_path"]
    assert final_path in artifact_paths
    assert "chunks" in pending_result["status"]
    assert "audio" in pending_result["status"]
    assert pending_result["history"][-1]["content"][0]["path"] == final_path
    assert "audio" in pending_result["history"][-1]["content"][1]

    with wave.open(final_path, "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 6
        frames = wav_file.readframes(wav_file.getnframes())

    expected = (
        np.array([0.0, 0.1, -0.1, 0.0, 0.05, -0.05], dtype=np.float32) * 32767.0
    ).astype(np.int16)
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
    (
        failed_history,
        _,
        _,
        _,
        status,
        artifact_paths,
        pending_result,
        failed_synth_button,
        failed_stream_button,
    ) = outputs[-1]
    assert artifact_paths == []
    assert "Request failed" in status
    assert "stream closed early" in failed_history[-1]["content"]
    assert pending_result is None
    assert failed_synth_button["interactive"] is True
    assert failed_stream_button["interactive"] is True


def test_streaming_handler_resets_audio_outputs_for_followup_request(
    monkeypatch,
) -> None:
    first = encode_wav(np.zeros(2400, dtype=np.float32), 24000)

    def _stream(self, request):
        del self, request
        yield SpeechStreamEvent(index=0, audio_bytes=first, sample_rate=24000)
        yield SpeechStreamEvent(index=1, finish_reason="stop")

    monkeypatch.setattr("playground.tts.ui.SpeechDemoClient.stream_synthesize", _stream)

    handler = make_streaming_handler("http://localhost:8000")
    first_outputs = list(
        handler(
            "first request",
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
    pending_result = first_outputs[-1][6]
    artifact_paths = first_outputs[-1][5]
    assert pending_result is not None
    history = pending_result["history"]

    second_outputs = list(
        handler(
            "second request",
            None,
            "",
            0.8,
            0.8,
            30,
            256,
            history,
            artifact_paths,
        )
    )

    (
        _,
        _,
        reset_live_audio,
        reset_final_audio,
        reset_status,
        _,
        reset_pending,
        reset_synth_button,
        reset_stream_button,
    ) = second_outputs[0]
    _assert_reset_update(reset_live_audio)
    _assert_reset_update(reset_final_audio)
    assert reset_status == "Connecting to speech stream..."
    assert reset_pending is None
    assert reset_synth_button["interactive"] is False
    assert reset_stream_button["interactive"] is False


def test_streaming_handler_keeps_live_audio_at_final_handoff(monkeypatch) -> None:
    chunk = encode_wav(np.zeros(2400, dtype=np.float32), 24000)

    def _stream(self, request):
        del self, request
        yield SpeechStreamEvent(index=0, audio_bytes=chunk, sample_rate=24000)
        yield SpeechStreamEvent(index=1, finish_reason="stop")

    monkeypatch.setattr("playground.tts.ui.SpeechDemoClient.stream_synthesize", _stream)

    handler = make_streaming_handler("http://localhost:8000")
    outputs = list(
        handler(
            "handoff test",
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

    flushed_live_audio = outputs[-2][2]
    assert isinstance(flushed_live_audio, bytes)
    final_live_audio = outputs[-1][2]
    _assert_not_reset_update(final_live_audio)
    assert final_live_audio == gr.skip()


def test_non_streaming_handler_summary_includes_audio_duration(monkeypatch) -> None:
    audio_bytes = encode_wav(np.zeros(4800, dtype=np.float32), 24000)

    def _synthesize(self, request):
        del self, request
        return NonStreamingSpeechResult(audio_bytes=audio_bytes, elapsed_s=1.25)

    monkeypatch.setattr(
        "playground.tts.ui.SpeechDemoClient.synthesize",
        _synthesize,
    )

    handler = make_non_streaming_handler("http://localhost:8000")
    history, _, audio_path, status, artifact_paths, synth_button, stream_button = (
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

    assert audio_path is not None
    assert artifact_paths == [audio_path]
    assert status == "0.2s audio | 1.2s total | 9 KB"
    assert history[-1]["content"][1] == status
    assert synth_button["interactive"] is True
    assert stream_button["interactive"] is True
