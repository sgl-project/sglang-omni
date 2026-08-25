# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import wave
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from sglang_omni.serve import speech_to_text
from sglang_omni.serve.transcriptions import (
    build_transcription_generate_request as legacy_transcription_request_builder,
)


def _build_request(*, task: str = "transcribe"):
    return speech_to_text.build_speech_to_text_generate_request(
        audio_bytes=b"RIFF",
        filename="sample.wav",
        content_type="audio/wav",
        model="openai/whisper-large-v3",
        language="en",
        prompt=None,
        temperature=None,
        task=task,
    )


def test_transcription_builder_import_keeps_shared_callable() -> None:
    """Keep the pre-extraction import stable while stacked consumers migrate."""
    assert (
        legacy_transcription_request_builder
        is speech_to_text.build_speech_to_text_generate_request
    )


def test_build_request_defaults_to_transcribe_task() -> None:
    assert _build_request().extra_params["task"] == "transcribe"


def test_build_request_accepts_sibling_endpoint_task() -> None:
    assert _build_request(task="translate").extra_params["task"] == "translate"


def test_response_format_validation_preserves_endpoint_error_contract() -> None:
    with pytest.raises(HTTPException) as exc_info:
        speech_to_text.validate_speech_to_text_response_format(
            " SRT ",
            stream=False,
            endpoint_path="/v1/audio/transcriptions",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Unsupported response_format for /v1/audio/transcriptions: ' SRT '"
    )


def test_assemble_uses_the_caller_probed_duration(monkeypatch) -> None:
    # A handler that already probed the upload passes the duration in;
    # assembling the response must not probe the same bytes again.
    monkeypatch.setattr(
        speech_to_text,
        "probe_audio_duration",
        lambda audio_bytes: pytest.fail("re-probed the upload"),
    )

    response = speech_to_text.assemble_speech_to_text_response(
        text="hello world",
        response_format="json",
        endpoint_path="/v1/audio/transcriptions",
        task="transcribe",
        language="en",
        audio_bytes=b"not-a-real-audio-file",
        architectures=None,
        duration_s=2.4,
    )

    assert json.loads(response.body)["usage"] == {"seconds": 3, "type": "duration"}


def test_probe_measures_wav_without_the_av_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        speech_to_text,
        "_av_duration",
        lambda audio_bytes: pytest.fail("fell back to av for a wav upload"),
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 8000)

    assert speech_to_text.probe_audio_duration(buffer.getvalue()) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "header",
    [
        pytest.param(
            b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" * 32, id="id3-mp3"
        ),
        pytest.param(b"\xff\xfb\x90\x00" * 64, id="bare-mp3-frames"),
        pytest.param(b"\x1a\x45\xdf\xa3" + b"\x00" * 64, id="webm"),
        pytest.param(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 64, id="m4a"),
    ],
)
def test_probe_keeps_estimated_containers_on_av(monkeypatch, header) -> None:
    import soundfile

    monkeypatch.setattr(
        soundfile,
        "info",
        lambda *args, **kwargs: pytest.fail(
            "soundfile consulted for a container it can only estimate"
        ),
    )
    monkeypatch.setattr(speech_to_text, "_av_duration", lambda audio_bytes: 86.0)

    assert speech_to_text.probe_audio_duration(header) == 86.0


def test_probe_falls_back_when_soundfile_cannot_measure(monkeypatch) -> None:
    monkeypatch.setattr(speech_to_text, "_av_duration", lambda audio_bytes: 12.5)

    assert speech_to_text.probe_audio_duration(b"fLaC" + b"\x00" * 64) == 12.5


@pytest.mark.parametrize(
    ("header", "info"),
    [
        pytest.param(
            b"RIFF\x16\x2c\x0a\x00WAVEfmt ",
            SimpleNamespace(
                samplerate=44100, frames=1_471_236, subtype="MPEG_LAYER_III"
            ),
            id="mpeg-inside-wav",
        ),
        pytest.param(
            b"fLaC" + b"\x00" * 64,
            SimpleNamespace(samplerate=44100, frames=2**63 - 1, subtype="PCM_16"),
            id="flac-unknown-length",
        ),
    ],
)
def test_probe_distrusts_inexact_soundfile_answers(monkeypatch, header, info) -> None:
    import soundfile

    monkeypatch.setattr(soundfile, "info", lambda *args, **kwargs: info)
    monkeypatch.setattr(speech_to_text, "_av_duration", lambda audio_bytes: 100.0)

    assert speech_to_text.probe_audio_duration(header) == 100.0


def test_probe_returns_zero_for_unreadable_bytes() -> None:
    assert speech_to_text.probe_audio_duration(b"\x00not-audio") == 0.0


def test_verbose_response_uses_requested_task() -> None:
    response = speech_to_text.assemble_speech_to_text_response(
        text="hello world",
        response_format="verbose_json",
        endpoint_path="/v1/audio/transcriptions",
        task="translate",
        language="en",
        audio_bytes=b"not-a-real-audio-file",
        architectures=None,
    )

    assert json.loads(response.body)["task"] == "translate"
