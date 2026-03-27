# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the S2-Pro TTS playground request models."""

from __future__ import annotations

import pytest

from playground.tts.models import GenerationSettings, SpeechSynthesisRequest


def test_speech_synthesis_request_builds_base_payload() -> None:
    request = SpeechSynthesisRequest(
        text="hello world",
        reference_audio_path=None,
        reference_text="",
        settings=GenerationSettings(
            temperature=0.8,
            top_p=0.9,
            top_k=30,
            max_new_tokens=512,
        ),
    )

    payload = request.to_payload()

    assert payload == {
        "input": "hello world",
        "voice": "default",
        "response_format": "wav",
        "max_new_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 30,
    }


def test_speech_synthesis_request_builds_reference_payload() -> None:
    request = SpeechSynthesisRequest(
        text="hello world",
        reference_audio_path="/tmp/ref.wav",
        reference_text="reference transcript",
        settings=GenerationSettings(
            temperature=0.8,
            top_p=0.9,
            top_k=30,
            max_new_tokens=512,
        ),
    )

    payload = request.to_payload()

    assert payload["ref_audio"] == "/tmp/ref.wav"
    assert payload["ref_text"] == "reference transcript"


def test_speech_synthesis_request_skips_blank_reference_text() -> None:
    request = SpeechSynthesisRequest(
        text="hello world",
        reference_audio_path="/tmp/ref.wav",
        reference_text="   ",
        settings=GenerationSettings(
            temperature=0.8,
            top_p=0.9,
            top_k=30,
            max_new_tokens=512,
        ),
    )

    payload = request.to_payload()

    assert payload["ref_audio"] == "/tmp/ref.wav"
    assert "ref_text" not in payload


def test_speech_synthesis_request_builds_history_content_with_reference() -> None:
    request = SpeechSynthesisRequest(
        text="hello world",
        reference_audio_path="/tmp/ref.wav",
        reference_text="reference transcript",
        settings=GenerationSettings(
            temperature=0.8,
            top_p=0.9,
            top_k=30,
            max_new_tokens=512,
        ),
    )

    assert request.build_history_user_content() == [
        "hello world",
        {"path": "/tmp/ref.wav", "mime_type": "audio/wav"},
    ]


def test_speech_synthesis_request_rejects_blank_text() -> None:
    request = SpeechSynthesisRequest(
        text="   ",
        reference_audio_path=None,
        reference_text="",
        settings=GenerationSettings(
            temperature=0.8,
            top_p=0.9,
            top_k=30,
            max_new_tokens=512,
        ),
    )

    with pytest.raises(ValueError, match="Please enter some text"):
        request.validate()
