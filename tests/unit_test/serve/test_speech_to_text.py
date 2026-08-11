# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

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
