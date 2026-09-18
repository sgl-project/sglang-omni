# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from sglang_omni.serve import speech_to_text
from sglang_omni.serve.transcription_adapters import resolve_adapter

ARCHITECTURES = ["Nemotron3_5AsrForRNNT"]


@pytest.mark.parametrize("response_format", ["text", "json", "verbose_json"])
def test_response_assembler_cleans_text_and_populates_language(response_format) -> None:
    response = speech_to_text.assemble_speech_to_text_response(
        text="hello <en-US> world <en-US>",
        response_format=response_format,
        endpoint_path="/v1/audio/transcriptions",
        task="transcribe",
        language="auto",
        audio_bytes=b"invalid",
        architectures=ARCHITECTURES,
        duration_s=1.0,
    )
    if response_format == "text":
        assert response.body.decode() == "hello world"
    else:
        data = json.loads(response.body)
        assert data["text"] == "hello world"
        if response_format == "verbose_json":
            assert data["language"] == "en-US"
