# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from sglang_omni.serve import speech_to_text
from sglang_omni.serve.transcription_adapters import resolve_adapter

_ARCHITECTURES = ["Nemotron3_5AsrForRNNT"]


def test_adapter_removes_repeated_tags_and_resolves_detected_locale() -> None:
    adapter = resolve_adapter(_ARCHITECTURES)
    raw = (
        "What is happening in this video? <en-US> "
        "Answer in ten words or fewer. <en-US>"
    )

    assert adapter.resolve_language(raw, "auto") == "en-US"
    assert adapter.postprocess_text(raw) == (
        "What is happening in this video? Answer in ten words or fewer."
    )


def test_adapter_does_not_invent_one_language_for_mixed_output() -> None:
    adapter = resolve_adapter(_ARCHITECTURES)

    assert adapter.resolve_language("hello <en-US> ni hao <zh-CN>", "auto") is None


def test_response_assembler_cleans_text_and_populates_verbose_language() -> None:
    raw = "hello <en-US> world <en-US>"

    text_response = speech_to_text.assemble_speech_to_text_response(
        text=raw,
        response_format="text",
        endpoint_path="/v1/audio/transcriptions",
        task="transcribe",
        language="auto",
        audio_bytes=b"invalid",
        architectures=_ARCHITECTURES,
        duration_s=1.0,
    )
    json_response = speech_to_text.assemble_speech_to_text_response(
        text=raw,
        response_format="json",
        endpoint_path="/v1/audio/transcriptions",
        task="transcribe",
        language="auto",
        audio_bytes=b"invalid",
        architectures=_ARCHITECTURES,
        duration_s=1.0,
    )
    verbose_response = speech_to_text.assemble_speech_to_text_response(
        text=raw,
        response_format="verbose_json",
        endpoint_path="/v1/audio/transcriptions",
        task="transcribe",
        language="auto",
        audio_bytes=b"invalid",
        architectures=_ARCHITECTURES,
        duration_s=1.0,
    )

    assert text_response.body.decode() == "hello world"
    assert json.loads(json_response.body)["text"] == "hello world"
    verbose = json.loads(verbose_response.body)
    assert verbose["text"] == "hello world"
    assert verbose["language"] == "en-US"


def test_plain_text_hook_does_not_change_existing_adapter_output() -> None:
    adapter = resolve_adapter(["MossTranscribeDiarizeForConditionalGeneration"])
    raw = "<|im_start|>hello"

    assert adapter.postprocess_plain_text(raw) == raw
    assert adapter.postprocess_text(raw) == "hello"
