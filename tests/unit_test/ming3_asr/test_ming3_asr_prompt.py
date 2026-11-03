# SPDX-License-Identifier: Apache-2.0
"""Transcription prompt layout and transcript extraction for Ming 3 ASR."""

import pytest

from sglang_omni.models.ming3_asr.preprocessing import (
    ASR_INSTRUCTION,
    AUDIO_END_TOKEN,
    AUDIO_PATCH_TOKEN,
    AUDIO_START_TOKEN,
    Transcript,
    build_transcription_prompt,
    extract_transcript,
)
from sglang_omni.serve.openai_errors import generation_error_status_code

AUDIO_TOKEN_COUNT = 3
AUDIO_SPAN = AUDIO_START_TOKEN + AUDIO_PATCH_TOKEN * AUDIO_TOKEN_COUNT + AUDIO_END_TOKEN


def test_prompt_is_instruction_followed_by_audio_span() -> None:
    assert (
        build_transcription_prompt(AUDIO_TOKEN_COUNT, None, None)
        == ASR_INSTRUCTION + AUDIO_SPAN
    )


@pytest.mark.parametrize(
    ("language", "answer_prefix"), [("en", "English\t"), ("zh", "Chinese\t")]
)
def test_known_language_prefills_language_label(
    language: str, answer_prefix: str
) -> None:
    assert build_transcription_prompt(AUDIO_TOKEN_COUNT, language, None) == (
        ASR_INSTRUCTION + AUDIO_SPAN + answer_prefix
    )


def test_context_follows_instruction_before_audio() -> None:
    context = "This is an audio about films."
    assert (
        build_transcription_prompt(AUDIO_TOKEN_COUNT, None, context)
        == ASR_INSTRUCTION + context + AUDIO_SPAN
    )


@pytest.mark.parametrize(
    ("language", "context"), [("fr", None), (None, "first line\nsecond line")]
)
def test_unsupported_requests_are_rejected_as_caller_errors(
    language: str | None, context: str | None
) -> None:
    with pytest.raises(ValueError) as rejection:
        build_transcription_prompt(AUDIO_TOKEN_COUNT, language, context)
    assert generation_error_status_code(rejection.value) == 400


@pytest.mark.parametrize(
    ("generated_text", "transcript", "language"),
    [
        ("English\thello there ", "hello there", "English"),
        ("English hello there", "hello there", "English"),
        ("hello there", "hello there", None),
    ],
)
def test_detected_language_label_is_split_from_transcript(
    generated_text: str, transcript: str, language: str | None
) -> None:
    """Auto-detected transcriptions reported no language, though the model writes one."""
    assert extract_transcript(generated_text, None) == Transcript(
        text=transcript, language=language
    )


def test_requested_language_keeps_leading_language_name() -> None:
    """With a requested language, a transcript starting with a language name lost that word."""
    generated_text = "Chinese New Year is coming."
    assert extract_transcript(generated_text, "en") == Transcript(
        text=generated_text, language=None
    )
