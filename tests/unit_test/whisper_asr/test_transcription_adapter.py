# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Whisper transcription adapter behavior."""

from __future__ import annotations

import pytest

from sglang_omni.serve.transcription_adapters.whisper import WhisperTranscriptionAdapter


@pytest.mark.parametrize(
    "phrase",
    [
        "the decoder keeps repeating this ending",
        "这是持续重复的结尾",
        "これは繰り返される末尾です",
        "นี่คือข้อความซ้ำที่ส่วนท้าย",
    ],
)
def test_sustained_terminal_repetition_triggers_retry(phrase: str) -> None:
    adapter = WhisperTranscriptionAdapter()
    assert adapter.should_retry_chunk_without_context(phrase * 3)


@pytest.mark.parametrize(
    "text",
    [
        "yes yes yes",
        "please try again please try again",
        "We repeated the same phrase three times before moving on normally.",
        "好好好，我们继续讨论下一部分。",
        "はい、はい、はい。でも話は続きます。",
    ],
)
def test_ordinary_repetition_does_not_trigger_retry(text: str) -> None:
    adapter = WhisperTranscriptionAdapter()
    assert not adapter.should_retry_chunk_without_context(text)
