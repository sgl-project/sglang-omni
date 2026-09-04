# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Whisper transcription adapter behavior."""

from __future__ import annotations

import pytest

from sglang_omni.serve.transcription_adapters.base import resolve_adapter
from sglang_omni.serve.transcription_adapters.whisper_asr import WhisperASRAdapter


def test_resolved_whisper_adapter_combines_timestamp_and_chunk_context() -> None:
    """Keep both Whisper adapters active through their shared registry key."""
    adapter = resolve_adapter(["WhisperForConditionalGeneration"])

    assert isinstance(adapter, WhisperASRAdapter)
    assert adapter.supports_segment_timestamps is True
    assert adapter.requires_ordered_chunk_decoding is True


def test_parse_whisper_timestamp_segments() -> None:
    adapter = WhisperASRAdapter()

    response = adapter.build_timestamped_response(
        text="<|0.00|> Hello there.<|1.20|><|1.20|> Goodbye.<|2.40|>",
        language="en",
        audio_duration_s=2.4,
    )

    assert [
        (segment.id, segment.start, segment.end, segment.text)
        for segment in response.segments
    ] == [
        (0, 0.0, 1.2, "Hello there."),
        (1, 1.2, 2.4, "Goodbye."),
    ]
    assert response.text == "Hello there. Goodbye."


def test_postprocess_strips_whisper_timestamp_markers() -> None:
    adapter = WhisperASRAdapter()

    assert adapter.postprocess_text("<|0.00|> hello<|1.20|>") == "hello"


def test_verbose_json_falls_back_for_markerless_text() -> None:
    adapter = WhisperASRAdapter()

    response = adapter.build_verbose_response(
        text="plain transcript", language="en", audio_duration_s=3.5
    )

    assert [
        (segment.id, segment.start, segment.end, segment.text)
        for segment in response.segments
    ] == [(0, 0.0, 3.5, "plain transcript")]


def test_subtitle_response_rejects_markerless_text() -> None:
    adapter = WhisperASRAdapter()

    with pytest.raises(ValueError, match="model did not produce segment timestamps"):
        adapter.build_timestamped_response(
            text="plain transcript", language="en", audio_duration_s=3.5
        )


@pytest.mark.parametrize(
    "text",
    [
        "<|0.00|>hello<|1.20|> trailing unpaired text",
        "hello <|1.20|>",
        "<|0.00|> hello",
        "<|2.00|>foo<|1.00|>",
        "<|0.00|>hello<|1.20|><|1.00|>backwards<|2.00|>",
        "<|0.00|>hello<|1.20|><|1.20|>",
    ],
)
def test_subtitle_response_rejects_incomplete_timestamp_coverage(text: str) -> None:
    adapter = WhisperASRAdapter()

    with pytest.raises(ValueError, match="model did not produce segment timestamps"):
        adapter.build_timestamped_response(
            text=text, language="en", audio_duration_s=3.5
        )


@pytest.mark.parametrize(
    "phrase",
    [
        "the decoder keeps repeating this ending",
        "这是持续重复的结尾",
        "これは繰り返される末尾です",
        "นี่คือข้อความซ้ำที่ส่วนท้าย",
    ],
)
def test_sustained_repetition_at_end_triggers_retry(phrase: str) -> None:
    adapter = WhisperASRAdapter()
    assert adapter.should_retry_chunk_without_context(phrase * 3)


def test_sustained_repetition_before_valid_continuation_triggers_retry() -> None:
    adapter = WhisperASRAdapter()
    phrase = "The Buttes Trail crosses the western ridge. "
    text = (
        "The speaker introduces the route. "
        + phrase * 3
        + "Afterward, the report continues with a valid conclusion."
    )
    assert adapter.should_retry_chunk_without_context(text)


@pytest.mark.parametrize(
    "text",
    [
        "yes yes yes",
        "important important important",
        "please try again please try again",
        (
            "The Buttes Trail crosses the western ridge. "
            "The Buttes Trail winds along the western ridge. "
            "The Buttes Trail crosses beyond the western ridge."
        ),
        "We repeated the same phrase three times before moving on normally.",
        "好好好，我们继续讨论下一部分。",
        "はい、はい、はい。でも話は続きます。",
    ],
)
def test_ordinary_repetition_does_not_trigger_retry(text: str) -> None:
    adapter = WhisperASRAdapter()
    assert not adapter.should_retry_chunk_without_context(text)
