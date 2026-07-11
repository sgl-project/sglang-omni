from __future__ import annotations

from sglang_omni.models.ming_omni.components.streaming_text import (
    SegmenterConfig,
    SegmenterState,
    text_to_uint8_tensor,
    uint8_tensor_to_text,
)


def _token_count(text: str) -> int:
    return len(text.split())


def test_utf8_tensor_round_trip_chinese_and_english():
    text = "world，streaming TTS!"

    assert uint8_tensor_to_text(text_to_uint8_tensor(text)) == text


def test_segmenter_flushes_on_punctuation_after_min_tokens():
    state = SegmenterState(
        SegmenterConfig(
            segment_min_tokens=2,
            segment_max_tokens=10,
            first_segment_max_wait_ms=500,
        ),
        token_count_fn=_token_count,
    )

    assert state.push("hello", now_ms=0) == []
    out = state.push(" world.", now_ms=10)

    assert [segment.text for segment in out] == ["hello world."]
    assert out[0].is_final_segment is False


def test_first_segment_max_wait_flushes_before_punctuation():
    state = SegmenterState(
        SegmenterConfig(
            segment_min_tokens=10,
            segment_max_tokens=40,
            first_segment_min_tokens=3,
            first_segment_max_wait_ms=400,
        ),
        token_count_fn=_token_count,
    )

    assert state.push("one two three", now_ms=0) == []
    out = state.push("", now_ms=401)

    assert [segment.text for segment in out] == ["one two three"]
    assert out[0].is_final_segment is False


def test_segmenter_caps_max_tokens_and_retains_overflow():
    state = SegmenterState(
        SegmenterConfig(
            segment_min_tokens=1,
            segment_max_tokens=3,
            first_segment_min_tokens=3,
            first_segment_max_wait_ms=9999,
        ),
        token_count_fn=_token_count,
    )

    out = state.push("one two three four five", now_ms=0)
    tail = state.flush()

    assert [segment.text for segment in out] == ["one two three"]
    assert [segment.text for segment in tail] == ["four five"]
