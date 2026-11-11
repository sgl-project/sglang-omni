# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.serve.subtitle_cues import (
    build_subtitle_cues,
    segments_from_word_timestamps,
)


def word(text: str, start: float, end: float) -> dict[str, object]:
    return {"word": text, "start": start, "end": end}


def test_short_words_share_one_cue() -> None:
    cues = build_subtitle_cues([word("Hello", 0.0, 0.4), word("there.", 0.4, 0.8)])
    assert cues == [{"start": 0.0, "end": 0.8, "text": "Hello there."}]


def test_a_sentence_break_is_preferred_once_the_cue_is_long_enough() -> None:
    cues = build_subtitle_cues(
        [
            word("Hello.", 0.0, 0.9),
            word("there", 0.9, 1.2),
        ]
    )
    assert [cue["text"] for cue in cues] == ["Hello.", "there"]


def test_duration_limit_starts_a_new_cue() -> None:
    cues = build_subtitle_cues([word("one", 0.0, 4.0), word("two", 4.0, 8.1)])
    assert [cue["text"] for cue in cues] == ["one", "two"]


def test_one_oversized_word_is_its_own_cue() -> None:
    long_word = "x" * 50
    cues = build_subtitle_cues([word(long_word, 0.0, 0.2), word("next", 0.2, 0.4)])
    assert cues[0]["text"] == long_word
    assert cues[1]["text"] == "next"


def test_unavailable_alignment_refuses_subtitles() -> None:
    with pytest.raises(ValueError, match="ctc_evidence_unavailable"):
        segments_from_word_timestamps(
            {
                "words": [],
                "word_timestamp_alignment": {
                    "status": "unavailable",
                    "failure_reason": "ctc_evidence_unavailable",
                },
            }
        )
