# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from benchmarks.metrics.transcribe_diarize_metrics import (
    clean_no_speaker,
    split_clean_by_speaker,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[S01]我笑了", "我笑了"),
        ("[S01]她喜欢音乐", "她喜欢音乐"),
        ("[S01]I love music", "ilovemusic"),
        ("[S01][笑声]你好", "你好"),
        ("[S01]<silence>Hello [music]", "hello"),
    ],
)
def test_clean_no_speaker_only_strips_marked_events(text: str, expected: str) -> None:
    assert clean_no_speaker(text) == expected


def test_split_clean_by_speaker_preserves_spoken_event_words() -> None:
    assert split_clean_by_speaker(
        "[S01]我笑了[S02]I love music", implicit_single_speaker=False
    ) == {
        "[S1]": "我笑了",
        "[S2]": "ilovemusic",
    }
