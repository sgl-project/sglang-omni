# SPDX-License-Identifier: Apache-2.0
"""The strict prefix of a known text that a transcript of its speech says."""

import pytest

from sglang_omni.audio.spoken_prefix import spoken_prefix_end


@pytest.mark.parametrize(
    "text, transcript, prefix",
    [
        ("1, 2, 3, 4, 5, 6, 7", "一二三四五六", "1, 2, 3, 4, 5, 6,"),
        (
            "一，二，三，四，五，六，七",
            "one two three four five six",
            "一，二，三，四，五，六，",
        ),
        ("One TWO three", "1，2", "One TWO"),
        ("预算是500，日期不变", "预算是五百", "预算是500，"),
        ("1 2 3 4", "1 3 4", "1"),
        ("1 2 3 4", "1 2 2 3", "1 2"),
        ("go go stop", "go stop", "go"),
        ("1 2 3", "unrelated", ""),
        ("1 2 3 4 5 6 7 8 9 10", "一二三四五六七八九十", "1 2 3 4 5 6 7 8 9 10"),
        ("21", "二十一", "21"),
        ("三百零五个", "305个", "三百零五个"),
        (
            "语速不变，预算500，日期周五。后面不用说。",
            "语速不变。预算五百，日期周五",
            "语速不变，预算500，日期周五。",
        ),
        (
            "first sentence. second sentence. third sentence",
            "first sentence second sentence",
            "first sentence. second sentence.",
        ),
        ("1 2 3", "", ""),
    ],
)
def test_the_prefix_stops_at_the_first_unit_the_transcript_does_not_say(
    text, transcript, prefix
):
    assert text[: spoken_prefix_end(text, transcript)] == prefix
