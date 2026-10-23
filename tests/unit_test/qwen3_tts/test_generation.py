# SPDX-License-Identifier: Apache-2.0
"""Text budgets must bound short speech without shrinking longer requests."""

import pytest

from sglang_omni.models.qwen3_tts.generation import (
    build_generation_kwargs,
    text_generation_budget,
)


@pytest.mark.parametrize("text", ["Hi!", "好的。", "🎉", "", "...", "12345"])
def test_short_text_cannot_use_minutes_of_default_generation(text: str) -> None:
    """Short inputs previously inherited the full 2048-frame allowance."""
    settings = build_generation_kwargs({}, text=text, tts_engine_params={})
    assert 120 <= settings["max_new_tokens"] <= 180


@pytest.mark.parametrize(
    "text",
    [
        "Please take a seat and tell me what happened. ",
        "请坐下来，慢慢告诉我今天发生了什么事情。",
        "これは日本語の長い文章です。",
        "이 문장을 천천히 읽어 주세요.",
        "Bitte lesen Sie diesen Satz langsam vor. ",
    ],
)
def test_longer_speech_gets_more_room_until_the_service_ceiling(text: str) -> None:
    budgets = [text_generation_budget(text * count) for count in (1, 2, 10, 100)]
    assert budgets[0] < budgets[1]
    assert budgets == sorted(budgets)
    assert all(budget <= 2048 for budget in budgets)


def test_digits_and_unsegmented_text_have_room_to_be_spoken() -> None:
    latin_budget = text_generation_budget("abcdefghijklmno")
    assert text_generation_budget("123456789012345") > latin_budget
    assert text_generation_budget("今天我们一起去公园看看美丽的风景") > latin_budget


def test_formatting_does_not_buy_extra_generation_time() -> None:
    plain_text = "Veuillez répéter cette phrase."
    assert text_generation_budget(plain_text) == text_generation_budget(
        "  Veuillez\n re\u0301pe\u0301ter\t cette phrase.\u200d  "
    )


@pytest.mark.parametrize(
    "params, stage_params, expected",
    [
        ({"max_new_tokens": 1}, {}, 1),
        ({"max_new_tokens": 4096}, {}, 4096),
        ({"max_new_tokens": 300}, {"max_new_tokens": 400}, 400),
        ({"max_new_tokens": 300}, {"max_new_tokens": None}, 300),
        ({"max_new_tokens": None}, {"max_new_tokens": None}, 125),
    ],
)
def test_explicit_budgets_keep_precedence_and_can_extend_slow_speech(
    params: dict[str, int | None],
    stage_params: dict[str, int | None],
    expected: int,
) -> None:
    settings = build_generation_kwargs(
        params, text="Hi.", tts_engine_params=stage_params
    )
    assert settings["max_new_tokens"] == expected


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "12", float("inf")])
def test_invalid_explicit_budget_fails_before_generation(
    budget: int | float | str,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_generation_kwargs(
            {"max_new_tokens": budget}, text="Hi.", tts_engine_params={}
        )


def test_budget_does_not_change_explicit_sampling() -> None:
    settings = build_generation_kwargs(
        {"temperature": 0.0, "subtalker_dosample": False, "seed": 7},
        text="Hi.",
        tts_engine_params={"top_k": 1},
    )
    assert settings == {
        "max_new_tokens": 125,
        "temperature": 0.0,
        "subtalker_dosample": False,
        "top_k": 1,
    }
