# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmarks.metrics.accuracy.

These tests pin behaviour of the answer-parsing helpers that previously
lived in ``benchmarks.tasks.visual_understand``. They serve as a guard
against accidental drift while metric logic is being consolidated under
``benchmarks/metrics/`` (RFC: sgl-project/sglang-omni#360).
"""

from __future__ import annotations

import random

import pytest

from benchmarks.metrics.accuracy import (
    INDEX_TO_LETTER,
    eval_open,
    extract_answer_letter,
    parse_multi_choice_response,
    parse_open_response,
)


class TestExtractAnswerLetter:
    def test_bare_letter(self) -> None:
        assert extract_answer_letter("B") == 1
        assert extract_answer_letter("b.") == 1
        assert extract_answer_letter("C) something") == 2

    def test_phrase_forms(self) -> None:
        assert extract_answer_letter("The answer is A") == 0
        assert extract_answer_letter("answer: D") == 3
        assert extract_answer_letter("Option C") == 2

    def test_does_not_match_word_boundary(self) -> None:
        # "Because..." should not match "B"
        assert extract_answer_letter("Because the option is wrong") is None

    def test_empty_or_whitespace(self) -> None:
        assert extract_answer_letter("") is None
        assert extract_answer_letter("   ") is None

    def test_index_letter_roundtrip(self) -> None:
        for letter, idx in (("A", 0), ("B", 1), ("C", 2), ("D", 3)):
            assert INDEX_TO_LETTER[idx] == letter


class TestParseMultiChoiceResponse:
    def _index2ans(self, choices: list[str]) -> dict[str, str]:
        return {c: f"option-text-{c}" for c in choices}

    def test_explicit_answer_line(self) -> None:
        choices = ["A", "B", "C", "D"]
        choice, fallback = parse_multi_choice_response(
            "Reasoning here. Answer: B",
            choices,
            self._index2ans(choices),
        )
        assert choice == "B"
        assert fallback is False

    def test_bracketed_letter(self) -> None:
        choices = ["A", "B", "C", "D"]
        choice, fallback = parse_multi_choice_response(
            "I think it is (C) for the reason that...",
            choices,
            self._index2ans(choices),
        )
        assert choice == "C"
        assert fallback is False

    def test_space_padded_letter(self) -> None:
        choices = ["A", "B", "C", "D"]
        choice, fallback = parse_multi_choice_response(
            "the choice should be A here",
            choices,
            self._index2ans(choices),
        )
        assert choice == "A"
        assert fallback is False

    def test_option_text_match_with_long_response(self) -> None:
        choices = ["A", "B", "C", "D"]
        index2ans = {
            "A": "alpha",
            "B": "bravo signal",
            "C": "charlie",
            "D": "delta",
        }
        choice, fallback = parse_multi_choice_response(
            "After careful reasoning the bravo signal is correct here",
            choices,
            index2ans,
        )
        assert choice == "B"
        assert fallback is False

    def test_random_fallback_signalled(self) -> None:
        # Stable seed → deterministic fallback choice
        random.seed(0)
        choices = ["A", "B", "C", "D"]
        choice, fallback = parse_multi_choice_response(
            "asdf",
            choices,
            {c: "" for c in choices},
        )
        assert choice in choices
        assert fallback is True

    def test_last_occurrence_tie_break(self) -> None:
        # Two bracketed letters; later position wins
        choices = ["A", "B", "C", "D"]
        choice, fallback = parse_multi_choice_response(
            "First (A), but actually (C) is correct.",
            choices,
            self._index2ans(choices),
        )
        assert choice == "C"
        assert fallback is False


class TestParseOpenResponse:
    def test_explicit_answer_tag(self) -> None:
        out = parse_open_response("Reasoning. Answer: 42")
        assert 42.0 in out

    def test_answer_tag_with_boxed(self) -> None:
        out = parse_open_response(r"Steps... Answer: \boxed{13.0}")
        assert 13.0 in out

    def test_answer_tag_text(self) -> None:
        out = parse_open_response("Answer: MgS")
        assert "mgs" in out

    def test_extracts_number_with_commas(self) -> None:
        out = parse_open_response("Answer: 1,234")
        assert 1234.0 in out

    def test_falls_back_to_heuristic_when_no_tag(self) -> None:
        # No "Answer:" line; heuristic should still surface candidates
        out = parse_open_response("So the result is 7.")
        assert out  # non-empty
        assert any(p == 7.0 or (isinstance(p, str) and "7" in p) for p in out)


class TestEvalOpen:
    def test_string_match(self) -> None:
        assert eval_open("hello", ["this contains hello somewhere"]) is True

    def test_number_exact_match(self) -> None:
        assert eval_open("42", [42.0]) is True

    def test_no_match(self) -> None:
        assert eval_open("dog", ["cat", "fish"]) is False

    def test_list_of_acceptable_answers(self) -> None:
        assert eval_open(["foo", "bar"], ["the bar is set high"]) is True


@pytest.mark.parametrize(
    "response,expected_idx",
    [
        ("A", 0),
        ("B.", 1),
        ("C)", 2),
        ("D ", 3),
        ("The answer is A", 0),
        ("answer: D", 3),
        ("Option C", 2),
    ],
)
def test_extract_answer_letter_table(response: str, expected_idx: int) -> None:
    assert extract_answer_letter(response) == expected_idx
