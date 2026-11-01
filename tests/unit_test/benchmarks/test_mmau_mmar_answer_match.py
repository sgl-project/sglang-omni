# SPDX-License-Identifier: Apache-2.0
"""Answer-key matching for the MMAU and MMAR loaders."""

import pytest

from benchmarks.dataset.mmar import _match_answer as match_mmar_answer
from benchmarks.dataset.mmau import _match_answer as match_mmau_answer


@pytest.mark.parametrize("match_answer", [match_mmau_answer, match_mmar_answer])
@pytest.mark.parametrize(
    ("choices", "answer", "expected"),
    [
        (["G", "A#", "D", "E"], "D", 2),
        (["twenty-one", "Berman's", "said", "a"], "a", 3),
        (["cat", "dog"], "B", 1),
    ],
)
def test_single_letter_answer_matches_choice_text_first(
    match_answer, choices: list[str], answer: str, expected: int
) -> None:
    """A one-letter answer text was read as an option letter."""
    assert match_answer(choices, answer) == expected
