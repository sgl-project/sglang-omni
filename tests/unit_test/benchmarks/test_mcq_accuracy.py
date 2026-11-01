# SPDX-License-Identifier: Apache-2.0
"""Answer-letter parsing for multiple-choice benchmarks."""

import pytest

from benchmarks.metrics.accuracy import extract_answer_letter


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("I'm fairly sure the answer is B.", 1),
        ("I'd go with option C.", 2),
        ("I.", 8),
    ],
)
def test_leading_contraction_is_not_answer_letter(response: str, expected: int) -> None:
    """A reply that started with "I'm" or "I'd" was scored as choice I."""
    assert extract_answer_letter(response) == expected
