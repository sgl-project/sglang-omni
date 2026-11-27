# SPDX-License-Identifier: Apache-2.0
"""English number normalization tests for the Ming-Omni talker."""

from __future__ import annotations

import pytest

from sglang_omni.models.ming_omni.talker.front.number_en import normalize_numbers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("May 1st", "May first"),
        ("the 21st century", "the twenty-first century"),
        ("the 2nd and 3rd floors", "the second and third floors"),
    ],
)
def test_normalize_numbers_speaks_ordinals(text: str, expected: str) -> None:
    assert normalize_numbers(text) == expected
