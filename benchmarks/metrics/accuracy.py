# SPDX-License-Identifier: Apache-2.0
"""Accuracy metrics for multiple-choice evaluation (e.g. MMSU)."""

from __future__ import annotations

import re

ANSWER_LETTERS = {chr(ord("A") + i): i for i in range(10)}
INDEX_TO_LETTER = {v: k for k, v in ANSWER_LETTERS.items()}
_LETTER_RANGE = "A-J"

# Patterns tried in order: first match wins
_PATTERNS = [
    # Bare letter at start: "B", "B.", "B) ..." (not start of a word like "Because")
    re.compile(rf"^\s*([{_LETTER_RANGE}])(?!\w)(?!\s+[a-z])", re.IGNORECASE),
    # "The answer is B" / "answer: B"
    re.compile(
        rf"(?:answer|choice)\s*(?:is|:)\s*([{_LETTER_RANGE}])\b",
        re.IGNORECASE,
    ),
    # "Option B" / "option B"
    re.compile(rf"option\s+([{_LETTER_RANGE}])\b", re.IGNORECASE),
]


def extract_answer_letter(text: str) -> int | None:
    """Extract the predicted answer index (0-3) from model response text.

    Tries multiple patterns in priority order. Returns None if no answer
    letter can be parsed.
    """
    text = text.strip()
    if not text:
        return None

    for pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            letter = match.group(1).upper()
            return ANSWER_LETTERS[letter]

    return None
