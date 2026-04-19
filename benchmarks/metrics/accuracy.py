# SPDX-License-Identifier: Apache-2.0
"""Accuracy metrics for multiple-choice evaluation (e.g. MMSU)."""

from __future__ import annotations

import re

ANSWER_LETTERS = {"A": 0, "B": 1, "C": 2, "D": 3}
INDEX_TO_LETTER = {v: k for k, v in ANSWER_LETTERS.items()}

# Patterns tried in order: first match wins
_PATTERNS = [
    # Bare letter at start: "B", "B.", "B) ..." (not start of a word like "Because")
    re.compile(r"^\s*([A-D])(?!\w)(?!\s+[a-z])", re.IGNORECASE),
    # "The answer is B" / "answer: B"
    re.compile(r"(?:answer|choice)\s*(?:is|:)\s*([A-D])\b", re.IGNORECASE),
    # "Option B" / "option B"
    re.compile(r"option\s+([A-D])\b", re.IGNORECASE),
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
