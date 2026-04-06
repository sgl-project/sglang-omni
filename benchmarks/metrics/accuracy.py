# SPDX-License-Identifier: Apache-2.0
"""Accuracy metrics for multiple-choice evaluation (e.g. Speech MMLU)."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

ANSWER_LETTERS = {"A": 0, "B": 1, "C": 2, "D": 3}
INDEX_TO_LETTER = {v: k for k, v in ANSWER_LETTERS.items()}

# Patterns tried in order: first match wins
_PATTERNS = [
    # Bare letter at start: "B", "B.", "B) ..."
    re.compile(r"^\s*([A-D])\b", re.IGNORECASE),
    # "The answer is B" / "answer: B"
    re.compile(r"(?:answer|choice)\s*(?:is|:)\s*([A-D])\b", re.IGNORECASE),
    # "Option B" / "option B"
    re.compile(r"option\s+([A-D])\b", re.IGNORECASE),
    # Standalone letter anywhere
    re.compile(r"\b([A-D])\b"),
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


def compute_accuracy_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute overall and per-subject accuracy from evaluation results.

    Args:
        results: List of dicts with keys: subject, correct_answer (int),
                 predicted_answer (int | None), is_correct (bool),
                 is_parseable (bool).

    Returns:
        Dict with overall and per-subject accuracy statistics.
    """
    total = len(results)
    parseable = sum(1 for r in results if r["is_parseable"])
    unparseable = total - parseable
    correct = sum(1 for r in results if r["is_correct"])

    # Per-subject
    by_subject: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "parseable": 0}
    )
    for r in results:
        subj = r["subject"]
        by_subject[subj]["total"] += 1
        if r["is_parseable"]:
            by_subject[subj]["parseable"] += 1
        if r["is_correct"]:
            by_subject[subj]["correct"] += 1

    per_subject = {}
    for subj, counts in sorted(by_subject.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        per_subject[subj] = {
            "total": counts["total"],
            "correct": counts["correct"],
            "parseable": counts["parseable"],
            "accuracy": round(acc, 4),
        }

    overall_accuracy = correct / total if total > 0 else 0.0

    return {
        "total_samples": total,
        "parseable_samples": parseable,
        "unparseable_samples": unparseable,
        "correct": correct,
        "incorrect": parseable - correct,
        "overall_accuracy": round(overall_accuracy, 4),
        "per_subject": per_subject,
    }
