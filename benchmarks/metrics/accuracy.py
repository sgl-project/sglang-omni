# SPDX-License-Identifier: Apache-2.0
"""Accuracy parsing and answer extraction for benchmark evaluation.

Houses the pure functions that turn free-form model responses into a
predicted answer (or correctness signal). No IO, no async, no task glue —
inputs are strings and option lists; outputs are letters / indices /
booleans / lists of normalised candidates.

Public API:

* ``extract_answer_letter`` — bare-letter A-D extraction (MMSU).
* ``parse_multi_choice_response`` — variable-choice MCQ extraction
  with bracket/space/option-text fallbacks (MMMU, Video-MME).
* ``parse_open_response`` + ``eval_open`` — open-ended answer
  extraction and fuzzy match (MMMU open split).
"""

from __future__ import annotations

import random
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


def _check_is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def _normalize_str(s: str) -> list[float | str]:
    """Normalize a string for open-ended answer comparison."""
    s = s.strip()
    if _check_is_number(s):
        return [round(float(s.replace(",", "")), 2)]
    return [s.lower()] if len(s) > 1 else [" " + s, s + " "]


def _extract_numbers(s: str) -> list[str]:
    """Extract all numbers (with commas, scientific notation, etc.) from *s*."""
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    return (
        re.findall(pattern_commas, s)
        + re.findall(pattern_scientific, s)
        + re.findall(pattern_simple, s)
    )


def _parse_open_answer_tag(response: str) -> str | None:
    """Try to extract the answer from an explicit 'Answer: ...' line.

    Supports formats like ``Answer: 42``, ``Answer: MgS``,
    ``Answer: \\boxed{13.0}``.  Returns ``None`` when no match is found.
    """
    matches = re.findall(
        r"[Aa]nswer\s*:\s*\*?\*?\s*(.+)",
        response,
    )
    if not matches:
        return None
    raw = matches[-1].strip().rstrip(".")
    # Unwrap \boxed{...} if present
    boxed = re.search(r"\\boxed\{(.+?)\}", raw)
    if boxed:
        raw = boxed.group(1)
    # Strip surrounding ** (bold markdown)
    raw = raw.strip("*").strip()
    return raw if raw else None


def parse_open_response(response: str) -> list[float | str]:
    """Extract answer candidates from an open-ended model response.

    First tries to extract from an explicit ``Answer: ...`` line.
    Falls back to heuristic key-subresponse extraction.
    """
    # Fast path: explicit "Answer: ..."
    tag_answer = _parse_open_answer_tag(response)
    if tag_answer is not None:
        out: list = []
        out.extend(_normalize_str(tag_answer))
        for num in _extract_numbers(tag_answer):
            out.extend(_normalize_str(num))
        return list(dict.fromkeys(out))

    # Fallback: heuristic extraction
    def _get_key_subresponses(resp: str) -> list[str]:
        resp = resp.strip().strip(".").lower()
        subs = re.split(r"\.\s(?=[A-Z])|\n", resp)
        indicators = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        keys: list[str] = []
        for i, s in enumerate(subs):
            cands = indicators + ["="] if i == len(subs) - 1 else indicators
            shortest = None
            for ind in cands:
                if ind in s:
                    part = s.split(ind)[-1].strip()
                    if not shortest or len(part) < len(shortest):
                        shortest = part
            if shortest and shortest not in (":", ",", ".", "!", "?", ";", "'"):
                keys.append(shortest)
        return keys or [resp]

    key_resps = _get_key_subresponses(response)
    pred_list = key_resps.copy()
    for r in key_resps:
        pred_list.extend(_extract_numbers(r))
    out = []
    for x in pred_list:
        out.extend(_normalize_str(x))
    return list(dict.fromkeys(out))


def eval_open(gold: str | list[str], preds: list[float | str]) -> bool:
    """Check if any prediction matches the gold answer (fuzzy)."""
    if isinstance(gold, list):
        norm_answers: list = []
        for ans in gold:
            norm_answers.extend(_normalize_str(ans))
    else:
        norm_answers = _normalize_str(gold)
    for p in preds:
        if isinstance(p, str):
            for na in norm_answers:
                if isinstance(na, str) and na in p:
                    return True
        else:
            if p in norm_answers:
                return True
    return False


def parse_multi_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: dict[str, str],
) -> tuple[str, bool]:
    """Extract a single answer letter from the model response.

    Priority: ``Answer: X`` → ``(A)`` bracket → ``·A·`` space-padded →
    option-text match → last-occurrence tie-break → random fallback.

    Returns ``(choice, is_fallback)``. ``is_fallback`` is ``True`` iff
    nothing could be parsed out of *response* and a random choice was
    returned — this counter is observational and doesn't change scoring
    behavior vs. the MMMU reference eval.
    """
    answer_matches = re.findall(r"[Aa]nswer\s*:\s*\*?\*?\s*\(?([A-Z])\)?", response)
    if answer_matches:
        candidate = answer_matches[-1]
        if candidate in all_choices:
            return candidate, False

    for char in (",", ".", "!", "?", ";", ":", "'"):
        response = response.strip(char)
    response = " " + response + " "

    candidates: list[str] = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if not candidates and len(response.split()) > 5:
        for idx, ans in index2ans.items():
            if ans and ans.lower() in response.lower():
                candidates.append(idx)
    if not candidates:
        return random.choice(all_choices), True
    if len(candidates) == 1:
        return candidates[0], False

    starts: list[int] = []
    for can in candidates:
        pos = response.rfind(f"({can})")
        if pos == -1:
            pos = response.rfind(f" {can} ")
        if pos == -1 and index2ans.get(can):
            pos = response.lower().rfind(index2ans[can].lower())
        starts.append(pos)
    return candidates[max(range(len(candidates)), key=starts.__getitem__)], False
