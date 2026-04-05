# SPDX-License-Identifier: Apache-2.0
"""MMMU dataset loader for VLM accuracy evaluation.

Loads the MMMU/MMMU validation split from HuggingFace Datasets and prepares
samples with base64-encoded images for the sglang-omni chat completions API.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass, field

from datasets import concatenate_datasets, load_dataset
from PIL import Image

logger = logging.getLogger(__name__)

DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}


@dataclass
class MMMUSample:
    sample_id: str
    question: str
    options: list[str]
    answer: str
    image_uris: list[str]
    subject: str
    prompt: str
    all_choices: list[str] = field(default_factory=list)
    index2ans: dict[str, str] = field(default_factory=dict)
    question_type: str = "multiple-choice"  # "multiple-choice" or "open"


def image_to_data_uri(image: Image.Image) -> str:
    """Convert a PIL Image to a ``data:image/png;base64,...`` URI."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _strip_image_tags(text: str) -> str:
    """Remove ``<image N>`` placeholders from MMMU question text.

    sglang-omni's preprocessor injects image tokens automatically based on
    the top-level ``images`` field, so inline placeholders must be removed
    to avoid confusing the model.
    """
    return re.sub(r"<image\s*\d+>", "", text).strip()


def format_mmmu_prompt(question: str, options: list[str]) -> str:
    """Format an MMMU prompt (multiple-choice or open-ended).

    Image placeholders (``<image 1>``, etc.) are stripped because
    sglang-omni handles image injection separately via the ``images``
    request field.

    For multiple-choice, returns::

        <question>
        A. <opt1>
        B. <opt2>
        ...

        Answer the following multiple-choice question. The last line of
        your response should be of the following format: 'Answer: $LETTER'
        (without quotes) where LETTER is one of the options. Think step
        by step before answering.

    For open-ended (empty *options*), returns::

        <question>

        Answer:
    """
    clean_question = _strip_image_tags(question)
    prompt = f"{clean_question}\n"
    if options:
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            prompt += f"{letter}. {opt}\n"
        prompt += (
            "\nAnswer the following multiple-choice question. "
            "The last line of your response should be of the "
            "following format: 'Answer: $LETTER' (without quotes) "
            "where LETTER is one of the options. "
            "Think step by step before answering."
        )
    else:
        prompt += "\nAnswer: "
    return prompt


def load_mmmu_samples(max_samples: int | None = None) -> list[MMMUSample]:
    """Load MMMU validation samples from HuggingFace.

    Downloads and caches via the ``datasets`` library. Deterministic selection:
    all subjects are loaded, merged, sorted by sample id, and the first
    *max_samples* are returned.
    """
    subjects: list[str] = []
    for subs in DOMAIN_CAT2SUB_CAT.values():
        subjects.extend(subs)

    ds_list = []
    for subj in subjects:
        try:
            d = load_dataset("MMMU/MMMU", subj, split="validation")
            d = d.add_column("__subject__", [subj] * len(d))
            ds_list.append(d)
        except Exception:
            logger.warning("Failed to load MMMU subject: %s", subj)
            continue
    if not ds_list:
        raise RuntimeError("Failed to load any MMMU subject datasets")

    merged = concatenate_datasets(ds_list)

    def _sort_key(idx: int) -> str:
        ex = merged[idx]
        return str(ex.get("id", f"{ex['__subject__']}:{idx}"))

    order = sorted(range(len(merged)), key=_sort_key)
    if max_samples is not None:
        order = order[:max_samples]

    samples: list[MMMUSample] = []
    for idx in order:
        ex = merged[idx]
        subject = ex["__subject__"]

        image = ex.get("image_1")
        if image is None or not hasattr(image, "convert"):
            continue

        data_uri = image_to_data_uri(image)
        question = ex.get("question", "")
        answer = ex.get("answer")

        raw_options = ex.get("options")
        options: list[str] = []
        if raw_options:
            try:
                options = (
                    raw_options
                    if isinstance(raw_options, list)
                    else list(eval(raw_options))
                )
            except Exception:
                options = []

        all_choices: list[str] = []
        index2ans: dict[str, str] = {}
        question_type = "open"

        if options:
            all_choices = [chr(ord("A") + i) for i in range(len(options))]
            index2ans = {chr(ord("A") + i): opt for i, opt in enumerate(options)}
            question_type = "multiple-choice"

        prompt = format_mmmu_prompt(question, options)

        samples.append(
            MMMUSample(
                sample_id=ex.get("id", f"{subject}:{idx}"),
                question=question,
                options=options,
                answer=answer,
                image_uris=[data_uri],
                subject=subject,
                prompt=prompt,
                all_choices=all_choices,
                index2ans=index2ans,
                question_type=question_type,
            )
        )

    logger.info("Loaded %d MMMU samples", len(samples))
    return samples


def _check_is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except Exception:
        return False


def _normalize_str(s: str) -> list:
    """Normalize a string for open-ended answer comparison."""
    s = s.strip()
    if _check_is_number(s):
        s = s.replace(",", "")
        try:
            return [round(float(s), 2)]
        except Exception:
            return [s.lower()]
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


def parse_open_response(response: str) -> list:
    """Extract answer candidates from an open-ended model response.

    Ported from SGLang ``simple_eval_mmmu_vlm.py``.
    """

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
            cands = [*indicators]
            if i == len(subs) - 1:
                cands.append("=")
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
    out: list = []
    for x in pred_list:
        out.extend(_normalize_str(x))
    return list(dict.fromkeys(out))


def eval_open(gold, preds: list) -> bool:
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
) -> str:
    """Extract a single answer letter from the model response.

    Ported from SGLang ``simple_eval_mmmu_vlm.py`` with the same extraction
    priority: ``(A)`` bracket → ``·A·`` space-padded → option-text match →
    last-occurrence tie-break → fallback to first choice.
    """
    # First, look for explicit "Answer: X" pattern (last occurrence)
    answer_matches = re.findall(
        r"[Aa]nswer\s*:\s*\*?\*?\s*\(?([A-Z])\)?", response
    )
    if answer_matches:
        candidate = answer_matches[-1]
        if candidate in all_choices:
            return candidate

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
        return all_choices[0]
    if len(candidates) == 1:
        return candidates[0]

    starts: list[int] = []
    for can in candidates:
        pos = response.rfind(f"({can})")
        if pos == -1:
            pos = response.rfind(f" {can} ")
        if pos == -1 and index2ans.get(can):
            pos = response.lower().rfind(index2ans[can].lower())
        starts.append(pos)
    return candidates[int(max(range(len(starts)), key=lambda i: starts[i]))]
