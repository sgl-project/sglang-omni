# SPDX-License-Identifier: Apache-2.0
"""MMMU dataset loader for VLM accuracy evaluation.

Loads the MMMU/MMMU validation split from HuggingFace Datasets and prepares
samples with base64-encoded images for the sglang-omni chat completions API.
"""

from __future__ import annotations

import ast
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

    Memory note: All images are eagerly converted to base64 data URIs and held
    in memory. Each sample carries ~2MB of base64 data. The full validation set
    (~900 samples) would reach ~1.8GB. Use *max_samples* to cap memory usage in
    constrained environments.
    """
    subjects: list[str] = []
    for subs in DOMAIN_CAT2SUB_CAT.values():
        subjects.extend(subs)

    ds_list = []
    for subj in subjects:
        d = load_dataset("MMMU/MMMU", subj, split="validation")
        d = d.add_column("__subject__", [subj] * len(d))
        ds_list.append(d)

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

        image_uris: list[str] = []
        for i in range(1, 8):
            image = ex.get(f"image_{i}")
            if image is not None and hasattr(image, "convert"):
                image_uris.append(image_to_data_uri(image))
        if not image_uris:
            continue

        question = ex.get("question", "")
        answer = ex.get("answer")

        raw_options = ex.get("options")
        options: list[str] = []
        if raw_options:
            try:
                options = (
                    raw_options
                    if isinstance(raw_options, list)
                    else list(ast.literal_eval(raw_options))
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
                image_uris=image_uris,
                subject=subject,
                prompt=prompt,
                all_choices=all_choices,
                index2ans=index2ans,
                question_type=question_type,
            )
        )

    logger.info("Loaded %d MMMU samples", len(samples))
    return samples
