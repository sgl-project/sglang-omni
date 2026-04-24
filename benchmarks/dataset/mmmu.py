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
    images: list[Image.Image]
    subject: str
    prompt: str
    all_choices: list[str] = field(default_factory=list)
    index2ans: dict[str, str] = field(default_factory=dict)
    question_type: str = "multiple-choice"  # "multiple-choice" or "open"


def image_to_data_uri(image: Image.Image) -> str:
    """Convert a PIL Image to a data:image/png;base64,... URI."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _strip_image_tags(text: str) -> str:
    """Remove <image N> placeholders from MMMU question text.

    sglang-omni's preprocessor injects image tokens automatically based on
    the top-level images field, so inline placeholders must be removed
    to avoid confusing the model.
    """
    return re.sub(r"<image\s*\d+>", "", text).strip()


def format_mmmu_prompt(
    question: str,
    options: list[str],
    instruction_override: str | None = None,
) -> str:
    """Format an MMMU prompt (multiple-choice or open-ended).

    Image placeholders (<image 1>, etc.) are stripped because
    sglang-omni handles image injection separately via the ``images``
    request field.

    When *instruction_override* is provided for a multiple-choice
    sample, it replaces the default instruction block (callers are
    responsible for including the 'Answer: $LETTER' format directive).

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

        Answer the following question. The last line of your response
        should be of the following format: 'Answer: $ANSWER' ...
    """
    from benchmarks.tasks.visual_understand import MULTI_CHOICE_INSTRUCTION

    clean_question = _strip_image_tags(question)
    prompt = f"{clean_question}\n"
    if options:
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            prompt += f"{letter}. {opt}\n"
        if instruction_override is not None:
            prompt += f"\n{instruction_override}"
        else:
            prompt += MULTI_CHOICE_INSTRUCTION
    else:
        prompt += (
            "\nAnswer the following question. "
            "The last line of your response should be of the "
            "following format: 'Answer: $ANSWER' (without quotes) "
            "where $ANSWER is your final answer. "
            "Think step by step before answering."
        )
    return prompt


def _load_full_mmmu() -> list:
    """Load and merge all 30 subjects from MMMU/MMMU, sorted by sample id."""
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
    return merged.select(order)


def _dataset_to_samples(
    dataset,
    max_samples: int | None,
    instruction_override: str | None = None,
) -> list[MMMUSample]:
    """Convert HuggingFace dataset rows to MMMUSample objects."""
    samples: list[MMMUSample] = []
    for idx in range(len(dataset)):
        if max_samples is not None and len(samples) >= max_samples:
            break
        ex = dataset[idx]
        subject = ex.get("__subject__", "unknown")

        images: list[Image.Image] = []
        for i in range(1, 8):
            image = ex.get(f"image_{i}")
            if image is not None and hasattr(image, "convert"):
                images.append(image)
        if not images:
            continue

        question = ex.get("question", "")
        answer = ex.get("answer")

        raw_options = ex.get("options")
        options: list[str] = []
        if raw_options:
            if isinstance(raw_options, list):
                options = raw_options
            else:
                try:
                    options = list(ast.literal_eval(raw_options))
                except (ValueError, SyntaxError) as exc:
                    logger.warning(
                        f"Skipping MMMU sample {ex.get('id', idx)}: "
                        f"failed to parse options {raw_options!r}: {exc}"
                    )
                    continue

        all_choices: list[str] = []
        index2ans: dict[str, str] = {}
        question_type = "open"

        if options:
            all_choices = [chr(ord("A") + i) for i in range(len(options))]
            index2ans = {chr(ord("A") + i): opt for i, opt in enumerate(options)}
            question_type = "multiple-choice"

        prompt = format_mmmu_prompt(question, options, instruction_override)

        samples.append(
            MMMUSample(
                sample_id=ex.get("id", f"{subject}:{idx}"),
                question=question,
                options=options,
                answer=answer,
                images=images,
                subject=subject,
                prompt=prompt,
                all_choices=all_choices,
                index2ans=index2ans,
                question_type=question_type,
            )
        )

    return samples


def load_mmmu_samples(
    max_samples: int | None = None,
    *,
    repo_id: str | None = None,
    instruction_override: str | None = None,
) -> list[MMMUSample]:
    """Load MMMU validation samples.

    Args:
        max_samples: Cap on how many samples to return.  None = all.
        repo_id: HuggingFace dataset repo to load from.  Defaults to
            None which loads the full MMMU/MMMU (all 30 subjects,
            ~900 samples).  Pass a repo id like
            "zhaochenyang20/mmmu-ci-50" to load a pre-built subset.
        instruction_override: Optional replacement for the default
            multiple-choice instruction block (see format_mmmu_prompt).
    """
    if repo_id is not None:
        ds = load_dataset(repo_id, split="validation")
    else:
        ds = _load_full_mmmu()

    samples = _dataset_to_samples(ds, max_samples, instruction_override)
    logger.info(f"Loaded {len(samples)} MMMU samples")
    return samples
