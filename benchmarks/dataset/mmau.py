# SPDX-License-Identifier: Apache-2.0
"""Dataset loader for MMAU."""

from __future__ import annotations

import ast
import json
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Audio, load_dataset

from benchmarks.dataset.mmsu import normalize_text


@dataclass
class MmauSample:
    sample_id: str
    audio_path: str
    question: str
    choices: list[str]
    answer_text: str
    answer_index: int | None
    task_name: str
    category: str
    sub_category: str
    sub_sub_category: str = ""
    linguistics_sub_discipline: str = ""
    difficulty: str = ""
    dataset: str = ""


def _parse_choices(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
    matches = re.findall(
        r"(?:^|\n)\s*[A-Ja-j][\).:\s]+(.+?)(?=\n\s*[A-Ja-j][\).:\s]+|\Z)", text, re.S
    )
    if matches:
        return [match.strip() for match in matches]
    return [part.strip() for part in re.split(r"\n+|\s*\|\s*", text) if part.strip()]


def _match_answer(choices: list[str], answer: str) -> int | None:
    answer = answer.strip()
    if len(answer) == 1 and answer.upper().isalpha():
        index = ord(answer.upper()) - ord("A")
        if 0 <= index < len(choices):
            return index
    norm = normalize_text(answer)
    for i, choice in enumerate(choices):
        if normalize_text(choice) == norm:
            return i
    return None


def _dump_audio(cache_dir: Path, sample_id: str, audio: dict[str, Any]) -> str:
    suffix = Path(str(audio.get("path") or "")).suffix or ".wav"
    path = cache_dir / f"{sample_id}{suffix}"
    if not path.exists():
        path.write_bytes(audio["bytes"])
    return str(path)


def load_mmau_samples(
    max_samples: int | None = None,
    categories: list[str] | None = None,
    tasks: list[str] | None = None,
    seed: int | None = None,
    *,
    repo_id: str = "lmms-lab/mmau",
    split: str = "test_mini",
) -> list[MmauSample]:
    load_kwargs: dict[str, Any] = {"split": split}
    if repo_id == "lmms-lab/mmau" and split in {"test", "test_mini"}:
        load_kwargs["data_files"] = {split: f"data/{split}-*.parquet"}
        load_kwargs["verification_mode"] = "no_checks"
    ds = load_dataset(repo_id, **load_kwargs)
    ds = ds.cast_column("audio", Audio(decode=False))
    category_set = {c.strip() for c in (categories or []) if c.strip()} or None
    task_set = {t.strip() for t in (tasks or []) if t.strip()} or None
    cache_dir = Path(tempfile.mkdtemp(prefix="mmau_audio_"))

    samples: list[MmauSample] = []
    for row in ds:
        category = str(row.get("category", "")).strip()
        task = str(row.get("task", "")).strip()
        if category_set and category not in category_set:
            continue
        if task_set and task not in task_set:
            continue
        choices = _parse_choices(row["choices"])
        answer = str(row["answer"]).strip()
        sample_id = str(row["id"])
        samples.append(
            MmauSample(
                sample_id=sample_id,
                audio_path=_dump_audio(cache_dir, sample_id, row["audio"]),
                question=str(row["question"]).strip(),
                choices=choices,
                answer_text=answer,
                answer_index=_match_answer(choices, answer),
                task_name=task,
                category=category,
                sub_category=str(
                    row.get("sub-category") or row.get("sub_category") or ""
                ).strip(),
                difficulty=str(row.get("difficulty", "")).strip(),
                dataset=str(row.get("dataset", "")).strip(),
            )
        )

    if seed is not None and len(samples) > 1:
        random.Random(seed).shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples
