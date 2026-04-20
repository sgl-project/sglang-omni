# SPDX-License-Identifier: Apache-2.0
"""Dataset loader for MMSU (audio multiple-choice benchmark)."""

from __future__ import annotations

import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from datasets import Audio, load_dataset


@dataclass
class MmsuSample:
    sample_id: str
    audio_path: str
    question: str
    choices: list[str]
    answer_text: str
    answer_index: int | None
    task_name: str
    category: str
    sub_category: str
    sub_sub_category: str
    linguistics_sub_discipline: str


def normalize_text(text: str) -> str:
    """Lowercase, strip non-word chars, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text.lower())).strip()


def _match_answer(choices: list[str], answer: str) -> int | None:
    norm = normalize_text(answer)
    for i, c in enumerate(choices):
        if normalize_text(c) == norm:
            return i
    return None


def _dump_audio(cache_dir: Path, sample_id: str, audio_bytes: bytes) -> str:
    """Write audio bytes to cache and return the path."""
    path = cache_dir / f"{sample_id}.mp3"
    if not path.exists():
        path.write_bytes(audio_bytes)
    return str(path)


def load_mmsu_samples(
    max_samples: int | None = None,
    task_names: list[str] | None = None,
    categories: list[str] | None = None,
    seed: int | None = None,
    *,
    repo_id: str | None = None,
) -> list[MmsuSample]:
    """Load MMSU samples.


    Note (Yifei, Chenyang):
    repo_id defaults to None which loads the full ddwang2000/MMSU
    (train split, ~5000 samples).  zhaochenyang20/mmsu-ci-2000 to
    load our pre-built subset for CI.
    """

    ds = load_dataset(repo_id or "ddwang2000/MMSU")
    assert list(ds.keys()) == [
        "train"
    ], f"Expected only 'train' split, got {list(ds.keys())}"
    ds = ds.cast_column("audio", Audio(decode=False))
    task_set = {t.strip() for t in (task_names or []) if t.strip()} or None
    cat_set = {c.strip() for c in (categories or []) if c.strip()} or None
    cache_dir = Path(tempfile.mkdtemp(prefix="mmsu_audio_"))

    samples: list[MmsuSample] = []
    for row in ds["train"]:
        if task_set and row["task_name"] not in task_set:
            continue
        if cat_set and row["category"] not in cat_set:
            continue
        choices = [
            str(row[k]).strip()
            for k in ("choice_a", "choice_b", "choice_c", "choice_d")
        ]
        answer = str(row["answer_gt"]).strip()
        sid = str(row["id"])
        samples.append(
            MmsuSample(
                sample_id=sid,
                audio_path=_dump_audio(cache_dir, sid, row["audio"]["bytes"]),
                question=str(row["question"]).strip(),
                choices=choices,
                answer_text=answer,
                answer_index=_match_answer(choices, answer),
                task_name=row["task_name"],
                category=row["category"],
                sub_category=str(row["sub-category"]).strip(),
                sub_sub_category=str(row["sub-sub-category"]).strip(),
                linguistics_sub_discipline=str(
                    row["linguistics_sub_discipline"]
                ).strip(),
            )
        )

    if seed is not None and len(samples) > 1:
        random.Random(seed).shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples
