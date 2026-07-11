# SPDX-License-Identifier: Apache-2.0
"""Dataset loader for MMAR."""

from __future__ import annotations

import logging
import os
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from benchmarks.dataset.mmsu import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class MmarSample:
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
    modality: str = ""
    language: str = ""
    source: str = ""
    url: str = ""
    timestamp: str = ""


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
    candidates = [
        i
        for i, choice in enumerate(choices)
        if norm and (norm in normalize_text(choice) or normalize_text(choice) in norm)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _safe_extract(tar_path: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            member_path = (target_root / member.name).resolve()
            if os.path.commonpath([target_root, member_path]) != str(target_root):
                raise ValueError(f"Unsafe path in MMAR archive: {member.name}")
        tar.extractall(target_root)


def _ensure_audio_root(repo_id: str, audio_root: str | None) -> Path:
    if audio_root:
        return Path(audio_root).expanduser().resolve()

    cache_dir = (
        Path.home() / ".cache" / "sglang-omni" / "mmar" / repo_id.replace("/", "__")
    )
    marker = cache_dir / ".extracted"
    if not marker.exists():
        archive = hf_hub_download(repo_id, "mmar-audio.tar.gz", repo_type="dataset")
        _safe_extract(archive, cache_dir)
        marker.write_text("ok")
    return cache_dir


def _resolve_audio_path(audio_root: Path, audio_path: str) -> str:
    rel = audio_path.strip().removeprefix("./")
    path = (audio_root / rel).resolve()
    try:
        path.relative_to(audio_root)
    except ValueError as exc:
        raise ValueError(f"MMAR audio path escapes root: {audio_path}") from exc
    return str(path)


def load_mmar_samples(
    max_samples: int | None = None,
    categories: list[str] | None = None,
    modalities: list[str] | None = None,
    seed: int | None = None,
    *,
    repo_id: str = "BoJack/MMAR",
    split: str = "test",
    audio_root: str | None = None,
) -> list[MmarSample]:
    ds = load_dataset(repo_id, split=split)
    root = _ensure_audio_root(repo_id, audio_root)
    category_set = {c.strip() for c in (categories or []) if c.strip()} or None
    modality_set = {m.strip() for m in (modalities or []) if m.strip()} or None

    samples: list[MmarSample] = []
    for row in ds:
        category = str(row.get("category", "")).strip()
        modality = str(row.get("modality", "")).strip()
        if category_set and category not in category_set:
            continue
        if modality_set and modality not in modality_set:
            continue
        choices = [str(choice).strip() for choice in row["choices"]]
        answer = str(row["answer"]).strip()
        answer_index = _match_answer(choices, answer)
        if answer_index is None:
            logger.warning(
                "Skipping MMAR sample %s with unmatched answer %r",
                row.get("id"),
                answer,
            )
            continue
        samples.append(
            MmarSample(
                sample_id=str(row["id"]),
                audio_path=_resolve_audio_path(root, str(row["audio_path"])),
                question=str(row["question"]).strip(),
                choices=choices,
                answer_text=answer,
                answer_index=answer_index,
                task_name=modality,
                category=category,
                sub_category=str(
                    row.get("sub-category") or row.get("sub_category") or ""
                ).strip(),
                modality=modality,
                language=str(row.get("language", "")).strip(),
                source=str(row.get("source", "")).strip(),
                url=str(row.get("url", "")).strip(),
                timestamp=str(row.get("timestamp", "")).strip(),
            )
        )

    if seed is not None and len(samples) > 1:
        random.Random(seed).shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples
