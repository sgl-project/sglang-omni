# SPDX-License-Identifier: Apache-2.0
"""Pinned loader for the Ming Freeform Audio Edit benchmark."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from benchmarks.dataset.prepare import (
    MING_FREEFORM_AUDIO_EDIT_DATASET_ID,
    MING_FREEFORM_AUDIO_EDIT_DATASET_REVISION,
)

_COLUMNS = ("file_name", "path", "instruction", "original_text", "edited_text")
_SEMANTIC_TASK_DIRS = {
    "deletion": "del",
    "insertion": "ins",
    "substitution": "sub",
}
_ACOUSTIC_TASK_DIRS = {
    "time_stretch": "time_stretch",
    "pitch_shift": "pitch_shift",
    "volume": "vol",
    "emotion": "emotion",
    "dialect": "dialect",
}
SUPPORTED_TASKS = tuple((*_SEMANTIC_TASK_DIRS, *_ACOUSTIC_TASK_DIRS))
_SCALE_RE = re.compile(r"\b(?:speed|volume)\s+to\s+([0-9]+(?:\.[0-9]+)?)\s*$", re.I)


@dataclass(frozen=True)
class SpeechEditSample:
    """One source audio, edit instruction, and expected transcript."""

    sample_id: str
    task: str
    language: str
    source_audio: str
    source_audio_repo_path: str
    source_audio_url: str
    instruction: str
    original_text: str
    edited_text: str
    scale: float | None = None


def metadata_path(task: str, language: str, semantic_subset: str = "basic") -> str:
    """Return the repository path for a supported task's metadata file."""
    if language not in {"en", "zh"}:
        raise ValueError("language must be 'en' or 'zh'")
    if semantic_subset not in {"basic", "full"}:
        raise ValueError("semantic_subset must be 'basic' or 'full'")

    if task in _SEMANTIC_TASK_DIRS:
        suffix = "_basic" if semantic_subset == "basic" else ""
        directory = _SEMANTIC_TASK_DIRS[task]
        return f"meta/{directory}/meta_{language}_{task}{suffix}.csv"
    if task in _ACOUSTIC_TASK_DIRS:
        if task == "dialect" and language != "zh":
            raise ValueError("the dialect task is available only in Chinese")
        directory = _ACOUSTIC_TASK_DIRS[task]
        filename_task = "vol" if task == "volume" else task
        return f"meta/{directory}/meta_{language}_{filename_task}.csv"
    raise ValueError(f"unsupported task {task!r}; choose from {SUPPORTED_TASKS}")


def load_ming_freeform_samples(
    task: str,
    *,
    language: str = "en",
    semantic_subset: str = "basic",
    max_samples: int | None = None,
    repo_id: str = MING_FREEFORM_AUDIO_EDIT_DATASET_ID,
    revision: str | None = None,
) -> list[SpeechEditSample]:
    """Load metadata and source WAVs from a pinned Hugging Face revision."""
    if max_samples is not None and max_samples <= 0:
        return []
    if revision is None and repo_id == MING_FREEFORM_AUDIO_EDIT_DATASET_ID:
        revision = MING_FREEFORM_AUDIO_EDIT_DATASET_REVISION

    from huggingface_hub import hf_hub_download

    meta_repo_path = metadata_path(task, language, semantic_subset)
    download_kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
    }
    meta_file = hf_hub_download(filename=meta_repo_path, **download_kwargs)
    rows = _read_metadata(Path(meta_file), meta_repo_path)

    samples: list[SpeechEditSample] = []
    for row_number, row in rows:
        sample_id, audio_repo_path, instruction, original_text, edited_text = row
        _validate_row(
            sample_id,
            audio_repo_path,
            instruction,
            original_text,
            edited_text,
            meta_repo_path=meta_repo_path,
            row_number=row_number,
        )
        source_audio = hf_hub_download(filename=audio_repo_path, **download_kwargs)
        samples.append(
            SpeechEditSample(
                sample_id=sample_id,
                task=task,
                language=language,
                source_audio=source_audio,
                source_audio_repo_path=audio_repo_path,
                source_audio_url=_resolve_url(repo_id, revision, audio_repo_path),
                instruction=instruction,
                original_text=original_text,
                edited_text=edited_text,
                scale=_instruction_scale(task, instruction),
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def _read_metadata(path: Path, repo_path: str) -> list[tuple[int, tuple[str, ...]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            tuple(value.strip() for value in row)
            for row in csv.reader(handle, delimiter="|")
        ]
    rows = [row for row in rows if any(row)]
    first_data_line = 1
    if rows and rows[0] == _COLUMNS:
        rows = rows[1:]
        first_data_line = 2
    numbered_rows = list(enumerate(rows, start=first_data_line))
    for row_number, row in numbered_rows:
        if len(row) != len(_COLUMNS):
            raise ValueError(
                f"Expected {len(_COLUMNS)} columns in {repo_path} row {row_number}, "
                f"found {len(row)}"
            )
    return numbered_rows


def _validate_row(
    sample_id: str,
    audio_repo_path: str,
    instruction: str,
    original_text: str,
    edited_text: str,
    *,
    meta_repo_path: str,
    row_number: int,
) -> None:
    if not all((sample_id, instruction, original_text, edited_text)):
        raise ValueError(f"Empty required field in {meta_repo_path} row {row_number}")
    sample_path = PurePosixPath(sample_id)
    if sample_path.name != sample_id or sample_id in {".", ".."}:
        raise ValueError(f"Invalid sample id {sample_id!r} in {meta_repo_path}")
    audio_path = PurePosixPath(audio_repo_path)
    if (
        audio_path.is_absolute()
        or ".." in audio_path.parts
        or len(audio_path.parts) != 2
        or audio_path.parts[0] != "wavs"
        or audio_path.suffix.lower() != ".wav"
    ):
        raise ValueError(
            f"Invalid source audio path {audio_repo_path!r} in {meta_repo_path}"
        )


def _instruction_scale(task: str, instruction: str) -> float | None:
    if task not in {"time_stretch", "volume"}:
        return None
    match = _SCALE_RE.search(instruction)
    if match is None:
        raise ValueError(f"Cannot parse {task} scale from instruction {instruction!r}")
    scale = float(match.group(1))
    if scale <= 0:
        raise ValueError(f"{task} scale must be positive, found {scale}")
    return scale


def _resolve_url(repo_id: str, revision: str | None, repo_path: str) -> str:
    encoded_path = quote(repo_path, safe="/")
    encoded_revision = quote(revision or "main", safe="")
    return (
        f"https://huggingface.co/datasets/{repo_id}/resolve/"
        f"{encoded_revision}/{encoded_path}"
    )
