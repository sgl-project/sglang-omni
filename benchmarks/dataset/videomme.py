# SPDX-License-Identifier: Apache-2.0
"""Video-MME family dataset loaders for local benchmarks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "zhaochenyang20/Video_MME"
DEFAULT_VIDEOAMME_REPO_ID = "zhaochenyang20/Video_AMME_ci"
DEFAULT_VIDEOAMME_SOURCE_REPO_ID = "zhaochenyang20/Video_MME_ci"


@dataclass
class VideoMMESample:
    sample_id: str
    video_path: str
    question: str
    options: list[str]
    answer: str
    url: str = ""
    video_id: str = ""
    question_id: str = ""
    duration: str = "short"
    domain: str = "unknown"
    task_type: str = "understanding"
    sub_category: str = ""
    prompt: str = ""
    all_choices: list[str] = field(default_factory=list)
    index2ans: dict[str, str] = field(default_factory=dict)


@dataclass
class VideoAMMESample(VideoMMESample):
    audio_path: str = field(kw_only=True)


def _strip_option_prefix(option: str) -> str:
    return re.sub(r"^[A-D]\.\s*", "", option.strip())


def format_videomme_prompt(question: str, options: list[str]) -> str:
    from benchmarks.tasks.visual_understand import MULTI_CHOICE_INSTRUCTION

    prompt = f"{question.strip()}\n"
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        prompt += f"{letter}. {option}\n"
    prompt += MULTI_CHOICE_INSTRUCTION
    return prompt


def _snapshot_dir(repo_id: str) -> Path:
    local_path = Path(repo_id).expanduser()
    if local_path.exists():
        return local_path
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))


def _resolve_video_path(snapshot_dir: Path, row: dict, question_id: str) -> str | None:
    relative_path = row.get("video_path")
    if not relative_path:
        logger.warning(
            f"Skipping Video-MME sample {question_id} because the dataset row has no video_path",
        )
        return None
    absolute_path = snapshot_dir / str(relative_path)
    if not absolute_path.exists():
        logger.warning(
            f"Skipping Video-MME sample {question_id} because the video file does not exist at {absolute_path}"
        )
        return None
    return str(absolute_path)


def _resolve_videoamme_audio_path(
    snapshot_dir: Path, row: dict, sample_id: str
) -> str | None:
    relative_path = row.get("audio_path")
    if not relative_path:
        logger.warning("Skipping Video-AMME sample %s without audio_path", sample_id)
        return None

    audio_path = snapshot_dir / str(relative_path)
    if not audio_path.exists():
        logger.warning(
            "Skipping Video-AMME sample %s because audio file is missing at %s",
            sample_id,
            audio_path,
        )
        return None
    return str(audio_path)


def _resolve_videoamme_video_path(
    snapshot_dir: Path,
    row: dict,
    sample_id: str,
    source_snapshots: dict[str, Path],
) -> str | None:
    local_relative_path = row.get("video_path")
    if local_relative_path:
        local_video_path = snapshot_dir / str(local_relative_path)
        if local_video_path.exists():
            return str(local_video_path)

    source_repo_id = str(
        row.get("source_repo_id") or DEFAULT_VIDEOAMME_SOURCE_REPO_ID
    ).strip()
    source_video_path = str(row.get("source_video_path") or "").strip()
    if not source_video_path:
        logger.warning(
            "Skipping Video-AMME sample %s without source_video_path", sample_id
        )
        return None

    if source_repo_id not in source_snapshots:
        source_snapshots[source_repo_id] = _snapshot_dir(source_repo_id)
    video_path = source_snapshots[source_repo_id] / source_video_path
    if not video_path.exists():
        logger.warning(
            "Skipping Video-AMME sample %s because source video is missing at %s",
            sample_id,
            video_path,
        )
        return None
    return str(video_path)


def _build_sample_kwargs(row, *, question_id: str, video_path: str) -> dict:
    options = [_strip_option_prefix(str(option)) for option in row["options"]]
    all_choices = [chr(ord("A") + i) for i in range(len(options))]
    index2ans = {choice: option for choice, option in zip(all_choices, options)}
    question = str(row["question"]).strip()
    return {
        "sample_id": question_id,
        "video_path": video_path,
        "question": question,
        "options": options,
        "answer": str(row["answer"]).strip(),
        "url": str(row.get("url", "")).strip(),
        "video_id": str(row.get("video_id", "")).strip(),
        "question_id": question_id,
        "duration": str(row.get("duration", "short")).strip(),
        "domain": str(row.get("domain", "unknown")).strip(),
        "task_type": str(row.get("task_type", "understanding")).strip(),
        "sub_category": str(row.get("sub_category", "")).strip(),
        "prompt": format_videomme_prompt(question, options),
        "all_choices": all_choices,
        "index2ans": index2ans,
    }


def _dataset_to_samples(dataset, *, max_samples: int | None, build_sample):
    samples: list[VideoMMESample] = []
    for row_index, row in enumerate(dataset):
        sample = build_sample(row_index, row)
        if sample is None:
            continue
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def _load_metadata_dataset(snapshot_dir: Path, split: str):
    data_dir = snapshot_dir / "data"
    split_parts = sorted(data_dir.glob(f"{split}_part_*.jsonl"))
    if split_parts:
        return load_dataset(
            "json",
            data_files=[str(path) for path in split_parts],
            split="train",
        )

    split_file = data_dir / f"{split}.jsonl"
    if split_file.exists():
        return load_dataset("json", data_files=str(split_file), split="train")

    available = sorted(path.name for path in data_dir.glob("*.jsonl"))
    raise ValueError(
        f"Split '{split}' not found under {data_dir}. Available files: {available}"
    )


def load_videomme_samples(
    max_samples: int | None = None,
    *,
    repo_id: str | None = None,
    split: str = "test",
) -> list[VideoMMESample]:
    resolved_repo_id = repo_id or DEFAULT_REPO_ID
    snapshot_dir = Path(
        snapshot_download(repo_id=resolved_repo_id, repo_type="dataset")
    )
    dataset = _load_metadata_dataset(snapshot_dir, split)

    def build_sample(row_index, row):
        question_id = str(row.get("question_id", f"videomme:{row_index}")).strip()
        video_path = _resolve_video_path(snapshot_dir, row, question_id)
        if not video_path:
            return None
        return VideoMMESample(
            **_build_sample_kwargs(row, question_id=question_id, video_path=video_path)
        )

    samples = _dataset_to_samples(
        dataset,
        max_samples=max_samples,
        build_sample=build_sample,
    )
    logger.info(f"Loaded {len(samples)} Video-MME samples")
    return samples


def load_videoamme_samples(
    max_samples: int | None = None,
    *,
    repo_id: str | None = None,
    split: str = "test",
) -> list[VideoAMMESample]:
    resolved_repo_id = repo_id or DEFAULT_VIDEOAMME_REPO_ID
    snapshot_dir = _snapshot_dir(resolved_repo_id)
    dataset = _load_metadata_dataset(snapshot_dir, split)
    source_snapshots: dict[str, Path] = {}

    def build_sample(row_index, row):
        question_id = str(row.get("question_id", f"videoamme:{row_index}")).strip()
        audio_path = _resolve_videoamme_audio_path(snapshot_dir, row, question_id)
        video_path = _resolve_videoamme_video_path(
            snapshot_dir,
            row,
            question_id,
            source_snapshots,
        )
        if not audio_path or not video_path:
            return None
        return VideoAMMESample(
            **_build_sample_kwargs(row, question_id=question_id, video_path=video_path),
            audio_path=audio_path,
        )

    samples = _dataset_to_samples(
        dataset,
        max_samples=max_samples,
        build_sample=build_sample,
    )
    logger.info("Loaded %d Video-AMME samples", len(samples))
    return samples
