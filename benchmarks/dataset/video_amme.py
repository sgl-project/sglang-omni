# SPDX-License-Identifier: Apache-2.0
"""Video-AMME dataset loader for video + audio question benchmarks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from benchmarks.dataset.videomme import VideoMMESample, format_videomme_prompt

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Ratish21/Video_AMME_ci"
DEFAULT_SOURCE_REPO_ID = "zhaochenyang20/Video_MME_ci"


@dataclass
class VideoAMMESample(VideoMMESample):
    audio_path: str = ""
    audio_text: str = ""
    source_repo_id: str = DEFAULT_SOURCE_REPO_ID
    source_video_path: str = ""
    audio_sample_rate: int | None = None
    audio_duration_s: float | None = None
    tts_model: str = ""


def _strip_option_prefix(option: str) -> str:
    return re.sub(r"^[A-D]\.\s*", "", option.strip())


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


def _snapshot_dir(repo_id: str) -> Path:
    local_path = Path(repo_id).expanduser()
    if local_path.exists():
        return local_path
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))


def _resolve_audio_path(snapshot_dir: Path, row: dict, sample_id: str) -> str | None:
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


def _resolve_video_path(
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

    source_repo_id = str(row.get("source_repo_id") or DEFAULT_SOURCE_REPO_ID).strip()
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


def _dataset_to_samples(
    dataset,
    *,
    snapshot_dir: Path,
    max_samples: int | None,
) -> list[VideoAMMESample]:
    source_snapshots: dict[str, Path] = {}
    samples: list[VideoAMMESample] = []
    for row_index, row in enumerate(dataset):
        question_id = str(row.get("question_id", f"videoamme:{row_index}")).strip()
        audio_path = _resolve_audio_path(snapshot_dir, row, question_id)
        video_path = _resolve_video_path(
            snapshot_dir,
            row,
            question_id,
            source_snapshots,
        )
        if not audio_path or not video_path:
            continue

        options = [_strip_option_prefix(str(option)) for option in row["options"]]
        all_choices = [chr(ord("A") + i) for i in range(len(options))]
        index2ans = {choice: option for choice, option in zip(all_choices, options)}
        question = str(row["question"]).strip()

        samples.append(
            VideoAMMESample(
                sample_id=question_id,
                video_path=video_path,
                audio_path=audio_path,
                audio_text=str(row.get("audio_text", "")).strip(),
                question=question,
                options=options,
                answer=str(row["answer"]).strip(),
                url=str(row.get("url", "")).strip(),
                video_id=str(row.get("video_id", "")).strip(),
                question_id=question_id,
                duration=str(row.get("duration", "short")).strip(),
                domain=str(row.get("domain", "unknown")).strip(),
                task_type=str(row.get("task_type", "understanding")).strip(),
                sub_category=str(row.get("sub_category", "")).strip(),
                source_video_path=str(row.get("source_video_path", "")).strip(),
                source_repo_id=str(
                    row.get("source_repo_id") or DEFAULT_SOURCE_REPO_ID
                ).strip(),
                audio_sample_rate=row.get("audio_sample_rate"),
                audio_duration_s=row.get("audio_duration_s"),
                tts_model=str(row.get("tts_model", "")).strip(),
                prompt=format_videomme_prompt(question, options),
                all_choices=all_choices,
                index2ans=index2ans,
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def load_video_amme_samples(
    max_samples: int | None = None,
    *,
    repo_id: str | None = None,
    split: str = "test",
) -> list[VideoAMMESample]:
    resolved_repo_id = repo_id or DEFAULT_REPO_ID
    snapshot_dir = _snapshot_dir(resolved_repo_id)
    dataset = _load_metadata_dataset(snapshot_dir, split)
    samples = _dataset_to_samples(
        dataset,
        snapshot_dir=snapshot_dir,
        max_samples=max_samples,
    )
    logger.info("Loaded %d Video-AMME samples", len(samples))
    return samples
