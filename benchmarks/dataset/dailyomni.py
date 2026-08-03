# SPDX-License-Identifier: Apache-2.0
"""Dataset loader for the Daily-Omni audio-visual QA benchmark."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tarfile
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from benchmarks.dataset.videomme import VideoAMMESample

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "liarliar/Daily-Omni"
QA_FILENAME = "qa.json"
MEDIA_ARCHIVE_FILENAME = "Videos.tar"
SUPPORTED_INPUT_MODES = ("all", "visual", "audio")


def _strip_choice_prefix(choice: str) -> str:
    return re.sub(r"^[A-Da-d][.):]\s*", "", choice.strip())


def format_dailyomni_prompt(
    question: str,
    choices: list[str],
    *,
    input_mode: str = "all",
) -> str:
    """Build the multiple-choice prompt used by the official evaluation."""
    task_prompt = {
        "all": "given video and audio together",
        "visual": "given video",
        "audio": "given audio",
    }.get(input_mode)
    if task_prompt is None:
        raise ValueError(
            f"Unsupported Daily-Omni input mode {input_mode!r}; "
            f"choose one of {SUPPORTED_INPUT_MODES}"
        )

    choice_lines = "\n".join(
        f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices)
    )
    return (
        "Your task is to accurately answer multiple-choice questions based on the "
        f"{task_prompt}.\n"
        "Select the single most accurate answer from the given choices.\n"
        f"Question: {question.strip()}\n"
        f"Choices:\n{choice_lines}\n"
        "Your answer should be a capital letter representing your choice: "
        "A, B, C, or D. Don't generate any other text."
    )


def _load_qa_rows(repo_id: str, qa_path: str | None) -> list[dict[str, Any]]:
    if qa_path:
        resolved_qa_path = Path(qa_path).expanduser().resolve()
    else:
        local_repo = Path(repo_id).expanduser()
        if local_repo.exists():
            resolved_qa_path = (local_repo.resolve() / QA_FILENAME).resolve()
        else:
            resolved_qa_path = Path(
                hf_hub_download(repo_id, QA_FILENAME, repo_type="dataset")
            )

    with resolved_qa_path.open(encoding="utf-8") as qa_file:
        rows = json.load(qa_file)
    if not isinstance(rows, list):
        raise ValueError(f"Daily-Omni metadata must be a JSON list: {resolved_qa_path}")
    return rows


def _safe_extract(archive_path: str | Path, target_dir: Path) -> None:
    """Extract regular files/directories while rejecting traversal and links."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            member_path = (target_root / member.name).resolve()
            try:
                member_path.relative_to(target_root)
            except ValueError as exc:
                raise ValueError(
                    f"Unsafe path in Daily-Omni archive: {member.name}"
                ) from exc
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    f"Unsupported entry in Daily-Omni archive: {member.name}"
                )
        archive.extractall(target_root)


def _media_root_from_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    videos_dir = path / "Videos"
    return videos_dir if videos_dir.is_dir() else path


def _ensure_media_root(repo_id: str, media_root: str | None) -> Path:
    if media_root:
        root = _media_root_from_dir(Path(media_root))
        if not root.is_dir():
            raise ValueError(f"Daily-Omni media root does not exist: {root}")
        return root

    local_repo = Path(repo_id).expanduser()
    if local_repo.exists():
        local_repo = local_repo.resolve()
        direct_root = local_repo / "Videos"
        if direct_root.is_dir():
            return direct_root
        archive_path = local_repo / MEDIA_ARCHIVE_FILENAME
        cache_key = (
            "local__" + hashlib.sha256(str(local_repo).encode("utf-8")).hexdigest()[:16]
        )
    else:
        archive_path = Path(
            hf_hub_download(
                repo_id,
                MEDIA_ARCHIVE_FILENAME,
                repo_type="dataset",
            )
        )
        cache_key = repo_id.replace("/", "__")

    if not archive_path.is_file():
        raise ValueError(f"Daily-Omni media archive does not exist: {archive_path}")

    cache_dir = Path.home() / ".cache" / "sglang-omni" / "dailyomni" / cache_key
    marker = cache_dir / ".extracted"
    if not marker.exists():
        logger.info("Extracting Daily-Omni media archive to %s", cache_dir)
        _safe_extract(archive_path, cache_dir)
        marker.write_text("ok\n", encoding="utf-8")

    root = _media_root_from_dir(cache_dir)
    if not root.is_dir():
        raise ValueError(f"Daily-Omni media root was not extracted: {root}")
    return root


def _resolve_media_paths(
    media_root: Path,
    video_id: str,
    input_mode: str,
) -> tuple[str, str] | None:
    video_dir = (media_root / video_id).resolve()
    try:
        video_dir.relative_to(media_root)
    except ValueError as exc:
        raise ValueError(f"Daily-Omni video_id escapes media root: {video_id}") from exc

    video_path = video_dir / f"{video_id}_video.mp4"
    audio_path = video_dir / f"{video_id}_audio.wav"
    required_paths = []
    if input_mode in {"all", "visual"}:
        required_paths.append(video_path)
    if input_mode in {"all", "audio"}:
        required_paths.append(audio_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        logger.warning(
            "Skipping Daily-Omni video %s because media is missing: %s",
            video_id,
            ", ".join(missing),
        )
        return None
    return str(video_path), str(audio_path)


def load_dailyomni_samples(
    max_samples: int | None = None,
    *,
    repo_id: str | None = None,
    split: str = "train",
    media_root: str | None = None,
    qa_path: str | None = None,
    input_mode: str = "all",
) -> list[VideoAMMESample]:
    """Load Daily-Omni metadata and resolve its separate video/audio files."""
    if split != "train":
        raise ValueError("Daily-Omni publishes one full split named 'train'")
    if input_mode not in SUPPORTED_INPUT_MODES:
        raise ValueError(
            f"Unsupported Daily-Omni input mode {input_mode!r}; "
            f"choose one of {SUPPORTED_INPUT_MODES}"
        )

    resolved_repo_id = repo_id or DEFAULT_REPO_ID
    rows = _load_qa_rows(resolved_repo_id, qa_path)
    root = _ensure_media_root(resolved_repo_id, media_root)
    samples: list[VideoAMMESample] = []

    for row_index, row in enumerate(rows):
        question = str(row.get("Question") or "").strip()
        video_id = str(row.get("video_id") or "").strip()
        task_type = str(row.get("Type") or "").strip()
        duration = str(row.get("video_duration") or "").strip()
        domain = str(row.get("video_category") or "").strip()
        raw_choices = row.get("Choice")
        answer = str(row.get("Answer") or "").strip().upper()

        if not all((question, video_id, task_type, duration, domain)):
            logger.warning(
                "Skipping Daily-Omni row %d because required metadata is missing",
                row_index,
            )
            continue
        if not isinstance(raw_choices, list) or len(raw_choices) != 4:
            logger.warning(
                "Skipping Daily-Omni row %d because it does not have four choices",
                row_index,
            )
            continue

        choices = [_strip_choice_prefix(str(choice)) for choice in raw_choices]
        all_choices = ["A", "B", "C", "D"]
        if answer not in all_choices:
            logger.warning(
                "Skipping Daily-Omni row %d with invalid answer %r",
                row_index,
                answer,
            )
            continue

        media_paths = _resolve_media_paths(root, video_id, input_mode)
        if media_paths is None:
            continue
        video_path, audio_path = media_paths
        question_id = f"dailyomni:{row_index}"
        samples.append(
            VideoAMMESample(
                sample_id=question_id,
                video_path=video_path,
                audio_path=audio_path,
                question=question,
                options=choices,
                answer=answer,
                video_id=video_id,
                question_id=question_id,
                duration=duration,
                domain=domain,
                task_type=task_type,
                sub_category=str(
                    row.get("content_fine_category")
                    or row.get("content_parent_category")
                    or ""
                ).strip(),
                prompt=format_dailyomni_prompt(
                    question,
                    choices,
                    input_mode=input_mode,
                ),
                all_choices=all_choices,
                index2ans=dict(zip(all_choices, choices)),
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    logger.info("Loaded %d Daily-Omni samples", len(samples))
    return samples
