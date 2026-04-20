# SPDX-License-Identifier: Apache-2.0
"""SocialOmni dataset helpers.

Single-file layout aligned with the existing benchmark modules in this repo.
Sections below cover:
- frozen mini manifest
- dataset dataclasses
- full + mini materialization
- level1 loader
- level2 loader
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SOCIALOMNI_REPO_ID = "alexisty/SocialOmni"
SOCIALOMNI_DATASET_NAMES = ("socialomni", "socialomni-mini")
DEFAULT_SOCIALOMNI_DIRS = {
    "socialomni": "socialomni",
    "socialomni-mini": "socialomni-mini",
}

SOCIALOMNI_MINI_LEVEL1 = (1, 2, 3, 4)
SOCIALOMNI_MINI_LEVEL2 = (
    "video_0001",  # Q1 gold YES path
    "video_0002",  # Q1 gold YES path
    "video_0005",  # Q1 gold NO path
    "video_0007",  # Q1 gold NO path
)
SOCIALOMNI_MINI_MANIFEST: dict[str, Any] = {
    "dataset_name": "socialomni-mini",
    "source_dataset": "socialomni",
    "description": (
        "Deterministic SocialOmni smoke-test subset with both level1 coverage "
        "and level2 YES/NO branch coverage."
    ),
    "level1": {
        "sample_ids": list(SOCIALOMNI_MINI_LEVEL1),
        "coverage": ["multiple level1 MCQ samples"],
    },
    "level2": {
        "video_ids": list(SOCIALOMNI_MINI_LEVEL2),
        "coverage": [
            "at least one Q1 gold YES sample",
            "at least one Q1 gold NO sample",
        ],
    },
}

LEVEL1_DATASET_PATH = Path("level_1") / "dataset.json"
LEVEL1_VIDEOS_DIR = Path("level_1") / "videos"
LEVEL2_DATASET_PATH = Path("level_2") / "annotations.json"
LEVEL2_VIDEOS_DIR = Path("level_2") / "videos"


@dataclass(frozen=True)
class SocialOmniLevel1Sample:
    sample_id: str
    video_path: str
    question: str
    options: list[str]
    correct_answer: str
    asr_content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SocialOmniLevel2Question1:
    question: str
    timestamp: str
    option_a: str
    option_b: str
    correct_answer: str


@dataclass(frozen=True)
class SocialOmniLevel2Question2:
    question: str
    answer: str


@dataclass(frozen=True)
class SocialOmniLevel2Sample:
    sample_id: str
    video_path: str
    original_video_id: str
    source_dir: str
    question_1: SocialOmniLevel2Question1
    question_2: SocialOmniLevel2Question2
    metadata: dict[str, Any]
    full_asr: str


def get_socialomni_local_dir(
    dataset_name: str,
    local_dir: str | None = None,
) -> Path:
    """Resolve the local root for a SocialOmni dataset variant."""
    if dataset_name not in SOCIALOMNI_DATASET_NAMES:
        raise ValueError(f"Unsupported SocialOmni dataset: {dataset_name}")
    return Path(local_dir or DEFAULT_SOCIALOMNI_DIRS[dataset_name])


def write_socialomni_mini_manifest(root_dir: str | Path) -> Path:
    """Write the frozen mini manifest to *root_dir* and return the path."""
    path = Path(root_dir) / "mini_manifest.json"
    path.write_text(
        json.dumps(SOCIALOMNI_MINI_MANIFEST, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def get_level1_dataset_path(root_dir: str | Path) -> Path:
    return Path(root_dir) / LEVEL1_DATASET_PATH


def get_level1_videos_dir(root_dir: str | Path) -> Path:
    return Path(root_dir) / LEVEL1_VIDEOS_DIR


def get_level2_dataset_path(root_dir: str | Path) -> Path:
    return Path(root_dir) / LEVEL2_DATASET_PATH


def get_level2_videos_dir(root_dir: str | Path) -> Path:
    return Path(root_dir) / LEVEL2_VIDEOS_DIR


def _normalize_source_root(source_root: str | Path) -> Path:
    root = Path(source_root)
    if (root / "level_1").is_dir() and (root / "level_2").is_dir():
        return root
    if (root / "data" / "level_1").is_dir() and (root / "data" / "level_2").is_dir():
        return root / "data"
    raise FileNotFoundError(
        f"SocialOmni source root must contain level_1/ and level_2/ directories: {root}"
    )


def _download_socialomni_source(tmpdir: str | Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "SocialOmni dataset preparation requires huggingface_hub in the project environment."
        ) from exc

    snapshot_download(
        repo_id=SOCIALOMNI_REPO_ID,
        repo_type="dataset",
        local_dir=str(tmpdir),
        allow_patterns=["data/level_1/**", "data/level_2/**"],
    )
    return _normalize_source_root(tmpdir)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _dataset_ready(root_dir: Path, *, mini: bool) -> bool:
    level1_dataset = root_dir / LEVEL1_DATASET_PATH
    level2_dataset = root_dir / LEVEL2_DATASET_PATH
    level1_videos = root_dir / LEVEL1_VIDEOS_DIR
    level2_videos = root_dir / LEVEL2_VIDEOS_DIR
    if not all(
        [
            level1_dataset.is_file(),
            level2_dataset.is_file(),
            level1_videos.is_dir(),
            level2_videos.is_dir(),
        ]
    ):
        return False
    if mini and not (root_dir / "mini_manifest.json").is_file():
        return False
    return True


def _materialize_full_dataset(source_root: Path, target_root: Path) -> None:
    for relative in (LEVEL1_DATASET_PATH, LEVEL2_DATASET_PATH):
        _copy_file(source_root / relative, target_root / relative)

    for level_dir in ("level_1", "level_2"):
        src_dir = source_root / level_dir / "videos"
        dst_dir = target_root / level_dir / "videos"
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)


def _load_level1_rows(source_root: Path) -> list[dict[str, Any]]:
    return json.loads((source_root / LEVEL1_DATASET_PATH).read_text(encoding="utf-8"))


def _load_level2_payload(source_root: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads((source_root / LEVEL2_DATASET_PATH).read_text(encoding="utf-8"))


def _materialize_mini_dataset(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)

    level1_rows = [
        row
        for row in _load_level1_rows(source_root)
        if int(row.get("id", -1)) in SOCIALOMNI_MINI_LEVEL1
    ]
    level1_rows.sort(key=lambda row: SOCIALOMNI_MINI_LEVEL1.index(int(row["id"])))
    level1_dataset_path = target_root / LEVEL1_DATASET_PATH
    level1_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    level1_dataset_path.write_text(
        json.dumps(level1_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    level1_videos_dir = target_root / LEVEL1_VIDEOS_DIR
    level1_videos_dir.mkdir(parents=True, exist_ok=True)
    for row in level1_rows:
        rel_name = Path(str(row["video_path"]).strip()).name
        _copy_file(
            source_root / LEVEL1_VIDEOS_DIR / rel_name, level1_videos_dir / rel_name
        )

    level2_payload = _load_level2_payload(source_root)
    level2_rows = (
        level2_payload.get("data", level2_payload)
        if isinstance(level2_payload, dict)
        else level2_payload
    )
    if not isinstance(level2_rows, list):
        raise ValueError("SocialOmni level2 annotations must contain a list of samples")
    filtered_level2_rows = [
        row
        for row in level2_rows
        if str(row.get("video_id", "")).strip() in SOCIALOMNI_MINI_LEVEL2
    ]
    filtered_level2_rows.sort(
        key=lambda row: SOCIALOMNI_MINI_LEVEL2.index(str(row["video_id"]).strip())
    )
    if isinstance(level2_payload, dict):
        mini_level2_payload = dict(level2_payload)
        mini_level2_payload["dataset_name"] = "socialomni-mini"
        mini_level2_payload["total_samples"] = len(filtered_level2_rows)
        mini_level2_payload["data"] = filtered_level2_rows
    else:
        mini_level2_payload = filtered_level2_rows
    level2_dataset_path = target_root / LEVEL2_DATASET_PATH
    level2_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    level2_dataset_path.write_text(
        json.dumps(mini_level2_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    level2_videos_dir = target_root / LEVEL2_VIDEOS_DIR
    level2_videos_dir.mkdir(parents=True, exist_ok=True)
    for row in filtered_level2_rows:
        rel_name = Path(str(row["video_file"]).strip()).name
        _copy_file(
            source_root / LEVEL2_VIDEOS_DIR / rel_name, level2_videos_dir / rel_name
        )

    write_socialomni_mini_manifest(target_root)


def prepare_socialomni_dataset(
    dataset_name: str,
    local_dir: str | None = None,
    *,
    quiet: bool = False,
    source_root: str | Path | None = None,
) -> Path:
    """Materialize the full or frozen mini SocialOmni dataset locally."""
    target_root = get_socialomni_local_dir(dataset_name, local_dir)
    is_mini = dataset_name == "socialomni-mini"
    if _dataset_ready(target_root, mini=is_mini):
        if not quiet:
            logger.info("SocialOmni dataset already exists at %s", target_root)
        return target_root

    if not quiet:
        logger.info("Preparing %s under %s", dataset_name, target_root)

    normalized_source = (
        _normalize_source_root(source_root) if source_root is not None else None
    )
    if normalized_source is None:
        with tempfile.TemporaryDirectory(prefix="socialomni_prepare_") as tmpdir:
            normalized_source = _download_socialomni_source(tmpdir)
            if is_mini:
                _materialize_mini_dataset(normalized_source, target_root)
            else:
                _materialize_full_dataset(normalized_source, target_root)
        return target_root

    if is_mini:
        _materialize_mini_dataset(normalized_source, target_root)
    else:
        _materialize_full_dataset(normalized_source, target_root)
    return target_root


def load_socialomni_level1_samples(
    root_dir: str | Path,
    max_samples: int | None = None,
) -> list[SocialOmniLevel1Sample]:
    """Load SocialOmni level1 samples from a prepared local dataset root."""
    dataset_path = get_level1_dataset_path(root_dir)
    videos_dir = get_level1_videos_dir(root_dir)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"SocialOmni level1 dataset not found: {dataset_path}. "
            "Run `python -m benchmarks.dataset.prepare --dataset socialomni`."
        )
    if not videos_dir.is_dir():
        raise FileNotFoundError(
            f"SocialOmni level1 videos directory missing: {videos_dir}"
        )

    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(
            f"Expected a list in {dataset_path}, got {type(rows).__name__}"
        )

    samples: list[SocialOmniLevel1Sample] = []
    for row in rows:
        rel_video_path = str(row.get("video_path", "")).strip()
        if not rel_video_path:
            continue
        resolved_video = videos_dir / rel_video_path
        if not resolved_video.is_file():
            raise FileNotFoundError(
                f"SocialOmni level1 video missing for sample {row.get('id')}: {resolved_video}"
            )
        samples.append(
            SocialOmniLevel1Sample(
                sample_id=str(row.get("id", "")),
                video_path=str(resolved_video),
                question=str(row.get("question", "")).strip(),
                options=[str(option).strip() for option in row.get("options", [])],
                correct_answer=str(row.get("correct_answer", "")).strip().upper(),
                asr_content=str(row.get("asr_content", "")).strip(),
                metadata=dict(row.get("metadata", {})),
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def load_socialomni_level2_samples(
    root_dir: str | Path,
    max_samples: int | None = None,
) -> list[SocialOmniLevel2Sample]:
    """Load SocialOmni level2 samples from a prepared local dataset root."""
    dataset_path = get_level2_dataset_path(root_dir)
    videos_dir = get_level2_videos_dir(root_dir)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"SocialOmni level2 dataset not found: {dataset_path}. "
            "Run `python -m benchmarks.dataset.prepare --dataset socialomni`."
        )
    if not videos_dir.is_dir():
        raise FileNotFoundError(
            f"SocialOmni level2 videos directory missing: {videos_dir}"
        )

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(
            f"Expected a list in {dataset_path}, got {type(rows).__name__}"
        )

    samples: list[SocialOmniLevel2Sample] = []
    for row in rows:
        rel_video_path = str(row.get("video_file", "")).strip()
        if not rel_video_path:
            continue
        resolved_video = videos_dir / rel_video_path
        if not resolved_video.is_file():
            raise FileNotFoundError(
                f"SocialOmni level2 video missing for sample {row.get('video_id')}: {resolved_video}"
            )
        q1 = row.get("question_1", {})
        q2 = row.get("question_2", {})
        samples.append(
            SocialOmniLevel2Sample(
                sample_id=str(row.get("video_id", "")).strip(),
                video_path=str(resolved_video),
                original_video_id=str(row.get("original_video_id", "")).strip(),
                source_dir=str(row.get("source_dir", "")).strip(),
                question_1=SocialOmniLevel2Question1(
                    question=str(q1.get("question", "")).strip(),
                    timestamp=str(q1.get("timestamp", "")).strip(),
                    option_a=str(q1.get("option_A", "YES")).strip(),
                    option_b=str(q1.get("option_B", "NO")).strip(),
                    correct_answer=str(q1.get("correct_answer", "")).strip().upper(),
                ),
                question_2=SocialOmniLevel2Question2(
                    question=str(q2.get("question", "")).strip(),
                    answer=str(q2.get("answer", "")).strip(),
                ),
                metadata=dict(row.get("metadata", {})),
                full_asr=str(row.get("full_asr", "")).strip(),
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


__all__ = [
    "DEFAULT_SOCIALOMNI_DIRS",
    "SOCIALOMNI_DATASET_NAMES",
    "SOCIALOMNI_MINI_LEVEL1",
    "SOCIALOMNI_MINI_LEVEL2",
    "SOCIALOMNI_MINI_MANIFEST",
    "SOCIALOMNI_REPO_ID",
    "SocialOmniLevel1Sample",
    "SocialOmniLevel2Question1",
    "SocialOmniLevel2Question2",
    "SocialOmniLevel2Sample",
    "get_socialomni_local_dir",
    "load_socialomni_level1_samples",
    "load_socialomni_level2_samples",
    "prepare_socialomni_dataset",
    "write_socialomni_mini_manifest",
]
