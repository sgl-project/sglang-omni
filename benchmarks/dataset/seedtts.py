# SPDX-License-Identifier: Apache-2.0
"""SeedTTS dataset loader for local meta.lst and HuggingFace Arrow/Parquet repos.

Audio bytes are staged to a process-scoped temporary directory so downstream
consumers (which expect filesystem paths) work unchanged for Arrow/Parquet
sources. Local meta.lst files already point at local audio paths.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {
    "sample_id",
    "ref_text",
    "ref_audio_path",
    "target_text",
    "ref_audio",
}


@dataclass
class SampleInput:
    sample_id: str
    ref_text: str
    ref_audio: str
    target_text: str


_STAGED_CACHE: dict[tuple[str, str, int | None], list[SampleInput]] = {}


def _resolve_staged_audio_path(
    staging_root: Path,
    rel_path: str,
    *,
    repo_id: str,
    split: str,
    sample_id: str,
) -> Path:
    raw_path = rel_path.strip() if isinstance(rel_path, str) else ""
    error_prefix = (
        f"Invalid ref_audio_path for {repo_id}/{split}/{sample_id}: {rel_path!r}"
    )
    if not raw_path:
        raise ValueError(f"{error_prefix} (empty path)")

    posix_path = PurePosixPath(raw_path)
    windows_path = PureWindowsPath(raw_path)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
    ):
        raise ValueError(f"{error_prefix} (absolute or anchored path)")

    if ".." in posix_path.parts or ".." in windows_path.parts:
        raise ValueError(f"{error_prefix} (parent traversal)")

    staging_root = staging_root.resolve()
    wav_path = (staging_root / Path(raw_path)).resolve()
    try:
        wav_path.relative_to(staging_root)
    except ValueError as exc:
        raise ValueError(f"{error_prefix} (path escapes staging directory)") from exc
    return wav_path


def load_seedtts_samples(
    source: str,
    max_samples: int | None = None,
    *,
    split: str = "en",
) -> list[SampleInput]:
    """Load SeedTTS evaluation samples.

    *source* may be either a local ``meta.lst`` file in seed-tts-eval format or
    a HuggingFace dataset repo id (e.g.
    ``zhaochenyang20/seed-tts-eval-50-arrow``). Arrow/Parquet datasets are
    fetched via ``datasets.load_dataset`` and audio bytes are staged to a
    temporary directory.
    """
    if os.path.isfile(source) or source.endswith(".lst"):
        return _load_from_meta_lst(source, max_samples)
    return _load_from_arrow(source, split, max_samples)


def _load_from_meta_lst(path: str, max_samples: int | None) -> list[SampleInput]:
    """Parse a local seed-tts-eval meta.lst file."""
    if max_samples is not None and max_samples <= 0:
        return []

    base_dir = os.path.dirname(path)
    samples: list[SampleInput] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            samples.append(
                SampleInput(
                    sample_id=parts[0],
                    ref_text=parts[1],
                    ref_audio=os.path.join(base_dir, parts[2]),
                    target_text=parts[3],
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def _load_from_arrow(
    repo_id: str, split: str, max_samples: int | None
) -> list[SampleInput]:
    """Load from a HuggingFace Arrow/Parquet dataset repo."""
    full_cache_key = (repo_id, split, None)
    if full_cache_key in _STAGED_CACHE:
        samples = _STAGED_CACHE[full_cache_key]
        return samples[:max_samples] if max_samples is not None else list(samples)

    cache_key = (repo_id, split, max_samples)
    if cache_key in _STAGED_CACHE:
        return list(_STAGED_CACHE[cache_key])

    from datasets import Audio, load_dataset

    logger.info("Loading %s split=%s from HuggingFace ...", repo_id, split)
    ds = load_dataset(repo_id, split=split)

    missing = _REQUIRED_COLUMNS - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset {repo_id} split={split} is missing columns: {missing}"
        )

    ds = ds.cast_column("ref_audio", Audio(decode=False))
    if max_samples is not None:
        ds = ds.select(list(range(min(max_samples, len(ds)))))

    tmpdir = Path(tempfile.mkdtemp(prefix=f"seedtts_{split}_"))
    atexit.register(shutil.rmtree, str(tmpdir), True)
    logger.info("Staging audio to %s", tmpdir)

    samples: list[SampleInput] = []
    written: set[str] = set()
    staging_root = tmpdir.resolve()

    for row in ds:
        rel = row["ref_audio_path"]
        wav_path = _resolve_staged_audio_path(
            staging_root,
            rel,
            repo_id=repo_id,
            split=split,
            sample_id=row["sample_id"],
        )
        audio = row["ref_audio"] or {}
        audio_bytes = audio.get("bytes")
        if not audio_bytes:
            raise ValueError(
                f"Empty audio bytes for {repo_id}/{split}/{row['sample_id']}"
            )

        rel_key = wav_path.relative_to(staging_root).as_posix()
        if rel_key not in written:
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path.write_bytes(audio_bytes)
            written.add(rel_key)

        samples.append(
            SampleInput(
                sample_id=row["sample_id"],
                ref_text=row["ref_text"],
                ref_audio=str(wav_path),
                target_text=row["target_text"],
            )
        )

    _STAGED_CACHE[cache_key] = samples
    logger.info(
        "Loaded %d samples (%d unique audio files) from %s/%s",
        len(samples),
        len(written),
        repo_id,
        split,
    )
    return list(samples)
