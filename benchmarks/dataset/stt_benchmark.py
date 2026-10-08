# SPDX-License-Identifier: Apache-2.0
"""
Loader for the pipecat-ai/stt-benchmark-data dataset.
"""

from __future__ import annotations

import atexit
import io
import logging
import shutil
import tempfile
from pathlib import Path

import soundfile as sf

from benchmarks.dataset.prepare import (
    STT_BENCHMARK_DATASET_ID,
    STT_BENCHMARK_DATASET_REVISION,
)
from benchmarks.dataset.seedtts import SampleInput

logger = logging.getLogger(__name__)

STT_BENCHMARK_SPLIT = "train"
STT_BENCHMARK_LANG = "en"

_REQUIRED_COLUMNS = {"sample_id", "audio", "transcription"}
COLUMN_ALIASES = {"id": "sample_id", "text": "transcription"}

_STAGED_CACHE: dict[
    tuple[str, str | None, str, str | None, int | None], list[SampleInput]
] = {}


def _staged_wav_path(staging_root: Path, sample_id: str, *, repo_id: str) -> Path:
    """Return <staging_root>/<sample_id>.wav after rejecting unsafe ids."""
    error_prefix = f"Invalid sample_id for {repo_id}: {sample_id!r}"
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"{error_prefix} (empty id)")
    if "/" in sample_id or "\\" in sample_id or sample_id in {".", ".."}:
        raise ValueError(f"{error_prefix} (path characters)")
    wav_path = (staging_root / f"{sample_id}.wav").resolve()
    if wav_path.parent != staging_root:
        raise ValueError(f"{error_prefix} (escapes staging directory)")
    return wav_path


def load_stt_benchmark_samples(
    repo_id: str = STT_BENCHMARK_DATASET_ID,
    max_samples: int | None = None,
    *,
    config_name: str | None = None,
    split: str = STT_BENCHMARK_SPLIT,
    revision: str | None = None,
) -> list[SampleInput]:
    """Load STT benchmark samples from a HuggingFace Parquet repo."""
    if revision is None and repo_id == STT_BENCHMARK_DATASET_ID:
        revision = STT_BENCHMARK_DATASET_REVISION

    full_cache_key = (repo_id, config_name, split, revision, None)
    if full_cache_key in _STAGED_CACHE:
        samples = _STAGED_CACHE[full_cache_key]
        return samples[:max_samples] if max_samples is not None else list(samples)

    cache_key = (repo_id, config_name, split, revision, max_samples)
    if cache_key in _STAGED_CACHE:
        return list(_STAGED_CACHE[cache_key])

    from datasets import Audio, load_dataset

    logger.info(
        f"Loading {repo_id} config={config_name or 'default'} "
        f"split={split} revision={revision or 'default'} from HuggingFace ..."
    )
    load_kwargs = {"revision": revision} if revision else {}
    if repo_id == "openslr/librispeech_asr" and config_name:
        # note (MayDomine): selecting files avoids downloading unused train splits.
        ds = load_dataset(
            repo_id,
            data_files={split: f"{config_name}/{split}/*.parquet"},
            split=split,
            verification_mode="no_checks",
            **load_kwargs,
        )
    else:
        if config_name:
            load_kwargs["name"] = config_name
        ds = load_dataset(repo_id, split=split, **load_kwargs)
    aliases = {
        old: new
        for old, new in COLUMN_ALIASES.items()
        if old in ds.column_names and new not in ds.column_names
    }
    if aliases:
        ds = ds.rename_columns(aliases)

    missing = _REQUIRED_COLUMNS - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset {repo_id} split={split} is missing columns: {missing}"
        )

    ds = ds.cast_column("audio", Audio(decode=False))
    if max_samples is not None:
        ds = ds.select(list(range(min(max_samples, len(ds)))))

    tmpdir = Path(tempfile.mkdtemp(prefix=f"stt_benchmark_{split}_"))
    atexit.register(shutil.rmtree, str(tmpdir), True)
    logger.info(f"Staging audio to {tmpdir}")
    staging_root = tmpdir.resolve()

    samples: list[SampleInput] = []
    seen_ids: set[str] = set()
    for row in ds:
        sample_id = row["sample_id"]
        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate sample_id for {repo_id}/{split}: {sample_id!r}"
            )
        seen_ids.add(sample_id)
        wav_path = _staged_wav_path(staging_root, sample_id, repo_id=repo_id)
        audio = row["audio"] or {}
        audio_bytes = audio.get("bytes")
        if not audio_bytes:
            audio_path = audio.get("path")
            if not audio_path:
                raise ValueError(f"Empty audio bytes for {repo_id}/{split}/{sample_id}")
            audio_bytes = Path(audio_path).read_bytes()

        if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            waveform, sample_rate = sf.read(io.BytesIO(audio_bytes))
            sf.write(str(wav_path), waveform, sample_rate, format="WAV")
        else:
            wav_path.write_bytes(audio_bytes)

        transcription = str(row["transcription"] or "").strip()
        samples.append(
            SampleInput(
                sample_id=sample_id,
                ref_text=transcription,
                ref_audio=str(wav_path),
                target_text=transcription,
            )
        )

    _STAGED_CACHE[cache_key] = samples
    logger.info(f"Loaded {len(samples)} samples from {repo_id}/{split}")
    return list(samples)
