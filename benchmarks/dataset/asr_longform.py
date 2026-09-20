# SPDX-License-Identifier: Apache-2.0
"""Load the canonical long-form English ASR evaluation datasets."""

from __future__ import annotations

import atexit
import logging
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

from benchmarks.dataset.prepare import (
    LONGLIBRIHEAVY_DATASET_ID,
    LONGLIBRIHEAVY_DATASET_REVISION,
    MEANWHILE_DATASET_ID,
    MEANWHILE_DATASET_REVISION,
)
from benchmarks.dataset.seedtts import SampleInput
from sglang_omni.utils.audio import load_audio

logger = logging.getLogger(__name__)

ASR_LONGFORM_LANG = "en"
ASR_LONGFORM_SAMPLE_RATE = 16000


@dataclass(frozen=True, slots=True)
class LongFormDatasetConfig:
    name: str
    repo_id: str
    revision: str
    split: str
    expected_sample_count: int


ASR_LONGFORM_DATASETS: dict[str, LongFormDatasetConfig] = {
    "longlibriheavy-30": LongFormDatasetConfig(
        name="longlibriheavy-30",
        repo_id=LONGLIBRIHEAVY_DATASET_ID,
        revision=LONGLIBRIHEAVY_DATASET_REVISION,
        split="llh_test_30",
        expected_sample_count=1203,
    ),
    "longlibriheavy-60": LongFormDatasetConfig(
        name="longlibriheavy-60",
        repo_id=LONGLIBRIHEAVY_DATASET_ID,
        revision=LONGLIBRIHEAVY_DATASET_REVISION,
        split="llh_test_60",
        expected_sample_count=591,
    ),
    "meanwhile": LongFormDatasetConfig(
        name="meanwhile",
        repo_id=MEANWHILE_DATASET_ID,
        revision=MEANWHILE_DATASET_REVISION,
        split="test",
        expected_sample_count=64,
    ),
}

_REQUIRED_COLUMNS = {"audio", "text"}
_STAGED_CACHE: dict[tuple[str, str, int | None], list[SampleInput]] = {}


def _audio_source(
    audio: object, *, repo_id: str, split: str, sample_id: str
) -> bytes | str:
    if not isinstance(audio, Mapping):
        raise ValueError(f"Missing audio for {repo_id}/{split}/{sample_id}")
    audio_bytes = audio.get("bytes")
    if isinstance(audio_bytes, (bytes, bytearray, memoryview)) and audio_bytes:
        return bytes(audio_bytes)
    audio_path = audio.get("path")
    if isinstance(audio_path, str) and audio_path:
        return audio_path
    raise ValueError(f"Missing audio bytes/path for {repo_id}/{split}/{sample_id}")


def load_asr_longform_samples(
    dataset_name: str,
    max_samples: int | None = None,
    *,
    revision: str | None = None,
) -> list[SampleInput]:
    """Load one registered dataset and stage it as mono 16 kHz PCM WAV."""
    try:
        config = ASR_LONGFORM_DATASETS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown long-form ASR dataset: {dataset_name!r}") from exc
    dataset_revision = revision or config.revision

    full_cache_key = (dataset_name, dataset_revision, None)
    if full_cache_key in _STAGED_CACHE:
        samples = _STAGED_CACHE[full_cache_key]
        return samples[:max_samples] if max_samples is not None else list(samples)

    cache_key = (dataset_name, dataset_revision, max_samples)
    if cache_key in _STAGED_CACHE:
        return list(_STAGED_CACHE[cache_key])

    logger.info(
        "Loading %s split=%s revision=%s from HuggingFace ...",
        config.repo_id,
        config.split,
        dataset_revision,
    )
    dataset = load_dataset(
        config.repo_id,
        split=config.split,
        revision=dataset_revision,
    )
    missing = _REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset {config.repo_id} split={config.split} is missing columns: "
            f"{missing}"
        )
    if (
        max_samples is None
        and dataset_revision == config.revision
        and len(dataset) != config.expected_sample_count
    ):
        raise ValueError(
            f"Expected {config.expected_sample_count} samples for "
            f"{config.repo_id}/{config.split}, got {len(dataset)}"
        )

    dataset = dataset.cast_column("audio", Audio(decode=False))
    if max_samples is not None:
        dataset = dataset.select(list(range(min(max_samples, len(dataset)))))

    staging_dir = Path(tempfile.mkdtemp(prefix=f"asr_{dataset_name}_"))
    atexit.register(shutil.rmtree, str(staging_dir), True)
    logger.info("Staging normalized audio to %s", staging_dir)

    samples: list[SampleInput] = []
    for index, row in enumerate(dataset):
        sample_id = f"{dataset_name}-{index:06d}"
        reference = row["text"]
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(
                f"Empty text for {config.repo_id}/{config.split}/{sample_id}"
            )
        source = _audio_source(
            row["audio"],
            repo_id=config.repo_id,
            split=config.split,
            sample_id=sample_id,
        )
        waveform = load_audio(
            source,
            source_name=f"{config.repo_id}/{config.split}/{sample_id}",
            target_sample_rate=ASR_LONGFORM_SAMPLE_RATE,
            mono=True,
        )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(
                f"Invalid decoded audio for {config.repo_id}/{config.split}/{sample_id}"
            )
        wav_path = staging_dir / f"{sample_id}.wav"
        sf.write(
            wav_path,
            waveform,
            ASR_LONGFORM_SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )
        text = reference.strip()
        samples.append(
            SampleInput(
                sample_id=sample_id,
                ref_text=text,
                ref_audio=str(wav_path),
                target_text=text,
            )
        )

    _STAGED_CACHE[cache_key] = samples
    logger.info(
        "Loaded %d samples from %s/%s",
        len(samples),
        config.repo_id,
        config.split,
    )
    return list(samples)
