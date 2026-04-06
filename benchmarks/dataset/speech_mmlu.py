# SPDX-License-Identifier: Apache-2.0
"""Dataset loader for XiaomiMiMo/SpeechMMLU.

Downloads and caches the SpeechMMLU dataset from HuggingFace, extracting
audio files to disk so they can be passed to the sglang-omni API via file paths.

Usage::

    from benchmarks.dataset.speech_mmlu import load_speech_mmlu_samples

    samples = load_speech_mmlu_samples(max_samples=100)
    samples = load_speech_mmlu_samples(subjects=["anatomy", "virology"])
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

DATASET_REPO = "XiaomiMiMo/SpeechMMLU"
ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


@dataclass
class SpeechMmluSample:
    sample_id: str
    audio_path: str
    question_text: str
    correct_answer: int  # 0-3 mapping to A-D
    subject: str


def _extract_and_cache(cache_dir: str) -> list[dict]:
    """Download the dataset and cache audio files + metadata to disk."""
    from datasets import load_dataset

    logger.info("Downloading SpeechMMLU dataset from %s ...", DATASET_REPO)
    ds = load_dataset(DATASET_REPO, split="train")

    audio_dir = os.path.join(cache_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    metadata = []
    for idx, row in enumerate(ds):
        sample_id = row["id"]
        subject = row["subject"]

        # Write audio to disk
        subject_dir = os.path.join(audio_dir, subject)
        os.makedirs(subject_dir, exist_ok=True)
        audio_path = os.path.join(subject_dir, f"{sample_id}.mp3")

        if not os.path.exists(audio_path):
            audio_obj = row["question_audio"]
            if isinstance(audio_obj, dict) and "path" in audio_obj:
                # HF Audio feature: {"path": ..., "array": ..., "sampling_rate": ...}
                import soundfile as sf

                sf.write(
                    audio_path.replace(".mp3", ".wav"),
                    audio_obj["array"],
                    audio_obj["sampling_rate"],
                )
                audio_path = audio_path.replace(".mp3", ".wav")
            elif isinstance(audio_obj, dict) and "bytes" in audio_obj:
                with open(audio_path, "wb") as f:
                    f.write(audio_obj["bytes"])
            elif isinstance(audio_obj, str) and os.path.isfile(audio_obj):
                import shutil

                shutil.copy2(audio_obj, audio_path)
            else:
                logger.warning(
                    "Unexpected audio format for sample %s: %s", sample_id, type(audio_obj)
                )
                continue

        metadata.append(
            {
                "sample_id": sample_id,
                "audio_path": os.path.abspath(audio_path),
                "question_text": row["question_text"],
                "correct_answer": int(row["answer"]),
                "subject": subject,
            }
        )

        if (idx + 1) % 1000 == 0:
            logger.info("  cached %d / %d samples", idx + 1, len(ds))

    meta_path = os.path.join(cache_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    logger.info("Cached %d samples to %s", len(metadata), cache_dir)
    return metadata


def load_speech_mmlu_samples(
    cache_dir: str = "benchmarks/cache/speech_mmlu",
    max_samples: int | None = None,
    subjects: list[str] | None = None,
    seed: int | None = None,
) -> list[SpeechMmluSample]:
    """Load SpeechMMLU samples, downloading and caching on first call.

    Args:
        cache_dir: Directory for cached audio files and metadata.
        max_samples: Maximum number of samples to return.
        subjects: Optional list of subjects to filter by.
        seed: Random seed for reproducible subsampling.

    Returns:
        List of SpeechMmluSample.
    """
    meta_path = os.path.join(cache_dir, "metadata.json")

    if os.path.exists(meta_path):
        logger.info("Loading cached metadata from %s", meta_path)
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        os.makedirs(cache_dir, exist_ok=True)
        metadata = _extract_and_cache(cache_dir)

    # Filter by subjects
    if subjects:
        subject_set = set(subjects)
        metadata = [m for m in metadata if m["subject"] in subject_set]

    # Subsample
    if seed is not None:
        random.seed(seed)
        random.shuffle(metadata)
    if max_samples is not None and len(metadata) > max_samples:
        metadata = metadata[:max_samples]

    samples = [SpeechMmluSample(**m) for m in metadata]
    logger.info(
        "Loaded %d SpeechMMLU samples (%d subjects)",
        len(samples),
        len({s.subject for s in samples}),
    )
    return samples
