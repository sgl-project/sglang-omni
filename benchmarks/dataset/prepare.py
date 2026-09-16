# SPDX-License-Identifier: Apache-2.0
"""Dataset download helpers.

Usage:
    python -m benchmarks.dataset.prepare --dataset seedtts
    python -m benchmarks.dataset.prepare --dataset seedtts-mini
    python -m benchmarks.dataset.prepare --dataset seedtts-50
    python -m benchmarks.dataset.prepare --dataset stt-benchmark
    python -m benchmarks.dataset.prepare --dataset longlibriheavy-30
    python -m benchmarks.dataset.prepare --dataset longlibriheavy-60
    python -m benchmarks.dataset.prepare --dataset meanwhile
    python -m benchmarks.dataset.prepare --dataset mmmu
    python -m benchmarks.dataset.prepare --dataset mmmu-ci-50
    python -m benchmarks.dataset.prepare --dataset mmsu
    python -m benchmarks.dataset.prepare --dataset mmau-mini
    python -m benchmarks.dataset.prepare --dataset mmar
    python -m benchmarks.dataset.prepare --dataset videomme
    python -m benchmarks.dataset.prepare --dataset videomme-ci-50
    python -m benchmarks.dataset.prepare --dataset videomme-ci-25
    python -m benchmarks.dataset.prepare --dataset videoamme-ci-50
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

SEEDTTS_DATASET_ID = "zhaochenyang20/seed-tts-eval-arrow"
SEEDTTS_DATASET_REVISION = "27f4c1adee83b5b29b7c4b375f6b976324bda308"
STT_BENCHMARK_DATASET_ID = "pipecat-ai/stt-benchmark-data"
STT_BENCHMARK_DATASET_REVISION = "3fe50170d520c951957b86996ef082a6ab87b394"
LONGLIBRIHEAVY_DATASET_ID = "inesc-id/longlibriheavy"
LONGLIBRIHEAVY_DATASET_REVISION = "09bc067255eeb0d0bca62357ac985c2ebdc5169c"
MEANWHILE_DATASET_ID = "distil-whisper/meanwhile"
MEANWHILE_DATASET_REVISION = "5a6b431a268523a6603f199d859fc25a24c22900"

DATASETS: dict[str, str] = {
    "seedtts": SEEDTTS_DATASET_ID,
    "seedtts-mini": "zhaochenyang20/seed-tts-eval-mini-arrow",
    "seedtts-50": "zhaochenyang20/seed-tts-eval-50-arrow",
    "stt-benchmark": STT_BENCHMARK_DATASET_ID,
    "longlibriheavy-30": f"{LONGLIBRIHEAVY_DATASET_ID}:llh_test_30",
    "longlibriheavy-60": f"{LONGLIBRIHEAVY_DATASET_ID}:llh_test_60",
    "meanwhile": f"{MEANWHILE_DATASET_ID}:test",
    "mmmu": "MMMU/MMMU",
    "mmmu-ci-50": "zhaochenyang20/mmmu-ci-50",
    "mmsu": "ddwang2000/MMSU",
    "mmsu-ci-2000": "zhaochenyang20/mmsu-ci-2000",
    "mmau": "lmms-lab/mmau",
    "mmau-mini": "lmms-lab/mmau:test_mini",
    "mmar": "BoJack/MMAR",
    "videomme": "zhaochenyang20/Video_MME",
    "videomme-ci-50": "zhaochenyang20/Video_MME_ci",
    "videomme-ci-25": "zhaochenyang20/Video_MME_ci_25",
    "videoamme-ci-50": "zhaochenyang20/Video_AMME_ci",
}


def download_dataset(
    repo_id: str,
    *,
    revision: str | None = None,
    quiet: bool = False,
) -> None:
    """Pre-warm the HuggingFace ``datasets`` cache for *repo_id*."""
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import hf_hub_download

    dataset_id, separator, split = repo_id.partition(":")
    if revision is None and dataset_id == SEEDTTS_DATASET_ID:
        revision = SEEDTTS_DATASET_REVISION
    elif revision is None and dataset_id == STT_BENCHMARK_DATASET_ID:
        revision = STT_BENCHMARK_DATASET_REVISION
    elif revision is None and dataset_id == LONGLIBRIHEAVY_DATASET_ID:
        revision = LONGLIBRIHEAVY_DATASET_REVISION
    elif revision is None and dataset_id == MEANWHILE_DATASET_ID:
        revision = MEANWHILE_DATASET_REVISION
    revision_kwargs = {"revision": revision} if revision else {}
    if not quiet:
        logger.info(
            "Pre-warming HuggingFace cache for %s split=%s revision=%s ...",
            dataset_id,
            split if separator else "all",
            revision or "default",
        )

    if dataset_id == "MMMU/MMMU":
        config_names = get_dataset_config_names(dataset_id, **revision_kwargs)
        for config_name in config_names:
            load_dataset(
                dataset_id,
                config_name,
                split="validation",
                **revision_kwargs,
            )
    elif dataset_id == "BoJack/MMAR":
        load_dataset(dataset_id, **revision_kwargs)
        hf_hub_download(
            dataset_id,
            "mmar-audio.tar.gz",
            repo_type="dataset",
            **revision_kwargs,
        )
    elif dataset_id == "lmms-lab/mmau" and separator:
        load_dataset(
            dataset_id,
            split=split,
            data_files={split: f"data/{split}-*.parquet"},
            verification_mode="no_checks",
            **revision_kwargs,
        )
    elif separator:
        load_dataset(dataset_id, split=split, **revision_kwargs)
    else:
        load_dataset(dataset_id, **revision_kwargs)

    if not quiet:
        logger.info(f"Dataset {repo_id} cached.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="seedtts",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Dataset revision; known evaluation datasets use a pinned default.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    download_dataset(DATASETS[args.dataset], revision=args.revision)


if __name__ == "__main__":
    main()
