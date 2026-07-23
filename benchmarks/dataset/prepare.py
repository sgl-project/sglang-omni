# SPDX-License-Identifier: Apache-2.0
"""Dataset download helpers.

Usage:
    python -m benchmarks.dataset.prepare --dataset seedtts
    python -m benchmarks.dataset.prepare --dataset seedtts-mini
    python -m benchmarks.dataset.prepare --dataset seedtts-50
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

DATASETS: dict[str, str] = {
    "seedtts": "zhaochenyang20/seed-tts-eval-arrow",
    "seedtts-mini": "zhaochenyang20/seed-tts-eval-mini-arrow",
    "seedtts-50": "zhaochenyang20/seed-tts-eval-50-arrow",
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


def download_dataset(repo_id: str, *, quiet: bool = False) -> None:
    """Pre-warm the HuggingFace ``datasets`` cache for *repo_id*."""
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import hf_hub_download

    if not quiet:
        logger.info(f"Pre-warming HuggingFace cache for {repo_id} ...")

    if repo_id == "MMMU/MMMU":
        config_names = get_dataset_config_names(repo_id)
        for config_name in config_names:
            load_dataset(repo_id, config_name, split="validation")
    elif repo_id == "BoJack/MMAR":
        load_dataset(repo_id)
        hf_hub_download(repo_id, "mmar-audio.tar.gz", repo_type="dataset")
    elif repo_id.startswith("lmms-lab/mmau:"):
        dataset_id, split = repo_id.split(":", 1)
        load_dataset(
            dataset_id,
            split=split,
            data_files={split: f"data/{split}-*.parquet"},
            verification_mode="no_checks",
        )
    else:
        load_dataset(repo_id)

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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    download_dataset(DATASETS[args.dataset])


if __name__ == "__main__":
    main()
