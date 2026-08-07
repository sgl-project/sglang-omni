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

SEEDTTS_DATASET_ID = "zhaochenyang20/seed-tts-eval-arrow"
SEEDTTS_DATASET_REVISION = "27f4c1adee83b5b29b7c4b375f6b976324bda308"

DATASETS: dict[str, str] = {
    "seedtts": SEEDTTS_DATASET_ID,
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


def download_dataset(
    repo_id: str,
    *,
    revision: str | None = None,
    quiet: bool = False,
) -> None:
    """Pre-warm the HuggingFace ``datasets`` cache for *repo_id*."""
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import hf_hub_download

    if revision is None and repo_id == SEEDTTS_DATASET_ID:
        revision = SEEDTTS_DATASET_REVISION
    revision_kwargs = {"revision": revision} if revision else {}
    if not quiet:
        logger.info(
            "Pre-warming HuggingFace cache for %s revision=%s ...",
            repo_id,
            revision or "default",
        )

    if repo_id == "MMMU/MMMU":
        config_names = get_dataset_config_names(repo_id, **revision_kwargs)
        for config_name in config_names:
            load_dataset(
                repo_id,
                config_name,
                split="validation",
                **revision_kwargs,
            )
    elif repo_id == "BoJack/MMAR":
        load_dataset(repo_id, **revision_kwargs)
        hf_hub_download(
            repo_id,
            "mmar-audio.tar.gz",
            repo_type="dataset",
            **revision_kwargs,
        )
    elif repo_id.startswith("lmms-lab/mmau:"):
        dataset_id, split = repo_id.split(":", 1)
        load_dataset(
            dataset_id,
            split=split,
            data_files={split: f"data/{split}-*.parquet"},
            verification_mode="no_checks",
            **revision_kwargs,
        )
    else:
        load_dataset(repo_id, **revision_kwargs)

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
