# SPDX-License-Identifier: Apache-2.0
"""Dataset download helpers.

Usage:
    python -m benchmarks.dataset.prepare --dataset seedtts
    python -m benchmarks.dataset.prepare --dataset seedtts-mini
    python -m benchmarks.dataset.prepare --dataset seedtts-50
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

DATASETS = {
    "seedtts": "zhaochenyang20/seed-tts-eval",
    "seedtts-mini": "zhaochenyang20/seed-tts-eval-mini",
    "seedtts-50": "xuesongye/seed-tts-eval-50",
    "mmmu-ci-50": "zhaochenyang20/mmmu-ci-50",
}


def download_dataset(
    repo_id: str,
    local_dir: str = "seedtts_testset",
    *,
    quiet: bool = False,
) -> None:
    meta_en = os.path.join(local_dir, "en", "meta.lst")
    if os.path.exists(meta_en):
        if not quiet:
            logger.info("Dataset already exists at %s, skipping download.", local_dir)
        return

    if not quiet:
        logger.info("Downloading %s to %s ...", repo_id, local_dir)
    cmd = [
        "huggingface-cli",
        "download",
        repo_id,
        "--repo-type",
        "dataset",
        "--local-dir",
        local_dir,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=quiet, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to download dataset {repo_id} to {local_dir}.\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    if not quiet:
        logger.info("Dataset downloaded to %s", local_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="seedtts",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--local-dir",
        default="seedtts_testset",
        help="Local directory for the dataset.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    download_dataset(DATASETS[args.dataset], args.local_dir)


if __name__ == "__main__":
    main()
