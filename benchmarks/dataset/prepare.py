# SPDX-License-Identifier: Apache-2.0
"""Dataset download helpers.

Usage:
    # SeedTTS family (downloads into ./seedtts_testset by default)
    python -m benchmarks.dataset.prepare --dataset seedtts
    python -m benchmarks.dataset.prepare --dataset seedtts-mini
    python -m benchmarks.dataset.prepare --dataset seedtts-50

    # MMMU / MMSU / Video-MME / Video-AMME (pre-warm the HuggingFace datasets cache)
    python -m benchmarks.dataset.prepare --dataset mmmu
    python -m benchmarks.dataset.prepare --dataset mmmu-ci-50
    python -m benchmarks.dataset.prepare --dataset mmsu
    python -m benchmarks.dataset.prepare --dataset videomme
    python -m benchmarks.dataset.prepare --dataset videomme-ci-50
    python -m benchmarks.dataset.prepare --dataset videomme-ci-25
    python -m benchmarks.dataset.prepare --dataset videoamme-ci-50
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

DATASETS: dict[str, str] = {
    "seedtts": "zhaochenyang20/seed-tts-eval",
    "seedtts-mini": "zhaochenyang20/seed-tts-eval-mini",
    "seedtts-50": "zhaochenyang20/seed-tts-eval-50",
    "mmmu": "MMMU/MMMU",
    "mmmu-ci-50": "zhaochenyang20/mmmu-ci-50",
    "mmsu": "ddwang2000/MMSU",
    "mmsu-ci-2000": "zhaochenyang20/mmsu-ci-2000",
    "videomme": "zhaochenyang20/Video_MME",
    "videomme-ci-50": "zhaochenyang20/Video_MME_ci",
    "videomme-ci-25": "zhaochenyang20/Video_MME_ci_25",
    "videoamme-ci-50": "zhaochenyang20/Video_AMME_ci",
}

_CLI_LOCAL_DIRS: dict[str, str] = {
    "seedtts": "seedtts_testset",
    "seedtts-mini": "seedtts_testset",
    "seedtts-50": "seedtts_testset",
}

_SEEDTTS_EXISTENCE_MARKER = "en/meta.lst"


def download_dataset(
    repo_id: str,
    local_dir: str | None = "seedtts_testset",
    *,
    existence_marker: str | None = _SEEDTTS_EXISTENCE_MARKER,
    quiet: bool = False,
) -> None:
    """Download a HuggingFace dataset."""
    if local_dir is not None and existence_marker:
        marker_path = os.path.join(local_dir, existence_marker)
        if os.path.exists(marker_path):
            if not quiet:
                logger.info(
                    f"Dataset already exists at {local_dir}, skipping download."
                )
            return

    if not quiet:
        where = local_dir if local_dir is not None else "HuggingFace cache"
        logger.info(f"Downloading {repo_id} to {where} ...")

    cmd = [
        "huggingface-cli",
        "download",
        repo_id,
        "--repo-type",
        "dataset",
    ]
    if local_dir is not None:
        cmd += ["--local-dir", local_dir]

    try:
        subprocess.run(cmd, check=True, capture_output=quiet, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to download dataset {repo_id}.\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    if not quiet:
        logger.info(f"Dataset {repo_id} ready.")


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
        default=None,
        help="Override local directory for seedtts-family datasets. "
        "Ignored for datasets that are pulled into the HuggingFace cache.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    repo_id = DATASETS[args.dataset]
    default_local_dir = _CLI_LOCAL_DIRS.get(args.dataset)
    local_dir = args.local_dir or default_local_dir
    existence_marker = (
        _SEEDTTS_EXISTENCE_MARKER if args.dataset in _CLI_LOCAL_DIRS else None
    )
    download_dataset(
        repo_id,
        local_dir,
        existence_marker=existence_marker,
    )


if __name__ == "__main__":
    main()
