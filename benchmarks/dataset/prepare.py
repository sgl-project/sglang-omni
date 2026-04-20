# SPDX-License-Identifier: Apache-2.0
"""Dataset download and materialization helpers.

Usage:
    # SeedTTS family (downloads into ./seedtts_testset by default)
    python -m benchmarks.dataset.prepare --dataset seedtts
    python -m benchmarks.dataset.prepare --dataset seedtts-mini
    python -m benchmarks.dataset.prepare --dataset seedtts-50

    # MMMU / MMSU (pre-warm the HuggingFace datasets cache)
    python -m benchmarks.dataset.prepare --dataset mmmu
    python -m benchmarks.dataset.prepare --dataset mmmu-ci-50
    python -m benchmarks.dataset.prepare --dataset mmsu

    # SocialOmni (full + deterministic mini subset)
    python -m benchmarks.dataset.prepare --dataset socialomni
    python -m benchmarks.dataset.prepare --dataset socialomni-mini
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from collections.abc import Callable
from functools import partial
from pathlib import Path

from benchmarks.dataset.socialomni import prepare_socialomni_dataset

logger = logging.getLogger(__name__)

DATASETS: dict[str, str] = {
    "seedtts": "zhaochenyang20/seed-tts-eval",
    "seedtts-mini": "zhaochenyang20/seed-tts-eval-mini",
    "seedtts-50": "xuesongye/seed-tts-eval-50",
    "mmmu": "MMMU/MMMU",
    "mmmu-ci-50": "zhaochenyang20/mmmu-ci-50",
    "mmsu": "ddwang2000/MMSU",
    "socialomni": "alexisty/SocialOmni",
    "socialomni-mini": "alexisty/SocialOmni",
}

_CLI_LOCAL_DIRS: dict[str, str] = {
    "seedtts": "seedtts_testset",
    "seedtts-mini": "seedtts_testset",
    "seedtts-50": "seedtts_testset",
    "socialomni": "socialomni",
    "socialomni-mini": "socialomni-mini",
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
                    "Dataset already exists at %s, skipping download.",
                    local_dir,
                )
            return

    if not quiet:
        where = local_dir if local_dir is not None else "HuggingFace cache"
        logger.info("Downloading %s to %s ...", repo_id, where)

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
        logger.info("Dataset %s ready.", repo_id)


DatasetHandler = Callable[[str, str | None, bool], None]


def _prepare_hf_dataset(
    dataset_name: str,
    local_dir: str | None,
    quiet: bool,
    *,
    repo_id: str,
) -> None:
    existence_marker = (
        _SEEDTTS_EXISTENCE_MARKER
        if dataset_name in {"seedtts", "seedtts-mini", "seedtts-50"}
        else None
    )
    download_dataset(
        repo_id,
        local_dir,
        existence_marker=existence_marker,
        quiet=quiet,
    )


def _prepare_socialomni(dataset_name: str, local_dir: str | None, quiet: bool) -> None:
    prepare_socialomni_dataset(dataset_name, local_dir, quiet=quiet)


_DATASET_HANDLERS: dict[str, DatasetHandler] = {
    "seedtts": partial(_prepare_hf_dataset, repo_id=DATASETS["seedtts"]),
    "seedtts-mini": partial(_prepare_hf_dataset, repo_id=DATASETS["seedtts-mini"]),
    "seedtts-50": partial(_prepare_hf_dataset, repo_id=DATASETS["seedtts-50"]),
    "mmmu": partial(_prepare_hf_dataset, repo_id=DATASETS["mmmu"]),
    "mmmu-ci-50": partial(_prepare_hf_dataset, repo_id=DATASETS["mmmu-ci-50"]),
    "mmsu": partial(_prepare_hf_dataset, repo_id=DATASETS["mmsu"]),
    "socialomni": _prepare_socialomni,
    "socialomni-mini": _prepare_socialomni,
}


def prepare_dataset(
    dataset_name: str,
    local_dir: str | None = None,
    *,
    quiet: bool = False,
) -> Path | None:
    """Prepare a dataset by name using the appropriate handler."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    handler = _DATASET_HANDLERS[dataset_name]
    handler(dataset_name, local_dir, quiet)
    return Path(local_dir) if local_dir is not None else None


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
        help="Override local directory for datasets that materialize locally. "
        "Ignored for datasets that are pulled only into the HuggingFace cache.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    local_dir = args.local_dir or _CLI_LOCAL_DIRS.get(args.dataset)
    prepare_dataset(args.dataset, local_dir)


if __name__ == "__main__":
    main()
