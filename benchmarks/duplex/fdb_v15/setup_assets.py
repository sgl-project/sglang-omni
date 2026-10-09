# SPDX-License-Identifier: Apache-2.0
"""One-time setup: reference checkout, scoring venv, dataset and model checkpoints.
Safe to rerun; finished steps are skipped."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import zipfile

from huggingface_hub import hf_hub_download, model_info, snapshot_download

from benchmarks.duplex.fdb_v15.common import (
    FDB_SOURCE_REVISION,
    FDB_SOURCE_URL,
    JUDGE_MODEL_ID,
    MODEL_ID,
    PARAKEET_FILENAME,
    PARAKEET_REPO_ID,
    PARAKEET_REVISION,
    PARAKEET_SHA256,
    Settings,
    log,
)
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.v15_dataset import SUBSETS

REFERENCE_FILE_SHA256 = {
    "v1_v1.5/get_transcript/asr.py": "aedaee0d50f2bc47947caf6f3899939461e290225a98c0ddc434c360189596cf",
    "v1_v1.5/evaluation/get_timing.py": "4f551da4194ab4d9584db964f4b27223914eecf98cbc312388ecf45eaf6f8a17",
    "v1_v1.5/evaluation/eval_behavior.py": "0ff8179a437503581d65787da3a43924b45310c31a98d1bcbadce5c8605ca6f2",
    "v1_v1.5/evaluation/instruction/behavior.txt": "19e5477dac9a9a1e11de126783a0b820b3ecb70db5e91181824fa944e1947977",
}
SCORING_PYTHON_VERSION = "3.12"
SCORING_PACKAGES = [
    "torch==2.11.0",
    "torchaudio==2.11.0",
    "nemo_toolkit[asr]==3.0.0",
    "silero-vad==6.2.1",
    "openai",
    "pydantic",
    "scipy",
    "soundfile",
    "tqdm",
    "websockets",
    "gdown",
]
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
DATASET_DRIVE_FILE_IDS = {
    "user_interruption": "1wqYcYS4-30W2YMc3TfeMeoiaW9PLf4yT",
    "user_backchannel": "1EGwCd9CdGuh8jeqwPENzbBs5FSEgnYL3",
    "talking_to_other": "1Jh6ER4AUmqGgEZBTV0pcbDaMMQIWA7Kt",
    "background_speech": "1W63k1BlQ0QCgvYCb_8YJNqFfUhBwI97W",
}


def setup_reference_source(settings: Settings) -> None:
    log(f"== [1/6] Full-Duplex-Bench reference checkout at {FDB_SOURCE_REVISION}")
    if not (settings.fdb_source / ".git").is_dir():
        subprocess.run(
            ["git", "clone", FDB_SOURCE_URL, str(settings.fdb_source)], check=True
        )
    else:
        pass
    subprocess.run(
        [
            "git",
            "-C",
            str(settings.fdb_source),
            "checkout",
            "--quiet",
            FDB_SOURCE_REVISION,
        ],
        check=True,
    )
    for relative_path, expected in REFERENCE_FILE_SHA256.items():
        if file_sha256(settings.fdb_source / relative_path) != expected:
            raise SystemExit(
                f"ERROR: {relative_path} does not match its pinned SHA-256."
            )
        else:
            pass


def setup_scoring_venv(settings: Settings) -> None:
    log(f"== [2/6] Scoring venv at {settings.scoring_venv}")
    if not settings.scoring_python.is_file():
        subprocess.run(
            [
                "uv",
                "venv",
                "--python",
                SCORING_PYTHON_VERSION,
                str(settings.scoring_venv),
            ],
            check=True,
        )
    else:
        pass
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            str(settings.scoring_python),
            "--index-strategy",
            "unsafe-best-match",
            "--extra-index-url",
            TORCH_INDEX_URL,
            *SCORING_PACKAGES,
        ],
        check=True,
    )


def setup_dataset(settings: Settings) -> None:
    log(f"== [3/6] FDB v1.5 dataset at {settings.dataset}")
    zip_dir = settings.dataset_dir / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        zip_path = zip_dir / f"{subset}.zip"
        if not zip_path.is_file() or zip_path.stat().st_size == 0:
            subprocess.run(
                [
                    str(settings.scoring_venv / "bin" / "gdown"),
                    "--quiet",
                    DATASET_DRIVE_FILE_IDS[subset],
                    "-O",
                    str(zip_path),
                ],
                check=True,
            )
        else:
            pass
        if not (settings.dataset / subset).is_dir():
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(settings.dataset)
        else:
            pass
    shutil.rmtree(settings.dataset / "__MACOSX", ignore_errors=True)
    # Same text as `sha256sum ./*.zip`, so the revision matches earlier setups.
    checksums = "".join(
        f"{file_sha256(zip_path)}  ./{zip_path.name}\n"
        for zip_path in sorted(zip_dir.glob("*.zip"))
    )
    (settings.dataset_dir / "zips.sha256").write_text(checksums)
    revision = hashlib.sha256(checksums.encode()).hexdigest()
    settings.dataset_revision_file.write_text(f"sha256:{revision}\n")
    for subset in SUBSETS:
        sample_count = sum(
            path.is_dir() for path in (settings.dataset / subset).iterdir()
        )
        log(f"   {subset}: {sample_count} samples")


def setup_parakeet(settings: Settings) -> None:
    log("== [4/6] Parakeet ASR checkpoint")
    nemo = settings.parakeet_nemo
    if nemo.is_file() and file_sha256(nemo) == PARAKEET_SHA256:
        return
    else:
        pass
    hf_hub_download(
        PARAKEET_REPO_ID,
        PARAKEET_FILENAME,
        revision=PARAKEET_REVISION,
        local_dir=nemo.parent,
    )
    if file_sha256(nemo) != PARAKEET_SHA256:
        raise SystemExit(f"ERROR: {nemo} does not match its pinned SHA-256.")
    else:
        pass


def setup_models(settings: Settings) -> None:
    log(f"== [5/6] Model under test at {settings.model_path}")
    if not (settings.model_path / "config.json").is_file():
        snapshot_download(
            MODEL_ID, revision=settings.model_revision, local_dir=settings.model_path
        )
    else:
        pass
    log(f"== [6/6] Judge model at {settings.judge_model_path} (JUDGE={settings.judge})")
    if settings.judge == "qwen" and not settings.judge_revision_file.is_file():
        judge_revision = model_info(JUDGE_MODEL_ID).sha
        snapshot_download(
            JUDGE_MODEL_ID, revision=judge_revision, local_dir=settings.judge_model_path
        )
        settings.judge_revision_file.write_text(f"{judge_revision}\n")
    else:
        pass


def setup(settings: Settings) -> None:
    (settings.fdb_work / "models").mkdir(parents=True, exist_ok=True)
    setup_reference_source(settings)
    setup_scoring_venv(settings)
    setup_dataset(settings)
    setup_parakeet(settings)
    setup_models(settings)
    log("Setup complete.")
