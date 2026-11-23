# SPDX-License-Identifier: Apache-2.0
"""One-time setup: reference checkout, scoring venv, the v1.5 and v1.0 datasets
and model checkpoints. Safe to rerun; finished steps are skipped."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import zipfile
from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path

from huggingface_hub import hf_hub_download, model_info, snapshot_download
from huggingface_hub.errors import GatedRepoError
from packaging.requirements import Requirement

from benchmarks.duplex.fdb.common import (
    FDB_SOURCE_REVISION,
    FDB_SOURCE_URL,
    JUDGE_MODEL_ID,
    PARAKEET_FILENAME,
    PARAKEET_REPO_ID,
    PARAKEET_REVISION,
    PARAKEET_SHA256,
    Settings,
    log,
)
from benchmarks.duplex.reference_core import V10_REFERENCE_FILES
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.v10_dataset import SUBSETS as V10_SUBSETS
from benchmarks.duplex.v15_dataset import SUBSETS
from benchmarks.metrics.speaker_similarity_assets import (
    ensure_speaker_similarity_assets,
)
from benchmarks.metrics.utmos import ensure_utmos_assets

REFERENCE_FILE_SHA256 = {
    "v1_v1.5/get_transcript/asr.py": "aedaee0d50f2bc47947caf6f3899939461e290225a98c0ddc434c360189596cf",
    "v1_v1.5/evaluation/get_timing.py": "4f551da4194ab4d9584db964f4b27223914eecf98cbc312388ecf45eaf6f8a17",
    "v1_v1.5/evaluation/eval_behavior.py": "0ff8179a437503581d65787da3a43924b45310c31a98d1bcbadce5c8605ca6f2",
    "v1_v1.5/evaluation/instruction/behavior.txt": "19e5477dac9a9a1e11de126783a0b820b3ecb70db5e91181824fa944e1947977",
    **dict(V10_REFERENCE_FILES.values()),
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
    # note (luojiaxuan): the pinned v1.0 evaluate.py imports dotenv, and its
    # backchannel task reads WAV through torchaudio, which needs torchcodec.
    "python-dotenv",
    "torchcodec==0.11.*",
]
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
# note (zhaochenyang20): byte-identical copies of the official Google Drive
# archives, which refuse downloads once a file has had too many accesses.
DATASET_ARCHIVE_REPO_ID = "zhaochenyang20/full-duplex-bench"
DATASET_ARCHIVE_REVISION = "800e9b2d5493880fdd9e3396e9586344ed707ea8"
# note (zhaochenyang20): the CANDOR license forbids redistribution, so these clips
# come only from this gated repository; the HF_TOKEN account must accept it.
CANDOR_REPO_ID = "betterup/candor-full-duplex-bench"
CANDOR_REVISION = "e6ad443be50c779edf1900af22bc39687733719a"
CANDOR_SUBSETS = ("candor_pause_handling", "candor_turn_taking")
# note (jeffro): qwen-asr pins a transformers release of its own, so the
# cascade's forced-alignment worker runs from a venv apart from the server's.
# this need double check.
ALIGNER_PACKAGES = [
    "qwen-asr==0.0.6",
    "fastapi==0.156.4",
    "uvicorn==0.54.0",
    "httpx==0.28.1",
    "msgpack==1.2.3",
    "msgspec==0.22.0",
    "numpy==2.5.4",
    "pydantic==2.14.1",
    "regex==2026.9.29",
    "websockets==17.3",
]
SERVING_DISTRIBUTION = "sglang-omni"
MODEL_EXTRA = "minicpm-o"


def setup_model_extra() -> None:
    log(f"== [1/9] {MODEL_EXTRA} extra in the serving venv ({sys.executable})")
    try:
        declared = requires(SERVING_DISTRIBUTION) or []
    except PackageNotFoundError:
        raise SystemExit(
            f"ERROR: {SERVING_DISTRIBUTION} is not installed for {sys.executable}; "
            "run `uv pip install -e .` from the repository root first."
        ) from None
    extra_requirements = []
    for text in declared:
        requirement = Requirement(text)
        marker = requirement.marker
        # Note (jeffro): base requirements with platform markers also evaluate true under the extra.
        if (
            marker is not None
            and marker.evaluate({"extra": MODEL_EXTRA})
            and not marker.evaluate({"extra": ""})
        ):
            extra_requirements.append(requirement)
        else:
            pass
    if not extra_requirements:
        raise SystemExit(
            f"ERROR: the installed {SERVING_DISTRIBUTION} metadata has no "
            f"{MODEL_EXTRA} extra; it predates this checkout. Rerun "
            "`uv pip install -e .` from the repository root first."
        )
    else:
        pass
    missing_requirements = []
    for requirement in extra_requirements:
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            missing_requirements.append(requirement)
        else:
            if installed not in requirement.specifier:
                missing_requirements.append(requirement)
            else:
                pass
    if not missing_requirements:
        return
    else:
        pass
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            sys.executable,
            *(
                f"{requirement.name}{requirement.specifier}"
                for requirement in missing_requirements
            ),
        ],
        check=True,
    )


def setup_reference_source(settings: Settings) -> None:
    log(f"== [2/9] Full-Duplex-Bench reference checkout at {FDB_SOURCE_REVISION}")
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
    log(f"== [3/9] Scoring venv at {settings.scoring_venv}")
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
    # note (luojiaxuan): torchcodec installs without FFmpeg and only fails when
    # its decoder library loads, which the pinned backchannel evaluator does in
    # `judge`; loading it here surfaces a missing FFmpeg during setup.
    probe = subprocess.run(
        [
            str(settings.scoring_python),
            "-c",
            "from torchcodec.decoders import AudioDecoder",
        ],
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(
            "ERROR: torchcodec cannot load its FFmpeg libraries in the scoring venv; "
            "install FFmpeg (shared libraries, version 4 to 8) and rerun setup."
        )
    else:
        pass


def setup_dataset(
    step: str, dataset: Path, revision_file: Path, subsets: tuple[str, ...]
) -> None:
    log(f"== {step} FDB {dataset.name} dataset at {dataset}")
    dataset.mkdir(parents=True, exist_ok=True)
    checksum_lines: list[str] = []
    for subset in sorted(subsets):
        if subset in CANDOR_SUBSETS:
            try:
                snapshot = Path(
                    snapshot_download(
                        CANDOR_REPO_ID,
                        repo_type="dataset",
                        revision=CANDOR_REVISION,
                        allow_patterns=f"{subset}/*",
                    )
                )
            except GatedRepoError as error:
                raise SystemExit(
                    f"ERROR: {CANDOR_REPO_ID} is gated; accept its license on the "
                    "Hugging Face Hub and export HF_TOKEN for that account."
                ) from error
            if not (dataset / subset).is_dir():
                shutil.copytree(snapshot / subset, dataset / subset)
            else:
                pass
            checksum_lines.append(f"{CANDOR_REPO_ID}@{CANDOR_REVISION}  ./{subset}\n")
        else:
            archive_path = Path(
                hf_hub_download(
                    DATASET_ARCHIVE_REPO_ID,
                    f"{dataset.name}/{subset}.zip",
                    repo_type="dataset",
                    revision=DATASET_ARCHIVE_REVISION,
                )
            )
            if not (dataset / subset).is_dir():
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(dataset)
            else:
                pass
            checksum_lines.append(f"{file_sha256(archive_path)}  ./{subset}.zip\n")
    shutil.rmtree(dataset / "__MACOSX", ignore_errors=True)
    # An all-archive dataset gets the text of sha256sum ./*.zip, so its revision matches earlier setups.
    revision = hashlib.sha256("".join(checksum_lines).encode()).hexdigest()
    revision_file.write_text(f"sha256:{revision}\n")
    for subset in subsets:
        sample_count = sum(path.is_dir() for path in (dataset / subset).iterdir())
        log(f"   {subset}: {sample_count} samples")


def setup_parakeet(settings: Settings) -> None:
    log("== [6/9] Parakeet ASR checkpoint")
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


def setup_aligner_venv(settings: Settings) -> None:
    log(f"   forced-alignment worker venv at {settings.aligner_python.parent.parent}")
    if not settings.aligner_python.is_file():
        subprocess.run(
            [
                "uv",
                "venv",
                "--python",
                SCORING_PYTHON_VERSION,
                str(settings.aligner_python.parent.parent),
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
            str(settings.aligner_python),
            *ALIGNER_PACKAGES,
        ],
        check=True,
    )


def setup_models(settings: Settings) -> None:
    engine = settings.engine
    if engine.launch_config is None:
        log(f"== [7/9] Model under test at {settings.model_path}")
        if not (settings.model_path / "config.json").is_file():
            snapshot_download(
                engine.model_id,
                revision=settings.model_revision,
                local_dir=settings.model_path,
            )
        else:
            pass
    else:
        log(f"== [7/9] Models of {engine.name} in the Hugging Face cache")
        for model_id in engine.hf_model_ids:
            snapshot_download(model_id)
    if engine.needs_aligner:
        setup_aligner_venv(settings)
    else:
        pass
    log(f"== [8/9] Judge model at {settings.judge_model_path} (JUDGE={settings.judge})")
    if settings.judge == "qwen" and not settings.judge_revision_file.is_file():
        judge_revision = model_info(JUDGE_MODEL_ID).sha
        snapshot_download(
            JUDGE_MODEL_ID, revision=judge_revision, local_dir=settings.judge_model_path
        )
        settings.judge_revision_file.write_text(f"{judge_revision}\n")
    else:
        pass


def setup_listening_models() -> None:
    log("== [9/9] UTMOS and WavLM speaker models for the listening metrics")
    ensure_utmos_assets()
    ensure_speaker_similarity_assets()


def setup(settings: Settings) -> None:
    setup_model_extra()
    (settings.fdb_work / "models").mkdir(parents=True, exist_ok=True)
    setup_reference_source(settings)
    setup_scoring_venv(settings)
    setup_dataset("[4/9]", settings.dataset, settings.dataset_revision_file, SUBSETS)
    setup_dataset(
        "[5/9]", settings.dataset_v10, settings.dataset_v10_revision_file, V10_SUBSETS
    )
    setup_parakeet(settings)
    setup_models(settings)
    setup_listening_models()
    log("Setup complete.")
