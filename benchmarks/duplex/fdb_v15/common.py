# SPDX-License-Identifier: Apache-2.0
"""Pinned inputs and settings shared by every FDB v1.5 step.

Settings come from environment variables; see the Settings table in
docs/developer_reference/full_duplex_bench.md.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]

FDB_SOURCE_URL = "https://github.com/DanielLin94144/Full-Duplex-Bench.git"
FDB_SOURCE_REVISION = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
PARAKEET_REPO_ID = "nvidia/parakeet-tdt-0.6b-v2"
PARAKEET_FILENAME = "parakeet-tdt-0.6b-v2.nemo"
PARAKEET_REVISION = "ae9ad07059c7c739ffaf932226a8fe64ae2620b0"
PARAKEET_SHA256 = "d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c"

MODEL_ID = "openbmb/MiniCPM-o-4_5"
DEFAULT_MODEL_REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
RECORD_PROFILE = "minicpmo-native-pr2377"
ENGINE_LABEL = "minicpmo"

JudgeName = Literal["qwen", "gpt"]
JUDGE_NAMES: tuple[JudgeName, ...] = ("qwen", "gpt")
JUDGE_MODEL_ID = "Qwen/Qwen3.8-27B"
JUDGE_SERVED_MODEL = "qwen3.8-27b"
GPT_JUDGE_MODEL = "gpt-4o-2024-08-06"
CUSTOM_JUDGE_API_KEY_ENV = "CUSTOM_JUDGE_API_KEY"


@dataclass(frozen=True)
class Settings:
    fdb_work: Path
    scoring_venv: Path
    model_revision: str
    model_path: Path
    server_config: Path
    server_port: int
    judge: JudgeName
    judge_model_path: Path
    judge_port: int
    gpu: str
    session_timeout_s: str
    run_name: str

    @property
    def fdb_source(self) -> Path:
        return self.fdb_work / "Full-Duplex-Bench"

    @property
    def dataset_dir(self) -> Path:
        return self.fdb_work / "dataset"

    @property
    def dataset(self) -> Path:
        return self.dataset_dir / "v1.5"

    @property
    def dataset_revision_file(self) -> Path:
        return self.dataset_dir / "v1.5.revision"

    @property
    def parakeet_nemo(self) -> Path:
        return (
            self.fdb_work
            / "models"
            / PARAKEET_FILENAME.removesuffix(".nemo")
            / PARAKEET_FILENAME
        )

    @property
    def judge_revision_file(self) -> Path:
        return self.judge_model_path.with_name(self.judge_model_path.name + ".revision")

    @property
    def judge_dir(self) -> Path:
        return self.fdb_work / "judge"

    @property
    def realtime_url(self) -> str:
        return f"ws://127.0.0.1:{self.server_port}/v1/realtime"

    @property
    def judge_url(self) -> str:
        return f"http://127.0.0.1:{self.judge_port}/v1"

    @property
    def scoring_python(self) -> Path:
        return self.scoring_venv / "bin" / "python"

    @property
    def run_root(self) -> Path:
        return self.fdb_work / "runs" / self.run_name

    def repeat_dir(self, repeat: int) -> Path:
        return self.run_root / f"repeat-{repeat}"

    def shell_exports(self) -> dict[str, str]:
        """Variables used by the manual commands in the runbook's CLI reference."""
        return {
            "FDB_WORK": str(self.fdb_work),
            "FDB_SOURCE": str(self.fdb_source),
            "FDB_DATASET": str(self.dataset),
            "PARAKEET_NEMO": str(self.parakeet_nemo),
            "PARAKEET_SHA256": PARAKEET_SHA256,
            "MODEL_ID": MODEL_ID,
            "MODEL_REVISION": self.model_revision,
            "REALTIME_URL": self.realtime_url,
            "SESSION_TIMEOUT_S": self.session_timeout_s,
            "JUDGE_URL": self.judge_url,
        }


def env_text(name: str, default: str) -> str:
    return os.environ.get(name) or default


def env_path(name: str, default: Path) -> Path:
    return Path(env_text(name, str(default))).expanduser().absolute()


def load_settings(run_name: str) -> Settings:
    """Machine settings from environment variables; the run name comes from the CLI."""
    fdb_work = env_path("FDB_WORK", Path.home() / "fdb")
    judge = env_text("JUDGE", "qwen")
    if judge not in JUDGE_NAMES:
        raise SystemExit(f"ERROR: JUDGE must be qwen or gpt, got '{judge}'.")
    else:
        pass
    return Settings(
        fdb_work=fdb_work,
        scoring_venv=env_path("SCORING_VENV", fdb_work / "scoring-venv"),
        model_revision=env_text("MODEL_REVISION", DEFAULT_MODEL_REVISION),
        model_path=env_path("MODEL_PATH", fdb_work / "models" / "MiniCPM-o-4_5"),
        server_config=env_path(
            "SERVER_CONFIG", REPO_ROOT / "examples" / "full_duplex" / "minicpmo.yaml"
        ),
        server_port=int(env_text("SERVER_PORT", "8097")),
        judge=judge,
        judge_model_path=env_path(
            "JUDGE_MODEL_PATH", fdb_work / "models" / "Qwen3.8-27B"
        ),
        judge_port=int(env_text("JUDGE_PORT", "30000")),
        gpu=env_text("GPU", "0"),
        session_timeout_s=env_text("SESSION_TIMEOUT_S", "90"),
        run_name=run_name,
    )


LOG_TAIL_LINES = 20


def log(message: str) -> None:
    print(message, flush=True)


def step_command(step: str, settings: Settings, repeat: int | None = None) -> str:
    """The CLI line for a step of this run, for hints in messages."""
    repeat_argument = "" if repeat is None else f" --repeat {repeat}"
    return (
        f"python -m benchmarks.duplex.fdb_v15 {step} "
        f"--run-name {settings.run_name}{repeat_argument}"
    )


def log_tail(path: Path) -> str:
    return "\n".join(path.read_text().splitlines()[-LOG_TAIL_LINES:])


def reference_command(settings: Settings, phase: str, *arguments: str) -> list[str]:
    """A benchmark_duplex_reference phase, run in the scoring venv."""
    return [
        str(settings.scoring_python),
        "-m",
        "benchmarks.eval.benchmark_duplex_reference",
        phase,
        *arguments,
    ]


def run_command(
    command: list[str],
    visible_gpus: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> bool:
    """Run one command from the repository root; True when it exits 0.

    visible_gpus=None inherits CUDA_VISIBLE_DEVICES; "" hides every GPU.
    """
    env = dict(os.environ)
    if visible_gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_gpus
    else:
        pass
    env.update(extra_env or {})
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode == 0
