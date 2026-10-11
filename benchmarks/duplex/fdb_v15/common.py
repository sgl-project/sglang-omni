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

from sglang_omni.utils.port_claim import NCCL_PORT_BASE, NCCL_PORT_SPAN

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

DEFAULT_SERVER_PORT = 8097
DEFAULT_JUDGE_PORT = 30000
JOB_PORT_STRIDE = 10
JUDGE_NCCL_PORT_OFFSET = 1
NCCL_PORT_END = NCCL_PORT_BASE + NCCL_PORT_SPAN


@dataclass(frozen=True)
class Settings:
    fdb_work: Path
    scoring_venv: Path
    model_revision: str
    model_path: Path
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
    def judge_nccl_port(self) -> int:
        return self.judge_port + JUDGE_NCCL_PORT_OFFSET

    @property
    def judge_dir(self) -> Path:
        return self.fdb_work / "judge" / f"port-{self.judge_port}"

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
            "JUDGE_CONFIG": str(self.judge_dir / "judge-config.json"),
        }

    def apply_job_isolation(self) -> None:
        """Give this GPU its own compile and dynamic-module caches.

        Two jobs writing one Triton, Inductor, DeepGEMM, CUDA, NeMo or
        Transformers dynamic-module directory corrupt it.
        """
        cache_root = self.fdb_work / "cache" / f"gpu-{self.gpu}"
        assignments = {
            "SGLANG_CACHE_DIR": cache_root / "sglang",
            "TRITON_CACHE_DIR": cache_root / "triton",
            "TORCHINDUCTOR_CACHE_DIR": cache_root / "inductor",
            "CUDA_CACHE_PATH": cache_root / "nv",
            "HF_MODULES_CACHE": cache_root / "hf-modules",
            "NEMO_CACHE_DIR": cache_root / "nemo",
        }
        for name, path in assignments.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)


def env_text(name: str, default: str) -> str:
    return os.environ.get(name) or default


def env_path(name: str, default: Path) -> Path:
    return Path(env_text(name, str(default))).expanduser().absolute()


def resolve_visible_gpu() -> str:
    """The one GPU this job uses.

    ``CUDA_VISIBLE_DEVICES`` wins, so each terminal can export a different
    index and the default ``GPU=0`` does not put every job on card 0.
    ``GPU`` is used only when ``CUDA_VISIBLE_DEVICES`` is unset.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        if not visible:
            raise SystemExit("ERROR: CUDA_VISIBLE_DEVICES is empty.")
        else:
            pass
        devices = [part.strip() for part in visible.split(",") if part.strip()]
        if len(devices) != 1:
            raise SystemExit(
                "ERROR: set CUDA_VISIBLE_DEVICES to exactly one GPU index "
                f"(got {visible!r})."
            )
        else:
            pass
        gpu = devices[0]
        explicit_gpu = os.environ.get("GPU")
        if explicit_gpu is not None and explicit_gpu.strip() not in {"", gpu}:
            raise SystemExit(
                "ERROR: GPU and CUDA_VISIBLE_DEVICES disagree "
                f"(GPU={explicit_gpu!r}, CUDA_VISIBLE_DEVICES={visible!r}). "
                "Unset GPU, or set it to the same index."
            )
        else:
            pass
    else:
        gpu = env_text("GPU", "0")
    if not gpu.isdigit():
        raise SystemExit(f"ERROR: GPU must be a single numeric index, got {gpu!r}.")
    else:
        pass
    return gpu


def derived_port(env_name: str, base_port: int, gpu: str) -> int:
    """An explicit port, or ``base_port + 10 * gpu`` when the variable is unset."""
    raw = os.environ.get(env_name)
    if raw:
        return int(raw)
    else:
        pass
    return base_port + int(gpu) * JOB_PORT_STRIDE


def reject_nccl_range(name: str, port: int) -> None:
    if NCCL_PORT_BASE <= port < NCCL_PORT_END:
        raise SystemExit(
            f"ERROR: {name}={port} overlaps NCCL ports "
            f"{NCCL_PORT_BASE}-{NCCL_PORT_END - 1}. "
            "Pick another port, or leave it unset."
        )
    else:
        pass


def load_settings(run_name: str) -> Settings:
    """Machine settings from environment variables; the run name comes from the CLI."""
    fdb_work = env_path("FDB_WORK", Path.home() / "fdb")
    judge = env_text("JUDGE", "qwen")
    if judge not in JUDGE_NAMES:
        raise SystemExit(f"ERROR: JUDGE must be qwen or gpt, got '{judge}'.")
    else:
        pass
    gpu = resolve_visible_gpu()
    server_port = derived_port("SERVER_PORT", DEFAULT_SERVER_PORT, gpu)
    judge_port = derived_port("JUDGE_PORT", DEFAULT_JUDGE_PORT, gpu)
    if server_port == judge_port:
        raise SystemExit("ERROR: SERVER_PORT and JUDGE_PORT are the same.")
    else:
        pass
    reject_nccl_range("SERVER_PORT", server_port)
    reject_nccl_range("JUDGE_PORT", judge_port)
    reject_nccl_range("judge NCCL port", judge_port + JUDGE_NCCL_PORT_OFFSET)
    return Settings(
        fdb_work=fdb_work,
        scoring_venv=env_path("SCORING_VENV", fdb_work / "scoring-venv"),
        model_revision=env_text("MODEL_REVISION", DEFAULT_MODEL_REVISION),
        model_path=env_path("MODEL_PATH", fdb_work / "models" / "MiniCPM-o-4_5"),
        server_port=server_port,
        judge=judge,
        judge_model_path=env_path(
            "JUDGE_MODEL_PATH", fdb_work / "models" / "Qwen3.8-27B"
        ),
        judge_port=judge_port,
        gpu=gpu,
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
