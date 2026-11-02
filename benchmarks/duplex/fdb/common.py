# SPDX-License-Identifier: Apache-2.0
"""Pinned inputs and settings shared by every FDB v1.5 and v1.0 step.

Settings come from environment variables; see the Settings table in
docs/developer_reference/full_duplex_guide.md.
"""

from __future__ import annotations

import codecs
import io
import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import JsonValue

from benchmarks.duplex.client import MAX_TIMEOUT_S
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
# note (luojiaxuan): v1.0 artifacts of a repeat live under repeat-N/v10/.
V10_DIR = "v10"
# note (luojiaxuan): the per-reply listening metrics of a repeat live under
# repeat-N/listening/.
LISTENING_DIR = "listening"


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
    v10_session_timeout_s: str
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
    def dataset_v10(self) -> Path:
        return self.dataset_dir / "v1.0"

    @property
    def dataset_v10_revision_file(self) -> Path:
        return self.dataset_dir / "v1.0.revision"

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
            "FDB10_DATASET": str(self.dataset_v10),
            "SCORING_PYTHON": str(self.scoring_python),
            "PARAKEET_NEMO": str(self.parakeet_nemo),
            "PARAKEET_SHA256": PARAKEET_SHA256,
            "MODEL_ID": MODEL_ID,
            "MODEL_REVISION": self.model_revision,
            "REALTIME_URL": self.realtime_url,
            "SESSION_TIMEOUT_S": self.session_timeout_s,
            "V10_SESSION_TIMEOUT_S": self.v10_session_timeout_s,
            "JUDGE_URL": self.judge_url,
            "JUDGE_CONFIG": str(self.judge_dir / "judge-config.json"),
        }

    def cache_env(self) -> dict[str, str]:
        """This GPU's own compile and dynamic-module caches, created on first use.

        Two processes writing one Triton, Inductor, DeepGEMM, CUDA, NeMo or
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
        for path in assignments.values():
            path.mkdir(parents=True, exist_ok=True)
        return {name: str(path) for name, path in assignments.items()}

    def apply_job_isolation(self) -> None:
        os.environ.update(self.cache_env())

    def on_gpu(self, gpu: str) -> Settings:
        """The same job with its model server on another GPU and that GPU's port."""
        return replace(
            self,
            gpu=gpu,
            server_port=derived_port("SERVER_PORT", DEFAULT_SERVER_PORT, gpu),
        )


def env_text(name: str, default: str) -> str:
    return os.environ.get(name) or default


def env_path(name: str, default: Path) -> Path:
    return Path(env_text(name, str(default))).expanduser().absolute()


def session_timeout(name: str, default: str) -> str:
    """A per-session deadline in seconds, within the recorder's ceiling."""
    text = env_text(name, default)
    try:
        value = float(text)
    except ValueError:
        raise SystemExit(
            f"ERROR: {name} must be a number of seconds, got {text!r}."
        ) from None
    if not 0 < value <= MAX_TIMEOUT_S:
        raise SystemExit(
            f"ERROR: {name}={text} is outside 0-{MAX_TIMEOUT_S:g} s, the recorder's ceiling."
        )
    else:
        pass
    return text


def visible_devices() -> tuple[str, ...]:
    """The GPU indices ``CUDA_VISIBLE_DEVICES`` lists, which must be set."""
    visible = os.environ["CUDA_VISIBLE_DEVICES"].strip()
    if not visible:
        raise SystemExit("ERROR: CUDA_VISIBLE_DEVICES is empty.")
    else:
        pass
    devices = tuple(part.strip() for part in visible.split(","))
    if not all(device.isdigit() for device in devices) or len(set(devices)) != len(
        devices
    ):
        raise SystemExit(
            "ERROR: CUDA_VISIBLE_DEVICES must list distinct numeric GPU indices "
            f"(got {visible!r})."
        )
    else:
        pass
    return devices


def resolve_visible_gpu() -> str:
    """The job GPU: it runs generate's one model server unless --gpus lists
    more, runs asr and judge, and names the ports and caches.

    ``CUDA_VISIBLE_DEVICES`` wins, so each terminal can export its own
    indices; the job GPU is the first one. ``GPU`` is used only when
    ``CUDA_VISIBLE_DEVICES`` is unset.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu = visible_devices()[0]
        explicit_gpu = os.environ.get("GPU")
        if explicit_gpu is not None and explicit_gpu.strip() not in {"", gpu}:
            raise SystemExit(
                "ERROR: GPU and CUDA_VISIBLE_DEVICES disagree "
                f"(GPU={explicit_gpu!r}, "
                f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}). "
                "Unset GPU, or set it to the first index."
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
        session_timeout_s=session_timeout("SESSION_TIMEOUT_S", "90"),
        v10_session_timeout_s=session_timeout("V10_SESSION_TIMEOUT_S", "230"),
        run_name=run_name,
    )


LOG_TAIL_LINES = 20
FOLLOW_INTERVAL_S = 0.5
LOGGER = logging.getLogger("benchmarks.duplex.fdb")


def configure_logging(level: str) -> None:
    """Print the runbook's messages as they are; at debug, also the followed logs.
    Other loggers in this process keep their own setup."""
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    LOGGER.setLevel(level.upper())


def log(message: str) -> None:
    LOGGER.info(message)


def echo_appended_lines(handle: BinaryIO, prefix: str, stop: threading.Event) -> None:
    """Emit the lines appended to handle at DEBUG every FOLLOW_INTERVAL_S; once
    stop is set, read to the end and emit the rest, an unterminated line too.
    The bytes are decoded as UTF-8 with universal newlines; a character or a
    CRLF that a read splits is held until the rest of its bytes arrive."""
    decoder = io.IncrementalNewlineDecoder(
        codecs.getincrementaldecoder("utf-8")(errors="replace"), translate=True
    )
    partial = ""
    is_stopping = False
    while not is_stopping:
        is_stopping = stop.wait(FOLLOW_INTERVAL_S)
        appended = decoder.decode(handle.read(), final=is_stopping)
        *lines, partial = (partial + appended).split("\n")
        for line in lines:
            LOGGER.debug(f"{prefix} {line}")
    if partial:
        LOGGER.debug(f"{prefix} {partial}")
    else:
        pass


@contextmanager
def follow_log(path: Path) -> Iterator[None]:
    """At DEBUG, stream path from its first byte, prefixed with [its stem], while
    the block runs. The file must exist and only grow while it is followed."""
    if LOGGER.isEnabledFor(logging.DEBUG):
        stop = threading.Event()
        with path.open("rb") as handle:
            follower = threading.Thread(
                target=echo_appended_lines, args=(handle, f"[{path.stem}]", stop)
            )
            follower.start()
            try:
                yield
            finally:
                stop.set()
                follower.join()
    else:
        yield


def step_command(step: str, settings: Settings, repeat: int | None = None) -> str:
    """The CLI line for a step of this run, for hints in messages."""
    repeat_argument = "" if repeat is None else f" --repeat {repeat}"
    return (
        f"python -m benchmarks.duplex.fdb {step} "
        f"--run-name {settings.run_name}{repeat_argument}"
    )


def selected_v15(settings: Settings, repeat: int) -> bool:
    """Whether a generated repeat selected v1.5 pairs; sample-ids.txt is written
    by generate before anything is recorded, so a missing file means the repeat
    was never generated."""
    ids_file = settings.repeat_dir(repeat) / "sample-ids.txt"
    if not ids_file.is_file():
        raise SystemExit(
            f"ERROR: run `{step_command('generate', settings, repeat)}` first."
        )
    else:
        return ids_file.read_text().strip() != ""


def require_v10_export(settings: Settings, repeat: int) -> Path | None:
    """The v1.0 export tree of a repeat that selected v1.0 samples; None for a
    v1.5-only repeat. A selected v1.0 whose export never landed stops the step,
    as a missing v1.5 export does."""
    repeat_dir = settings.repeat_dir(repeat)
    if not (repeat_dir / V10_DIR / "sample-ids.txt").is_file():
        return None
    else:
        pass
    tree = repeat_dir / V10_DIR / "reference"
    if not (tree / "manifest.json").is_file():
        raise SystemExit(
            f"ERROR: {tree / 'manifest.json'} is missing: `generate` did not finish "
            f"the v1.0 export. Delete {repeat_dir} and run "
            f"`{step_command('generate', settings, repeat)}` again."
        )
    else:
        return tree


def log_tail(path: Path) -> str:
    return "\n".join(path.read_text().splitlines()[-LOG_TAIL_LINES:])


def read_json(path: Path) -> dict[str, JsonValue]:
    return json.loads(path.read_text(encoding="utf-8"))


def reference_command(settings: Settings, phase: str, *arguments: str) -> list[str]:
    """A benchmark_duplex_reference phase, run in the scoring venv."""
    return [
        str(settings.scoring_python),
        "-m",
        "benchmarks.eval.benchmark_duplex_reference",
        phase,
        *arguments,
    ]


def v10_command(settings: Settings, phase: str, *arguments: str) -> list[str]:
    """A benchmark_duplex_v10 reference phase, run in the scoring venv."""
    return [
        str(settings.scoring_python),
        "-m",
        "benchmarks.eval.benchmark_duplex_v10",
        phase,
        *arguments,
    ]


def listening_command(python: Path, phase: str, *arguments: str) -> list[str]:
    """A benchmark_duplex_listening phase, run with the given interpreter."""
    return [
        str(python),
        "-m",
        "benchmarks.eval.benchmark_duplex_listening",
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
