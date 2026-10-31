# SPDX-License-Identifier: Apache-2.0
"""Model and judge servers: each step starts the one it needs and stops it when done."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.duplex.fdb.common import (
    JUDGE_MODEL_ID,
    JUDGE_SERVED_MODEL,
    REPO_ROOT,
    Settings,
    follow_log,
    log,
    log_tail,
    visible_devices,
)

SERVER_READY_TIMEOUT_S = 1800
JUDGE_MEM_FRACTION_STATIC = "0.8"
JUDGE_DECODING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": 512,
}
JUDGE_SEEDS = [1, 2, 3]
# note (jeffro): the cascade profile's alignment node posts to this port.
ALIGNER_PORT = 8010
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNER_SCRIPT = REPO_ROOT / "examples" / "streaming_cascade" / "alignment_worker.py"


def model_server_command(settings: Settings) -> list[str]:
    if settings.engine.launch_config is None:
        model_arguments = [
            "--model-path",
            str(settings.model_path),
            "--variant",
            "session",
        ]
    else:
        model_arguments = ["--config", str(REPO_ROOT / settings.engine.launch_config)]
    if settings.engine.sets_max_sessions:
        model_arguments += ["--max_sessions", str(settings.engine.max_sessions)]
    else:
        pass
    return [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        *model_arguments,
        "--enable-realtime",
        "--port",
        str(settings.server_port),
    ]


def aligner_command(settings: Settings) -> list[str]:
    return [
        str(settings.aligner_python),
        str(ALIGNER_SCRIPT),
        "--model",
        ALIGNER_MODEL_ID,
        "--device",
        "cuda:0",
        "--language",
        "English",
        "--port",
        str(ALIGNER_PORT),
    ]


def model_server_gpus(settings: Settings) -> str:
    """The job GPU, or every visible GPU for a system that spans several."""
    if settings.engine.server_gpu_count == 1:
        return settings.gpu
    elif len(visible_devices()) != settings.engine.server_gpu_count:
        raise SystemExit(
            f"ERROR: ENGINE={settings.engine.name} runs one server on "
            f"{settings.engine.server_gpu_count} GPUs; set CUDA_VISIBLE_DEVICES "
            "to exactly that many."
        )
    else:
        return ",".join(visible_devices())


def judge_server_command(settings: Settings) -> list[str]:
    return [
        str(Path(sys.executable).with_name("sglang")),
        "serve",
        "--model-path",
        str(settings.judge_model_path),
        "--served-model-name",
        JUDGE_SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(settings.judge_port),
        "--reasoning-parser",
        "qwen3",
        "--mem-fraction-static",
        JUDGE_MEM_FRACTION_STATIC,
        "--nccl-port",
        str(settings.judge_nccl_port),
    ]


def write_judge_config(settings: Settings, command: list[str]) -> Path:
    """Write the launch receipt and the pinned custom-judge config that hashes it."""
    revision = settings.judge_revision_file.read_text().strip()
    settings.judge_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = settings.judge_dir / "launch-receipt.json"
    receipt = {
        "server_command": command,
        "runtime": f"sglang {version('sglang')}",
        "model_id": JUDGE_MODEL_ID,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "precision": "bf16",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    config = {
        "model_id": JUDGE_MODEL_ID,
        "model_revision": revision,
        "tokenizer_id": JUDGE_MODEL_ID,
        "tokenizer_revision": revision,
        "served_model": JUDGE_SERVED_MODEL,
        "precision": "bf16",
        "enable_thinking": False,
        "decoding": JUDGE_DECODING,
        "seeds": JUDGE_SEEDS,
        "server_launch_receipt": receipt_path.name,
        "server_launch_receipt_sha256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
    }
    config_path = settings.judge_dir / "judge-config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


def ensure_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(
                f"ERROR: something already serves port {port}; stop it first."
            )
        else:
            pass


@dataclass(frozen=True)
class RunningServer:
    """A server that running_server started, and the log its output goes to."""

    name: str
    process: subprocess.Popen
    log_file: Path

    def ensure_alive(self, when: str) -> None:
        """Stop the step with the end of the server log if the server exited;
        when places the exit, as in "during the recordings"."""
        returncode = self.process.poll()
        if returncode is not None:
            raise SystemExit(
                f"ERROR: the {self.name} exited with code {returncode} {when}; "
                f"last lines of {self.log_file}:\n{log_tail(self.log_file)}"
            )
        else:
            pass


@contextmanager
def running_server(
    name: str,
    command: list[str],
    gpu: str,
    port: int,
    log_file: Path,
    health_path: str,
    ready_text: str,
    env: dict[str, str],
) -> Iterator[RunningServer]:
    """Serve on one GPU until the block exits, then stop the server's process group."""
    ensure_port_free(port)
    log(f"== Starting the {name} on GPU {gpu} (log: {log_file})")
    # note (luojiaxuan): the follower reads from the first byte, so a log left
    # by an earlier attempt is emptied before the follower opens it.
    log_file.write_text("")
    started = time.monotonic()
    with follow_log(log_file):
        try:
            process = start_server_from_cmd(
                command,
                log_file,
                port,
                timeout=SERVER_READY_TIMEOUT_S,
                env={
                    **env,
                    "CUDA_VISIBLE_DEVICES": gpu,
                    # A taken port must fail here. Falling over to a random port
                    # would leave the client talking to the port in the command.
                    "SGLANG_OMNI_STRICT_PORT": "1",
                },
                strip_proxy=True,
                health_path=health_path,
                health_body_contains=ready_text,
            )
        except (RuntimeError, TimeoutError):
            raise SystemExit(
                f"ERROR: the {name} did not become ready; "
                f"last lines of {log_file}:\n{log_tail(log_file)}"
            ) from None
        log(f"   ready after {time.monotonic() - started:.0f}s")
        try:
            yield RunningServer(name, process, log_file)
        finally:
            log(f"== Stopping the {name}")
            stop_server(process)


@contextmanager
def model_server(settings: Settings, log_file: Path) -> Iterator[RunningServer]:
    """The system under test, after its forced-alignment worker when it has
    one; the worker runs on the last of the server's GPUs, beside the speech
    models."""
    gpus = model_server_gpus(settings)
    with ExitStack() as stack:
        if settings.engine.needs_aligner:
            stack.enter_context(
                running_server(
                    "aligner",
                    aligner_command(settings),
                    gpus.split(",")[-1],
                    ALIGNER_PORT,
                    log_file.with_name(f"aligner-{log_file.name}"),
                    health_path="/health",
                    ready_text="ready",
                    env={**settings.cache_env(), "PYTHONPATH": str(REPO_ROOT)},
                )
            )
        else:
            pass
        server = stack.enter_context(
            running_server(
                "model server",
                model_server_command(settings),
                gpus,
                settings.server_port,
                log_file,
                health_path="/v1/realtime/capabilities",
                ready_text=settings.engine.ready_text,
                env=settings.cache_env(),
            )
        )
        yield server


@contextmanager
def judge_server(settings: Settings, log_file: Path) -> Iterator[RunningServer]:
    command = judge_server_command(settings)
    write_judge_config(settings, command)
    with running_server(
        "judge server",
        command,
        settings.gpu,
        settings.judge_port,
        log_file,
        health_path="/v1/models",
        ready_text=JUDGE_SERVED_MODEL,
        env=settings.cache_env(),
    ) as server:
        yield server


def serve_in_foreground(command: list[str], gpu: str) -> None:
    """Replace this process with the server; Ctrl-C stops it."""
    os.execve(command[0], command, {**os.environ, "CUDA_VISIBLE_DEVICES": gpu})


def serve_model(settings: Settings) -> None:
    serve_in_foreground(model_server_command(settings), model_server_gpus(settings))


def serve_judge(settings: Settings) -> None:
    command = judge_server_command(settings)
    log(f"wrote {write_judge_config(settings, command)}")
    serve_in_foreground(command, settings.gpu)
