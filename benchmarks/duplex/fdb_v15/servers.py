# SPDX-License-Identifier: Apache-2.0
"""Model and judge servers: each step starts the one it needs and stops it when done."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.duplex.fdb_v15.common import (
    JUDGE_MODEL_ID,
    JUDGE_SERVED_MODEL,
    Settings,
    log,
    log_tail,
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


def model_server_command(settings: Settings) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--config",
        str(settings.server_config),
        "--model-path",
        str(settings.model_path),
        "--enable-realtime",
        "--port",
        str(settings.server_port),
    ]


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


@contextmanager
def running_server(
    name: str,
    command: list[str],
    gpu: str,
    port: int,
    log_file: Path,
    health_path: str,
    ready_text: str,
) -> Iterator[None]:
    """Serve on one GPU until the block exits, then stop the server's process group."""
    ensure_port_free(port)
    log(f"== Starting the {name} on GPU {gpu} (log: {log_file})")
    started = time.monotonic()
    try:
        process = start_server_from_cmd(
            command,
            log_file,
            port,
            timeout=SERVER_READY_TIMEOUT_S,
            env={
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
        yield
    finally:
        log(f"== Stopping the {name}")
        stop_server(process)


@contextmanager
def model_server(settings: Settings, log_file: Path) -> Iterator[None]:
    with running_server(
        "model server",
        model_server_command(settings),
        settings.gpu,
        settings.server_port,
        log_file,
        health_path="/v1/realtime/capabilities",
        ready_text='"native_full_duplex":true',
    ):
        yield


@contextmanager
def judge_server(settings: Settings, log_file: Path) -> Iterator[None]:
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
    ):
        yield


def serve_in_foreground(command: list[str], gpu: str) -> None:
    """Replace this process with the server; Ctrl-C stops it."""
    os.execve(command[0], command, {**os.environ, "CUDA_VISIBLE_DEVICES": gpu})


def serve_model(settings: Settings) -> None:
    serve_in_foreground(model_server_command(settings), settings.gpu)


def serve_judge(settings: Settings) -> None:
    command = judge_server_command(settings)
    log(f"wrote {write_judge_config(settings, command)}")
    serve_in_foreground(command, settings.gpu)
