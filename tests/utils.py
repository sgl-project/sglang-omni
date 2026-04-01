# SPDX-License-Identifier: Apache-2.0
"""Shared test utilities — model-agnostic helpers for launching and managing servers."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

from benchmarks.benchmarker.utils import wait_for_service
from tests.test_model.helpers import disable_proxy

STARTUP_TIMEOUT = 600


def find_free_port() -> int:
    """Find and return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_handle:
        socket_handle.bind(("", 0))
        return socket_handle.getsockname()[1]


def start_server(
    model_path: str,
    config_path: str,
    log_file: Path,
    port: int,
    timeout: int = STARTUP_TIMEOUT,
) -> subprocess.Popen:
    """Start a server and wait until healthy."""
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        model_path,
        "--config",
        config_path,
        "--port",
        str(port),
    ]
    with open(log_file, "w") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    with disable_proxy():
        wait_for_service(
            f"http://localhost:{port}",
            timeout=timeout,
            server_process=proc,
            server_log_file=log_file,
            health_body_contains="healthy",
        )
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server process group, tolerating already-dead processes."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, ChildProcessError):
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except (ProcessLookupError, ChildProcessError):
            pass
