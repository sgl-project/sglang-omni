# SPDX-License-Identifier: Apache-2.0
"""Shared test utilities."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

from tests.test_model.helpers import disable_proxy

logger = logging.getLogger(__name__)

MODEL_PATH = "fishaudio/s2-pro"
CONFIG_PATH = "examples/configs/s2pro_tts.yaml"
DATASET_REPO = "zhaochenyang20/seed-tts-eval-mini"
STARTUP_TIMEOUT = 600  # seconds


def find_free_port() -> int:
    """Find and return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def download_dataset(cache_dir: Path) -> Path:
    """Download the mini seed-tts-eval dataset via huggingface_hub."""
    path = snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        local_dir=str(cache_dir / "data"),
    )
    return Path(path)


def start_s2pro_server(log_file: Path, port: int) -> subprocess.Popen:
    """Start the s2-pro TTS server and return the process handle."""
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        MODEL_PATH,
        "--config",
        CONFIG_PATH,
        "--port",
        str(port),
    ]
    with open(log_file, "w") as log_handle:
        return subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def wait_server_healthy(
    port: int,
    proc: subprocess.Popen,
    log_file: Path,
    timeout: int = STARTUP_TIMEOUT,
) -> None:
    """Poll the server health endpoint until healthy or raise on timeout."""
    api_base = f"http://localhost:{port}"
    for _ in range(timeout):
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited with code {proc.returncode}.\n{log_file.read_text()}"
            )
        try:
            with disable_proxy():
                resp = requests.get(f"{api_base}/health", timeout=2)
            if resp.status_code == 200 and "healthy" in resp.text:
                return
        except requests.RequestException as exc:
            logger.debug("Health check failed (transient): %s", exc)
        time.sleep(1)

    stop_server(proc)
    raise RuntimeError(
        f"Server did not become healthy within {timeout}s.\n{log_file.read_text()}"
    )


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server process group (SIGTERM, then SIGKILL)."""
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
