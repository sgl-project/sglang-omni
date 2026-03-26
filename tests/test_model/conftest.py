# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and hooks for test_model tests."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from tests.test_model.helpers import disable_proxy
from tests.utils import find_free_port

logger = logging.getLogger(__name__)

MODEL_PATH = "fishaudio/s2-pro"
CONFIG_PATH = "examples/configs/s2pro_tts.yaml"
DATASET_REPO = "zhaochenyang20/seed-tts-eval-mini"
STARTUP_TIMEOUT = 600  # seconds


@pytest.fixture(scope="module")
def server_port() -> int:
    """Allocate a free TCP port for the server."""
    return find_free_port()


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download the mini seed-tts-eval dataset via huggingface_hub."""
    from huggingface_hub import snapshot_download

    cache_dir = tmp_path_factory.mktemp("seed_tts_eval")
    path = snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        local_dir=str(cache_dir / "data"),
    )
    return Path(path)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory, server_port: int):
    """Start the s2-pro server and wait until healthy."""
    log_dir = tmp_path_factory.mktemp("server_logs")
    log_file = log_dir / "server.log"
    with open(log_file, "w") as log_handle:
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
            str(server_port),
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        api_base = f"http://localhost:{server_port}"
        is_healthy = False
        for _ in range(STARTUP_TIMEOUT):
            if proc.poll() is not None:
                server_log = log_file.read_text()
                pytest.fail(f"Server exited with code {proc.returncode}.\n{server_log}")
            try:
                with disable_proxy():
                    resp = requests.get(f"{api_base}/health", timeout=2)
                if resp.status_code == 200 and "healthy" in resp.text:
                    is_healthy = True
                    break
            except requests.RequestException as exc:
                logger.debug("Health check failed (transient): %s", exc)
            time.sleep(1)

        if not is_healthy:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
            server_log = log_file.read_text()
            pytest.fail(
                f"Server did not become healthy within {STARTUP_TIMEOUT}s.\n"
                f"{server_log}"
            )

        yield proc

        # Teardown
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()  # No timeout — SIGKILL is guaranteed to terminate
