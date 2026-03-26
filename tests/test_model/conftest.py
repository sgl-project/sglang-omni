# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for test_model tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.utils import (
    download_dataset,
    find_free_port,
    start_s2pro_server,
    stop_server,
    wait_server_healthy,
)


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download the mini seed-tts-eval dataset via huggingface_hub."""
    cache_dir = tmp_path_factory.mktemp("seed_tts_eval")
    return download_dataset(cache_dir)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the s2-pro server, wait until healthy, and yield (proc, port)."""
    port = find_free_port()
    log_dir = tmp_path_factory.mktemp("server_logs")
    log_file = log_dir / "server.log"
    proc = start_s2pro_server(log_file, port)
    wait_server_healthy(port, proc, log_file)
    yield proc, port
    stop_server(proc)
