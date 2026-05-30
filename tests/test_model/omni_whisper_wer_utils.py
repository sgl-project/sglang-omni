# SPDX-License-Identifier: Apache-2.0
"""Shared Omni Whisper router helpers for CI WER evaluation."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
)

WHISPER_MODEL_PATH = "openai/whisper-large-v3"
WHISPER_ASR_WORKER_ARGS = "--stages.0.factory-args.max-running-requests 1"
WHISPER_ROUTER_STARTUP_TIMEOUT = 600
REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_CLEANUP_SCRIPT = REPO_ROOT / ".github/scripts/ensure_gpus_idle.sh"
# Colocated Qwen3 router + Whisper router back-to-back needs headroom on both cards.
GPU_IDLE_THRESHOLD_MB = 2048
GPU_IDLE_WAIT_SECONDS = 600
GPU_IDLE_POLL_SECONDS = 5


def wait_for_gpu_memory_release(
    *,
    memory_threshold_mb: int | None = None,
    wait_timeout_seconds: int | None = None,
    poll_seconds: int | None = None,
) -> None:
    """Kill orphan GPU processes and block until every GPU is below threshold."""
    if not GPU_CLEANUP_SCRIPT.exists():
        raise FileNotFoundError(f"GPU cleanup script missing: {GPU_CLEANUP_SCRIPT}")

    env = os.environ.copy()
    env["OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB"] = str(
        memory_threshold_mb
        if memory_threshold_mb is not None
        else GPU_IDLE_THRESHOLD_MB
    )
    env["OMNI_CI_GPU_CLEAN_WAIT_SECONDS"] = str(
        wait_timeout_seconds
        if wait_timeout_seconds is not None
        else GPU_IDLE_WAIT_SECONDS
    )
    env["OMNI_CI_GPU_CLEAN_POLL_SECONDS"] = str(
        poll_seconds if poll_seconds is not None else GPU_IDLE_POLL_SECONDS
    )

    print(
        f"[gpu cleanup] running ensure_gpus_idle "
        f"(threshold={env['OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB']} MiB)...",
        flush=True,
    )
    result = subprocess.run(
        ["bash", str(GPU_CLEANUP_SCRIPT)],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "GPU memory was not released after stopping the inference server. "
            f"ensure_gpus_idle.sh exit={result.returncode}"
        )


@pytest.fixture(scope="module")
def omni_whisper_wer_router(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[ManagedRouterHandle]:
    """Launch DP=2 Whisper router for WER after upstream servers release GPU."""
    wait_for_gpu_memory_release()
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=WHISPER_MODEL_PATH,
        model_name=WHISPER_MODEL_PATH,
        worker_extra_args=WHISPER_ASR_WORKER_ARGS,
        wait_timeout=WHISPER_ROUTER_STARTUP_TIMEOUT,
        log_prefix="whisper_wer_router_logs",
    ) as router:
        yield router
