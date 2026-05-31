# SPDX-License-Identifier: Apache-2.0
"""Shared Omni Whisper router helpers for CI WER evaluation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
)
from tests.utils import wait_for_gpu_memory_release

WHISPER_MODEL_PATH = "openai/whisper-large-v3"
WHISPER_ASR_WORKER_ARGS = "--stages.0.factory-args.max-running-requests 1"
WHISPER_ROUTER_STARTUP_TIMEOUT = 600


@pytest.fixture
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
