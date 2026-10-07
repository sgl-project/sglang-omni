# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from sglang_omni_mlx.qwen3_asr import worker as worker_module  # noqa: E402
from sglang_omni_mlx.qwen3_asr.transcriber import (  # noqa: E402
    FinishReason,
    TranscriptionOptions,
    TranscriptionResult,
)


class GatedTranscriber:
    """Holds every transcription until released."""

    def __init__(self, model_directory: Path) -> None:
        self.release = threading.Event()

    def transcribe(
        self,
        samples: np.ndarray,
        options: TranscriptionOptions,
        cancel: threading.Event,
    ) -> TranscriptionResult:
        self.release.wait(timeout=10)
        return TranscriptionResult(
            text="",
            language=None,
            generated_token_count=0,
            finish_reason=FinishReason.STOP,
        )


@pytest.mark.asyncio
async def test_request_states_count_queued_and_running_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "Qwen3ASRTranscriber", GatedTranscriber)
    worker = worker_module.TranscriptionWorker(Path("unused"))
    assert worker.request_states() == {}
    samples = np.zeros(160, dtype=np.float32)
    jobs = [
        asyncio.create_task(
            worker.transcribe(samples, TranscriptionOptions(), threading.Event())
        )
        for _ in range(2)
    ]
    for _ in range(100):
        if worker.request_states() == {"running": 1, "queued": 1}:
            break
        await asyncio.sleep(0.01)
    assert worker.request_states() == {"running": 1, "queued": 1}
    worker.transcriber.release.set()
    await asyncio.gather(*jobs)
    assert worker.request_states() == {}


@pytest.mark.asyncio
async def test_a_request_cancelled_while_queued_leaves_no_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "Qwen3ASRTranscriber", GatedTranscriber)
    worker = worker_module.TranscriptionWorker(Path("unused"))
    samples = np.zeros(160, dtype=np.float32)
    running = asyncio.create_task(
        worker.transcribe(samples, TranscriptionOptions(), threading.Event())
    )
    queued = asyncio.create_task(
        worker.transcribe(samples, TranscriptionOptions(), threading.Event())
    )
    for _ in range(100):
        if worker.request_states() == {"running": 1, "queued": 1}:
            break
        await asyncio.sleep(0.01)
    queued.cancel()
    worker.transcriber.release.set()
    await running
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert worker.request_states() == {}
