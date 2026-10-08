# SPDX-License-Identifier: Apache-2.0
"""Runs every transcription on one thread, the thread that loaded the model."""

from __future__ import annotations

import asyncio
import threading
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from sglang_omni_mlx.qwen3_asr.transcriber import (
    Qwen3ASRTranscriber,
    TranscriptionOptions,
    TranscriptionResult,
)


class TranscriptionWorker:
    """Serializes requests: one model, one MLX stream, one request at a time."""

    def __init__(self, model_directory: Path) -> None:
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="qwen3-asr"
        )
        self.transcriber: Qwen3ASRTranscriber = self.executor.submit(
            Qwen3ASRTranscriber, model_directory
        ).result()
        self.state_lock = threading.Lock()
        self.state_counts: Counter[str] = Counter()

    def request_states(self) -> dict[str, int]:
        """Requests waiting for the worker and running on it; empty when idle."""
        with self.state_lock:
            return {state: count for state, count in self.state_counts.items() if count}

    def move_state(self, leaving: str | None, entering: str | None) -> None:
        with self.state_lock:
            if leaving is not None:
                self.state_counts[leaving] -= 1
            else:
                pass
            if entering is not None:
                self.state_counts[entering] += 1
            else:
                pass

    def run(
        self,
        samples: np.ndarray,
        options: TranscriptionOptions,
        cancel: threading.Event,
    ) -> TranscriptionResult:
        self.move_state("queued", "running")
        try:
            return self.transcriber.transcribe(samples, options, cancel)
        finally:
            self.move_state("running", None)

    async def transcribe(
        self,
        samples: np.ndarray,
        options: TranscriptionOptions,
        cancel: threading.Event,
    ) -> TranscriptionResult:
        self.move_state(None, "queued")
        future = self.executor.submit(self.run, samples, options, cancel)
        future.add_done_callback(self.forget_if_never_run)
        return await asyncio.wrap_future(future)

    def forget_if_never_run(self, future: Future[TranscriptionResult]) -> None:
        # A caller that gives up while queued cancels the job before it runs.
        if future.cancelled():
            self.move_state("queued", None)
        else:
            pass
