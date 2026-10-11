# SPDX-License-Identifier: Apache-2.0
"""What every model's transcriber shares, and the worker that runs one."""

from __future__ import annotations

import asyncio
import enum
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Generic, Protocol, TypeVar

import numpy as np


class FinishReason(enum.Enum):
    STOP = "stop"
    LENGTH = "length"


class CancelCheck(Protocol):
    def is_set(self) -> bool: ...


class TranscriptionCancelled(Exception):
    """The caller gave up on the transcription."""


OptionsT = TypeVar("OptionsT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)


class Transcriber(Protocol[OptionsT, ResultT]):
    def transcribe(
        self, samples: np.ndarray, options: OptionsT, cancel: CancelCheck
    ) -> ResultT: ...


class SerialWorker(Generic[OptionsT, ResultT]):
    """Serializes requests: one model, one MLX stream, one request at a time.

    The transcriber is built on the worker thread, the only thread that runs it.
    """

    def __init__(
        self,
        load_transcriber: Callable[[], Transcriber[OptionsT, ResultT]],
        thread_name: str,
    ) -> None:
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=thread_name
        )
        self.transcriber = self.executor.submit(load_transcriber).result()
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
        options: OptionsT,
        cancel: threading.Event,
    ) -> ResultT:
        self.move_state("queued", "running")
        try:
            return self.transcriber.transcribe(samples, options, cancel)
        finally:
            self.move_state("running", None)

    async def transcribe(
        self,
        samples: np.ndarray,
        options: OptionsT,
        cancel: threading.Event,
    ) -> ResultT:
        self.move_state(None, "queued")
        future = self.executor.submit(self.run, samples, options, cancel)
        future.add_done_callback(self.forget_if_never_run)
        return await asyncio.wrap_future(future)

    def forget_if_never_run(self, future: Future[ResultT]) -> None:
        # A caller that gives up while queued cancels the job before it runs.
        if future.cancelled():
            self.move_state("queued", None)
        else:
            pass
