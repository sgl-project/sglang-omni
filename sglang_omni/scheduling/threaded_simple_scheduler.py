# SPDX-License-Identifier: Apache-2.0
"""Threaded scheduler for simple CPU-bound pipeline stages."""

from __future__ import annotations

import asyncio
import inspect
import logging
import queue as _queue_mod
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


class ThreadedSimpleScheduler:
    """Run per-request work concurrently while preserving scheduler IO shape.

    This is meant for CPU-bound or blocking simple stages where legacy used
    async workers plus ``asyncio.to_thread``. GPU stages should usually prefer
    true tensor batching through ``SimpleScheduler(batch_compute_fn=...)``.
    """

    def __init__(self, compute_fn: Callable, *, max_concurrency: int = 8):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._fn = compute_fn
        self._max_concurrency = max(int(max_concurrency), 1)
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrency)
        self._pending: dict[str, Future] = {}
        self._aborted: set[str] = set()
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        self._running = True
        try:
            while self._running:
                self._wait_for_capacity()
                if not self._running:
                    break
                try:
                    msg = self.inbox.get(timeout=0.1)
                except _queue_mod.Empty:
                    continue
                if msg.type != "new_request":
                    continue
                with self._lock:
                    if msg.request_id in self._aborted:
                        continue
                    future = self._executor.submit(self._run_one, msg.data)
                    self._pending[msg.request_id] = future
                future.add_done_callback(
                    lambda fut, request_id=msg.request_id: self._finish(request_id, fut)
                )
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        with self._lock:
            self._aborted.add(request_id)
            future = self._pending.pop(request_id, None)
        if future is not None:
            future.cancel()

    def _wait_for_capacity(self) -> None:
        while self._running:
            with self._lock:
                if len(self._pending) < self._max_concurrency:
                    return
            time.sleep(0.001)

    def _run_one(self, payload: Any) -> Any:
        result = self._fn(payload)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return result

    def _finish(self, request_id: str, future: Future) -> None:
        with self._lock:
            self._pending.pop(request_id, None)
            aborted = request_id in self._aborted
        if aborted or future.cancelled():
            return

        try:
            result = future.result()
        except BaseException as exc:
            logger.exception(
                "ThreadedSimpleScheduler: compute_fn failed for %s", request_id
            )
            self.outbox.put(
                OutgoingMessage(request_id=request_id, type="error", data=exc)
            )
            return

        self.outbox.put(
            OutgoingMessage(request_id=request_id, type="result", data=result)
        )
