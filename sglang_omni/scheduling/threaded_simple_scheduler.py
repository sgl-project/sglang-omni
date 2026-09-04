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

_ABORTED_REQUEST_ID_LIMIT = 10000
_ABORTED_REQUEST_ID_RETAINED = 5000


class _CountingInbox(_queue_mod.Queue):
    """Track queued and claimed ``new_request`` ids."""

    def _init(self, maxsize: int) -> None:
        super()._init(maxsize)
        self._request_counts: dict[str, int] = {}
        self._claimed_counts: dict[str, int] = {}

    def _put(self, item: IncomingMessage) -> None:
        super()._put(item)
        if item.type != "new_request":
            return
        request_id = item.request_id
        self._request_counts[request_id] = self._request_counts.get(request_id, 0) + 1

    def _get(self) -> IncomingMessage:
        item = super()._get()
        if item.type != "new_request":
            return item
        request_id = item.request_id
        remaining = self._request_counts.get(request_id, 0) - 1
        if remaining > 0:
            self._request_counts[request_id] = remaining
        else:
            self._request_counts.pop(request_id, None)
        self._claimed_counts[request_id] = self._claimed_counts.get(request_id, 0) + 1
        return item

    def is_reachable(self, request_id: str) -> bool:
        with self.mutex:
            return (
                request_id in self._request_counts or request_id in self._claimed_counts
            )

    def release_claim(self, request_id: str) -> None:
        with self.mutex:
            remaining = self._claimed_counts.get(request_id, 0) - 1
            if remaining > 0:
                self._claimed_counts[request_id] = remaining
            else:
                self._claimed_counts.pop(request_id, None)


class ThreadedSimpleScheduler:
    """Run per-request work concurrently while preserving scheduler IO shape.

    This is meant for CPU-bound or blocking simple stages that previously used
    async workers plus ``asyncio.to_thread``. GPU stages should usually prefer
    true tensor batching through ``SimpleScheduler(batch_compute_fn=...)``.

    Request ids cannot be reused while an earlier lifecycle remains in the pipeline.
    """

    def __init__(
        self,
        compute_fn: Callable,
        *,
        max_concurrency: int = 8,
        abort_callback: Callable[[str], None] | None = None,
    ):
        self._lock = threading.Lock()
        self.inbox: _CountingInbox = _CountingInbox()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self.requires_tp_work_fanout: bool = True
        self._fn = compute_fn
        self._max_concurrency = max(int(max_concurrency), 1)
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrency)
        self._pending: dict[str, Future] = {}
        self._queued_aborts: set[str] = set()
        self._speculative_aborts: dict[str, None] = {}
        self._aborted_futures: set[Future] = set()
        self._running = False
        self._abort_callback = abort_callback

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
                request_id = msg.request_id
                with self._lock:
                    try:
                        if self._consume_reachable_tombstone(request_id):
                            continue
                        future = self._executor.submit(self._run_one, msg.data)
                        self._pending[request_id] = future
                    finally:
                        self.inbox.release_claim(request_id)
                future.add_done_callback(
                    lambda fut, request_id=request_id: self._finish(request_id, fut)
                )
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def stop(self) -> None:
        self._running = False

    def enqueue(self, msg: IncomingMessage) -> None:
        """Promote speculative aborts atomically with enqueue (scheduler lock first)."""
        if msg.type != "new_request":
            self.inbox.put(msg)
            return
        with self._lock:
            if msg.request_id in self._speculative_aborts:
                self._speculative_aborts.pop(msg.request_id, None)
                self._queued_aborts.add(msg.request_id)
            self.inbox.put(msg)

    def abort(self, request_id: str) -> None:
        """Cancel running work or suppress it before dispatch."""
        with self._lock:
            future = self._pending.pop(request_id, None)
            if future is not None:
                self._aborted_futures.add(future)
            elif self.inbox.is_reachable(request_id):
                self._queued_aborts.add(request_id)
            else:
                self._record_speculative_abort(request_id)
        if future is not None:
            future.cancel()
        self._run_abort_callback(request_id)

    def _run_abort_callback(self, request_id: str) -> None:
        if self._abort_callback is None:
            return
        try:
            self._abort_callback(request_id)
        except Exception:
            logger.exception(
                "ThreadedSimpleScheduler: abort_callback failed for %s", request_id
            )

    def _consume_reachable_tombstone(self, request_id: str) -> bool:
        if request_id in self._queued_aborts:
            self._queued_aborts.discard(request_id)
            self._speculative_aborts.pop(request_id, None)
            return True
        if request_id in self._speculative_aborts:
            self._speculative_aborts.pop(request_id, None)
            return True
        return False

    def _record_speculative_abort(self, request_id: str) -> None:
        if request_id in self._speculative_aborts:
            return
        if len(self._speculative_aborts) >= _ABORTED_REQUEST_ID_LIMIT:
            while len(self._speculative_aborts) >= _ABORTED_REQUEST_ID_RETAINED:
                self._speculative_aborts.pop(next(iter(self._speculative_aborts)), None)
        self._speculative_aborts[request_id] = None

    def _has_tombstone(self, request_id: str) -> bool:
        return (
            request_id in self._queued_aborts or request_id in self._speculative_aborts
        )

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
            if self._pending.get(request_id) is future:
                self._pending.pop(request_id, None)
            aborted = future in self._aborted_futures
            if aborted:
                self._aborted_futures.discard(future)
        if aborted or future.cancelled():
            # Note: (Jiaxin Deng) a compute that finished after the abort may
            # have registered side effects the abort-time callback ran too
            # early to see.
            self._run_abort_callback(request_id)
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
