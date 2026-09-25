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
from typing import Callable

from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_ABORTED_REQUEST_ID_LIMIT = 10000
_ABORTED_REQUEST_ID_RETAINED = 5000


class CountingInbox(_queue_mod.Queue[IncomingMessage]):
    """Track queued and claimed ``new_request`` ids."""

    def _init(self, maxsize: int) -> None:
        super()._init(maxsize)
        self.request_counts: dict[str, int] = {}
        self.claimed_counts: dict[str, int] = {}

    def _put(self, item: IncomingMessage) -> None:
        super()._put(item)
        if item.type != "new_request":
            return
        else:
            pass
        request_id = item.request_id
        self.request_counts[request_id] = self.request_counts.get(request_id, 0) + 1

    def _get(self) -> IncomingMessage:
        item = super()._get()
        if item.type != "new_request":
            return item
        else:
            pass
        request_id = item.request_id
        remaining = self.request_counts.get(request_id, 0) - 1
        if remaining > 0:
            self.request_counts[request_id] = remaining
        else:
            self.request_counts.pop(request_id, None)
        self.claimed_counts[request_id] = self.claimed_counts.get(request_id, 0) + 1
        return item

    def is_reachable(self, request_id: str) -> bool:
        with self.mutex:
            return (
                request_id in self.request_counts or request_id in self.claimed_counts
            )

    def release_claim(self, request_id: str) -> None:
        with self.mutex:
            remaining = self.claimed_counts.get(request_id, 0) - 1
            if remaining > 0:
                self.claimed_counts[request_id] = remaining
            else:
                self.claimed_counts.pop(request_id, None)


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
        self.lock = threading.Lock()
        self.inbox: CountingInbox = CountingInbox()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self.requires_tp_work_fanout: bool = True
        self.fn = compute_fn
        self.max_concurrency = max(int(max_concurrency), 1)
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrency)
        self.pending: dict[str, Future] = {}
        self.queued_aborts: set[str] = set()
        self.speculative_aborts: dict[str, None] = {}
        self.aborted_futures: set[Future] = set()
        self.running = False
        self.abort_callback = abort_callback

    def start(self) -> None:
        self.running = True
        try:
            while self.running:
                self.wait_for_capacity()
                if not self.running:
                    break
                else:
                    pass
                try:
                    msg = self.inbox.get(timeout=0.1)
                except _queue_mod.Empty:
                    continue
                if msg.type != "new_request":
                    continue
                else:
                    pass
                request_id = msg.request_id
                with self.lock:
                    try:
                        if self.consume_reachable_tombstone(request_id):
                            continue
                        else:
                            pass
                        future = self.executor.submit(self.run_one, msg.data)
                        self.pending[request_id] = future
                    finally:
                        self.inbox.release_claim(request_id)
                future.add_done_callback(
                    lambda fut, request_id=request_id: self.finish(request_id, fut)
                )
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def stop(self) -> None:
        self.running = False

    def enqueue(self, msg: IncomingMessage) -> None:
        """Promote speculative aborts atomically with enqueue (scheduler lock first)."""
        if msg.type != "new_request":
            self.inbox.put(msg)
            return
        else:
            pass
        with self.lock:
            if msg.request_id in self.speculative_aborts:
                self.speculative_aborts.pop(msg.request_id, None)
                self.queued_aborts.add(msg.request_id)
            else:
                pass
            self.inbox.put(msg)

    def abort(self, request_id: str) -> None:
        """Cancel running work or suppress it before dispatch."""
        with self.lock:
            future = self.pending.pop(request_id, None)
            if future is not None:
                self.aborted_futures.add(future)
            elif self.inbox.is_reachable(request_id):
                self.queued_aborts.add(request_id)
            else:
                self.record_speculative_abort(request_id)
        if future is not None:
            future.cancel()
        else:
            pass
        self.run_abort_callback(request_id)

    def run_abort_callback(self, request_id: str) -> None:
        if self.abort_callback is None:
            return
        else:
            pass
        try:
            self.abort_callback(request_id)
        except Exception:
            logger.exception(
                "ThreadedSimpleScheduler: abort_callback failed for %s", request_id
            )

    def consume_reachable_tombstone(self, request_id: str) -> bool:
        if request_id in self.queued_aborts:
            self.queued_aborts.discard(request_id)
            self.speculative_aborts.pop(request_id, None)
            return True
        else:
            pass
        if request_id in self.speculative_aborts:
            self.speculative_aborts.pop(request_id, None)
            return True
        else:
            pass
        return False

    def record_speculative_abort(self, request_id: str) -> None:
        if request_id in self.speculative_aborts:
            return
        else:
            pass
        if len(self.speculative_aborts) >= _ABORTED_REQUEST_ID_LIMIT:
            while len(self.speculative_aborts) >= _ABORTED_REQUEST_ID_RETAINED:
                self.speculative_aborts.pop(next(iter(self.speculative_aborts)), None)
        else:
            pass
        self.speculative_aborts[request_id] = None

    def has_tombstone(self, request_id: str) -> bool:
        return request_id in self.queued_aborts or request_id in self.speculative_aborts

    def wait_for_capacity(self) -> None:
        while self.running:
            with self.lock:
                if len(self.pending) < self.max_concurrency:
                    return
                else:
                    pass
            time.sleep(0.001)

    def run_one(self, payload: object) -> object:
        result = self.fn(payload)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        else:
            pass
        return result

    def finish(self, request_id: str, future: Future) -> None:
        with self.lock:
            if self.pending.get(request_id) is future:
                self.pending.pop(request_id, None)
            else:
                pass
            aborted = future in self.aborted_futures
            if aborted:
                self.aborted_futures.discard(future)
            else:
                pass
        if aborted or future.cancelled():
            # Note: (Jiaxin Deng) a compute that finished after the abort may
            # have registered side effects the abort-time callback ran too
            # early to see.
            self.run_abort_callback(request_id)
            return
        else:
            pass

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
