# SPDX-License-Identifier: Apache-2.0
"""SimpleScheduler — lightweight scheduler for non-AR stages.

For stages that just run a function (preprocessing, encoders, decode, code2wav).
No KV cache, no batching. Just: inbox.get() → run function → outbox.put().

Same inbox/outbox interface as OmniScheduler so Stage doesn't need branching.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import queue as _queue_mod
import time
from typing import Any, Callable

from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


class SimpleScheduler:
    """Process requests one at a time via a callable.

    Supports sync and async callables for ``new_request`` messages only.
    Streaming stages should provide a dedicated scheduler implementation
    (for example ``Code2WavScheduler``) rather than rely on SimpleScheduler.
    """

    def __init__(
        self,
        compute_fn: Callable,
        *,
        batch_compute_fn: Callable | None = None,
        max_batch_size: int = 1,
        max_batch_wait_ms: int = 0,
    ):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._fn = compute_fn
        self._batch_fn = batch_compute_fn
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._running = False
        self._pending_messages: collections.deque[IncomingMessage] = collections.deque()

    def _next_message(self) -> IncomingMessage | None:
        if self._pending_messages:
            return self._pending_messages.popleft()
        try:
            return self.inbox.get(timeout=0.1)
        except _queue_mod.Empty:
            return None

    def _collect_batch(self, first_msg: IncomingMessage) -> list[IncomingMessage]:
        batch = [first_msg]
        if self._batch_fn is None or self._max_batch_size <= 1:
            return batch

        deadline = time.monotonic() + self._max_batch_wait_s
        while len(batch) < self._max_batch_size:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = self.inbox.get(timeout=remaining)
                except _queue_mod.Empty:
                    break

            if msg.type == "new_request":
                batch.append(msg)
            else:
                self._pending_messages.append(msg)
        return batch

    @staticmethod
    def _emit_result(
        request_id: str, result: Any, outbox: _queue_mod.Queue[OutgoingMessage]
    ) -> None:
        outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=result,
            )
        )

    def _run_single(
        self, msg: IncomingMessage, loop: asyncio.AbstractEventLoop
    ) -> None:
        result = self._fn(msg.data)
        if asyncio.iscoroutine(result):
            result = loop.run_until_complete(result)
        self._emit_result(msg.request_id, result, self.outbox)

    def _run_batch(
        self,
        batch: list[IncomingMessage],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        if self._batch_fn is None or len(batch) <= 1:
            for msg in batch:
                self._run_single(msg, loop)
            return

        payloads = [msg.data for msg in batch]
        results = self._batch_fn(payloads)
        if asyncio.iscoroutine(results):
            results = loop.run_until_complete(results)
        if len(results) != len(batch):
            raise ValueError(
                f"batch_compute_fn returned {len(results)} results for {len(batch)} requests"
            )
        for msg, result in zip(batch, results):
            self._emit_result(msg.request_id, result, self.outbox)

    def start(self) -> None:
        """Run the processing loop (blocks the thread)."""
        self._running = True
        loop = asyncio.new_event_loop()
        try:
            while self._running:
                msg = self._next_message()
                if msg is None:
                    continue

                if msg.type == "new_request":
                    try:
                        batch = self._collect_batch(msg)
                        self._run_batch(batch, loop)
                    except Exception:
                        logger.exception(
                            "SimpleScheduler: compute_fn failed for %s", msg.request_id
                        )
        finally:
            loop.close()

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        pass  # Simple scheduler doesn't track request state
