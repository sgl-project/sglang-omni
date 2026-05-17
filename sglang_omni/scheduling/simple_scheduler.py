# SPDX-License-Identifier: Apache-2.0
"""SimpleScheduler — lightweight scheduler for non-AR stages.

For stages that just run a function (preprocessing, encoders, decode, code2wav).
No KV cache, no batching. Just: inbox.get() → run function → outbox.put().

Same inbox/outbox interface as OmniScheduler so Stage doesn't need branching.
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import queue as _queue_mod
import time
from typing import Any, Awaitable, Callable

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
        request_cost_fn: Callable[[Any], int] | None = None,
        max_batch_cost: int | None = None,
        max_concurrency: int = 1,
    ):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._fn = compute_fn
        self._batch_fn = batch_compute_fn
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._request_cost_fn = request_cost_fn
        self._max_batch_cost = (
            max(int(max_batch_cost), 0) if max_batch_cost is not None else None
        )
        # Note (Chenchen, Chenyang):
        # max_concurrency > 1 spawns N worker coroutines that dispatch compute_fn
        # via asyncio.to_thread so synchronous chunks do not pin the event loop.
        # Requires compute_fn to be re-entrant. Mutually exclusive with the
        # batch_compute_fn path (set one or the other, not both).
        self._max_concurrency = max(int(max_concurrency), 1)
        if self._max_concurrency > 1 and batch_compute_fn is not None:
            raise ValueError(
                "max_concurrency > 1 and batch_compute_fn are mutually exclusive"
            )
        self._running = False
        self._pending_messages: collections.deque[IncomingMessage] = collections.deque()

    def _message_cost(self, msg: IncomingMessage) -> int:
        if self._request_cost_fn is None or msg.type != "new_request":
            return 0
        return max(int(self._request_cost_fn(msg.data)), 0)

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

        batch_cost = self._message_cost(first_msg)
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
                if self._max_batch_cost is not None:
                    msg_cost = self._message_cost(msg)
                    if batch and batch_cost + msg_cost > self._max_batch_cost:
                        self._pending_messages.appendleft(msg)
                        break
                    batch_cost += msg_cost
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

    @staticmethod
    def _emit_error(
        request_id: str, error: BaseException, outbox: _queue_mod.Queue[OutgoingMessage]
    ) -> None:
        outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="error",
                data=error,
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

    @staticmethod
    async def _await_result(result: Awaitable[Any]) -> Any:
        return await result

    def _run_compute_in_thread(self, payload: Any) -> Any:
        result = self._fn(payload)
        if inspect.isawaitable(result):
            result = asyncio.run(self._await_result(result))
        return result

    def start(self) -> None:
        """Run the processing loop (blocks the thread)."""
        self._running = True
        if self._max_concurrency > 1:
            self._start_concurrent()
        else:
            self._start_serial()

    def _start_serial(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            while self._running:
                msg = self._next_message()
                if msg is None:
                    continue

                if msg.type == "new_request":
                    batch = [msg]
                    try:
                        batch = self._collect_batch(msg)
                        self._run_batch(batch, loop)
                    except Exception as exc:
                        logger.exception(
                            "SimpleScheduler: compute_fn failed for %s", msg.request_id
                        )
                        for failed_msg in batch:
                            self._emit_error(
                                failed_msg.request_id,
                                exc,
                                self.outbox,
                            )
        finally:
            loop.close()

    def _start_concurrent(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._run_workers(loop))
        finally:
            loop.close()

    async def _run_workers(self, loop: asyncio.AbstractEventLoop) -> None:
        async_inbox: asyncio.Queue[IncomingMessage] = asyncio.Queue()

        async def bridge_inbox() -> None:
            while self._running:
                try:
                    msg = await loop.run_in_executor(
                        None, lambda: self.inbox.get(timeout=0.05)
                    )
                except _queue_mod.Empty:
                    continue
                await async_inbox.put(msg)

        async def worker() -> None:
            while self._running:
                try:
                    msg = await asyncio.wait_for(async_inbox.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                if msg.type != "new_request":
                    continue
                try:
                    result = await asyncio.to_thread(
                        self._run_compute_in_thread, msg.data
                    )
                    self._emit_result(msg.request_id, result, self.outbox)
                except Exception as exc:
                    logger.exception(
                        "SimpleScheduler: compute_fn failed for %s", msg.request_id
                    )
                    self._emit_error(msg.request_id, exc, self.outbox)

        bridge_task = asyncio.create_task(bridge_inbox())
        worker_tasks = [
            asyncio.create_task(worker()) for _ in range(self._max_concurrency)
        ]
        try:
            await asyncio.gather(bridge_task, *worker_tasks)
        except asyncio.CancelledError:
            # Expected during shutdown/task cancellation; suppress intentionally.
            logger.debug("SimpleScheduler: _run_workers cancelled during shutdown")

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        pass  # Simple scheduler doesn't track request state
