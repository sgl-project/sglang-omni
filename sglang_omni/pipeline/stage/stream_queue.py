# SPDX-License-Identifier: Apache-2.0
"""Per-request bounded queue for streaming between pipeline stages."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StreamItem:
    """A single item of streaming data between stages."""

    chunk_id: int
    data: Any
    from_stage: str
    metadata: dict[str, Any] | None = None


@dataclass
class StreamSignal:
    """Non-data queue event such as per-source EOS or error."""

    from_stage: str | None = None
    is_done: bool = False
    error: BaseException | None = None


class StreamQueue:
    """Manages per-request unbounded async queues for streaming between stages.

    Backpressure is applied at the sender stage / scheduler boundary before
    chunks enter this queue. The per-request queues here are unbounded so that
    ordered chunks are never dropped.

    Usage:
        sq.open("req-1")              # create queue for request
        sq.put("req-1", item)         # sender puts items
        item = await sq.get(...)      # consumer awaits next item (returns None on EOS)
        sq.close("req-1")             # cleanup
    """

    def __init__(self, max_pending: int = 16):
        self.max_pending = max_pending
        self.queues: dict[str, asyncio.Queue] = {}
        self.closed: set[str] = set()  # track closed request IDs for abort race

    def open(self, request_id: str) -> None:
        self.closed.discard(request_id)
        if request_id not in self.queues:
            self.queues[request_id] = (
                asyncio.Queue()
            )  # unbounded; backpressure at sender
        else:
            pass

    def has(self, request_id: str) -> bool:
        return request_id in self.queues

    def put(self, request_id: str, item: StreamItem) -> None:
        queue = self.queues.get(request_id)
        if queue is None:
            # The queue can disappear between an ingress guard and delivery
            # when request cleanup races an in-flight stream chunk.
            return
        else:
            pass
        queue.put_nowait(item)

    def put_done(self, request_id: str, from_stage: str | None = None) -> None:
        queue = self.queues.get(request_id)
        if queue is None:
            return
        else:
            pass
        queue.put_nowait(StreamSignal(from_stage=from_stage, is_done=True))

    def put_error(
        self, request_id: str, error: BaseException, from_stage: str | None = None
    ) -> None:
        queue = self.queues.get(request_id)
        if queue is None:
            return
        else:
            pass
        queue.put_nowait(StreamSignal(from_stage=from_stage, error=error))

    async def get(self, request_id: str) -> StreamItem | None:
        """Get next item. Returns None when done or closed (abort)."""
        queue = self.queues.get(request_id)
        if queue is None:
            if request_id in self.closed:
                return None  # queue was closed — treat as done
            else:
                pass
            raise RuntimeError(f"No queue for {request_id}")
        else:
            pass

        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            item = await queue.get()

        if isinstance(item, StreamSignal):
            if item.error:
                raise item.error
            else:
                pass
            return None
        else:
            pass
        return item

    async def get_with_source(self, request_id: str) -> StreamItem | StreamSignal:
        """Get next item or signal while preserving the upstream stage info."""
        queue = self.queues.get(request_id)
        if queue is None:
            if request_id in self.closed:
                return StreamSignal(is_done=True)  # abort signal
            else:
                pass
            raise RuntimeError(f"No queue for {request_id}")
        else:
            pass

        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            item = await queue.get()
        return item

    def close(self, request_id: str) -> None:
        q = self.queues.pop(request_id, None)
        self.closed.add(request_id)
        # Cap _closed size to prevent unbounded growth
        if len(self.closed) > 10000:
            # Remove oldest entries (set is unordered, but bulk discard is fine)
            excess = len(self.closed) - 5000
            it = iter(self.closed)
            to_remove = [next(it) for _ in range(excess)]
            self.closed -= set(to_remove)
        else:
            pass
        if q is not None:
            # Wake any blocked get() calls with a proper sentinel
            q.put_nowait(StreamSignal(is_done=True))
        else:
            pass
