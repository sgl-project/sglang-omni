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

    Backpressure is applied at the sender side (Worker's ``_stream_send_queue``
    with maxsize=4096 and blocking put).  The per-request queues here are
    unbounded so that ordered chunks are never dropped.

    Usage:
        sq.open("req-1")              # create queue for request
        sq.put("req-1", item)         # sender puts items
        item = await sq.get(...)      # consumer awaits next item (returns None on EOS)
        sq.close("req-1")             # cleanup
    """

    def __init__(self, max_pending: int = 16):
        self._max_pending = max_pending
        self._queues: dict[str, asyncio.Queue] = {}
        self._signals: dict[str, StreamSignal] = {}  # guaranteed delivery

    def open(self, request_id: str) -> None:
        if request_id not in self._queues:
            self._queues[request_id] = (
                asyncio.Queue()
            )  # unbounded; backpressure at sender

    def has(self, request_id: str) -> bool:
        return request_id in self._queues

    def put(self, request_id: str, item: StreamItem) -> None:
        queue = self._queues.get(request_id)
        if queue is None:
            raise KeyError(f"No queue for {request_id}")
        queue.put_nowait(item)

    def put_done(self, request_id: str, from_stage: str | None = None) -> None:
        queue = self._queues.get(request_id)
        if queue is None:
            return
        queue.put_nowait(StreamSignal(from_stage=from_stage, is_done=True))

    def put_error(
        self, request_id: str, error: BaseException, from_stage: str | None = None
    ) -> None:
        queue = self._queues.get(request_id)
        if queue is None:
            return
        queue.put_nowait(StreamSignal(from_stage=from_stage, error=error))

    async def get(self, request_id: str) -> StreamItem | None:
        """Get next item. Returns None when done, raises on error."""
        queue = self._queues.get(request_id)
        if queue is None:
            # Check deferred signals
            sig = self._signals.pop(request_id, None)
            if sig is not None:
                if sig.error:
                    raise sig.error
                return None  # done
            raise RuntimeError(f"No queue for {request_id}")

        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            # Check deferred signals before waiting
            sig = self._signals.pop(request_id, None)
            if sig is not None:
                if sig.error:
                    raise sig.error
                return None
            item = await queue.get()

        if isinstance(item, StreamSignal):
            if item.error:
                raise item.error
            return None
        return item

    async def get_with_source(self, request_id: str) -> StreamItem | StreamSignal:
        """Get next item or signal while preserving the upstream stage info."""
        queue = self._queues.get(request_id)
        if queue is None:
            sig = self._signals.pop(request_id, None)
            if sig is not None:
                return sig
            raise RuntimeError(f"No queue for {request_id}")

        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            sig = self._signals.pop(request_id, None)
            if sig is not None:
                return sig
            item = await queue.get()
        return item

    def close(self, request_id: str) -> None:
        q = self._queues.pop(request_id, None)
        self._signals.pop(request_id, None)
        if q is not None:
            # Wake any blocked get() calls with a proper sentinel
            q.put_nowait(StreamSignal(is_done=True))
