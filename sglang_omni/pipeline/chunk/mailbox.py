# SPDX-License-Identifier: Apache-2.0
"""Per-request bounded queue for streaming chunks."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ChunkItem:
    """A single chunk of streaming data."""

    chunk_id: int
    tensor: torch.Tensor
    metadata: dict[str, Any] | None = None
    from_stage: str | None = None


@dataclass
class ChunkSignal:
    """Non-tensor mailbox event such as per-source EOS or error."""

    from_stage: str | None = None
    is_chunks_done: bool = False
    error: BaseException | None = None


_SENTINEL_DONE = object()


class ChunkMailbox:
    """Manages per-request bounded async queues for streaming chunks.

    Usage:
        mb.open("req-1")          # create queue for request
        mb.put("req-1", item)     # sender puts chunks (sync, raises on full)
        item = await mb.get(...)  # consumer awaits next chunk (returns None on EOS)
        mb.close("req-1")         # cleanup
    """

    def __init__(self, max_pending: int = 16):
        self._max_pending = max_pending
        self._queues: dict[str, asyncio.Queue] = {}

    def open(self, request_id: str) -> None:
        if request_id not in self._queues:
            self._queues[request_id] = asyncio.Queue(maxsize=self._max_pending)

    def has(self, request_id: str) -> bool:
        return request_id in self._queues

    def put(self, request_id: str, item: ChunkItem) -> None:
        queue = self._queues.get(request_id)
        if queue is None:
            raise KeyError(f"No mailbox for {request_id}")
        if queue.full():
            raise RuntimeError(f"Chunk mailbox full for {request_id}")
        queue.put_nowait(item)

    def put_chunks_done(self, request_id: str, from_stage: str | None = None) -> None:
        queue = self._queues.get(request_id)
        if queue is not None:
            queue.put_nowait(
                ChunkSignal(from_stage=from_stage, is_chunks_done=True)
            )

    def put_error(
        self, request_id: str, error: BaseException, from_stage: str | None = None
    ) -> None:
        queue = self._queues.get(request_id)
        if queue is not None:
            queue.put_nowait(ChunkSignal(from_stage=from_stage, error=error))

    async def get(self, request_id: str) -> ChunkItem | None:
        """Get next chunk. Returns None when done, raises on error."""
        item = await self.get_with_source(request_id)
        if isinstance(item, ChunkSignal):
            if item.is_chunks_done:
                return None
            if item.error is not None:
                raise item.error
        return item

    async def get_with_source(self, request_id: str) -> ChunkItem | ChunkSignal:
        """Get next chunk or signal while preserving the upstream stage."""
        queue = self._queues.get(request_id)
        if queue is None:
            raise KeyError(f"No mailbox for {request_id}")
        item = await queue.get()
        if item is _SENTINEL_DONE:
            return ChunkSignal(is_chunks_done=True)
        if isinstance(item, BaseException):
            return ChunkSignal(error=item)
        return item

    def close(self, request_id: str) -> None:
        self._queues.pop(request_id, None)
