# SPDX-License-Identifier: Apache-2.0
"""Scheduler - queue and batch collection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .encoder_runner import Request


@dataclass
class SchedulerConfig:
    max_batch_size: int = 32
    batch_timeout: float = 0.01


@dataclass
class PendingRequest:
    request: Request
    future: asyncio.Future = field(default_factory=asyncio.Future)


class Scheduler:
    """Queue + batch collection. No compute."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._pending: dict[str, PendingRequest] = {}
        self._aborted: set[str] = set()

    async def add(self, request_id: str, data) -> PendingRequest:
        if request_id in self._aborted:
            raise asyncio.CancelledError(f"Request {request_id} was aborted")
        pending = PendingRequest(Request(request_id, data))
        self._pending[request_id] = pending
        await self._queue.put(pending)
        return pending

    async def collect_batch(self) -> list[PendingRequest]:
        batch: list[PendingRequest] = []

        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            if first.request.request_id not in self._aborted:
                batch.append(first)
        except asyncio.TimeoutError:
            return []

        deadline = asyncio.get_event_loop().time() + self.config.batch_timeout
        while len(batch) < self.config.max_batch_size:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                p = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                if p.request.request_id not in self._aborted:
                    batch.append(p)
            except asyncio.TimeoutError:
                break

        return batch

    def complete(self, request_id: str, result) -> None:
        pending = self._pending.pop(request_id, None)
        if pending and not pending.future.done():
            pending.future.set_result(result)

    def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        pending = self._pending.pop(request_id, None)
        if pending and not pending.future.done():
            pending.future.cancel()
