# SPDX-License-Identifier: Apache-2.0
"""Encoder engine - combines runner and scheduler."""

from __future__ import annotations

import asyncio
import logging

from ..base import Engine
from .encoder_runner import EncoderRunner
from .scheduler import PendingRequest, Scheduler

logger = logging.getLogger(__name__)


class EncoderEngine(Engine):
    """Combines EncoderRunner + Scheduler, implements Engine interface."""

    def __init__(self, runner: EncoderRunner, scheduler: Scheduler):
        self.runner = runner
        self.scheduler = scheduler
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.runner.load()
        self._task = asyncio.create_task(self._loop())
        logger.info("EncoderEngine started on %s", self.runner.device)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        logger.info("EncoderEngine stopped")

    async def _loop(self) -> None:
        while self._running:
            batch = await self.scheduler.collect_batch()
            if not batch:
                continue
            await self._process(batch)

    async def _process(self, batch: list[PendingRequest]) -> None:
        requests = [p.request for p in batch]
        batch_input = self.runner.prepare(requests)

        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, self.runner.forward, batch_input)

        results = self.runner.unbatch(requests, output)
        for p, result in zip(batch, results):
            self.scheduler.complete(p.request.request_id, result)

    async def add_request(self, request_id: str, data) -> None:
        await self.scheduler.add(request_id, data)

    async def get_result(self, request_id: str):
        pending = self.scheduler._pending.get(request_id)
        if pending is None:
            raise ValueError(f"Unknown request: {request_id}")
        return await pending.future

    async def abort(self, request_id: str) -> None:
        self.scheduler.abort(request_id)
