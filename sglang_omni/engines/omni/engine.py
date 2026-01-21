# SPDX-License-Identifier: Apache-2.0
"""OmniEngine - unified engine combining Scheduler and ModelRunner."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..base import Engine
from .model_runner import ModelRunner
from .scheduler import Scheduler

logger = logging.getLogger(__name__)


class OmniEngine(Engine):
    """Unified engine for all model types.

    Combines:
    - Scheduler (owns state, makes scheduling decisions)
    - ModelRunner (stateless executor)

    Execution model:
    - Busy loop: schedule() -> execute() -> update()
    - Async-friendly: add_request() and get_result() are async
    """

    def __init__(
        self,
        scheduler: Scheduler,
        model_runner: ModelRunner,
    ):
        self.scheduler = scheduler
        self.model_runner = model_runner

        self._running = False
        self._loop_task: asyncio.Task[None] | None = None

    # -------------------------------------------------------------------------
    # Engine ABC Implementation
    # -------------------------------------------------------------------------

    async def add_request(self, request_id: str, data: Any) -> None:
        """Add a request for processing."""
        self.scheduler.add_request(request_id, data)

    async def get_result(self, request_id: str) -> Any:
        """Get result for a request (blocks until ready)."""
        request = await self.scheduler.get_result(request_id)
        return request.data

    async def stream(self, request_id: str):
        """Stream per-step outputs for a request."""
        async for item in self.scheduler.stream(request_id):
            yield item

    async def abort(self, request_id: str) -> None:
        """Abort a request."""
        self.scheduler.abort_request(request_id)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the engine processing loop."""
        if self._running:
            return

        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("OmniEngine started")

    async def stop(self) -> None:
        """Stop the engine processing loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("OmniEngine stopped")

    # -------------------------------------------------------------------------
    # Processing Loop
    # -------------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                await self._step()
            except Exception:
                logger.exception("Error in OmniEngine step")
            await asyncio.sleep(0)  # Yield to other coroutines

    async def _step(self) -> bool:
        """Execute one step. Returns True if work was done."""
        # 1. Schedule
        scheduler_output = self.scheduler.schedule()

        if scheduler_output is None:
            await asyncio.sleep(0.001)  # Brief sleep when idle
            return False

        # 2. Execute (run in executor to not block event loop)
        loop = asyncio.get_running_loop()
        model_output = await loop.run_in_executor(
            None,
            self.model_runner.execute,
            scheduler_output,
        )

        # 3. Update state
        finished = self.scheduler.update(scheduler_output, model_output)

        if finished:
            for req in finished:
                logger.debug("Request %s finished", req.request_id)

        return True
