# SPDX-License-Identifier: Apache-2.0
"""Runner for coordinator and stages."""

from __future__ import annotations

import asyncio
from typing import Iterable

from sglang_omni.pipeline import Coordinator, Stage


class PipelineRunner:
    """Manage coordinator and stage lifecycles."""

    def __init__(self, coordinator: Coordinator, stages: Iterable[Stage]):
        self._coordinator = coordinator
        self._stages = list(stages)
        self._completion_task: asyncio.Task[None] | None = None
        self._stage_tasks: list[asyncio.Task[None]] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("PipelineRunner already started")

        await self._coordinator.start()
        self._completion_task = asyncio.create_task(
            self._coordinator.run_completion_loop()
        )
        self._stage_tasks = [asyncio.create_task(stage.run()) for stage in self._stages]
        self._started = True

    async def wait(self) -> None:
        if not self._started:
            raise RuntimeError("PipelineRunner not started")

        tasks = [self._completion_task, *self._stage_tasks]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        for task in done:
            task.result()

    async def stop(self) -> None:
        if not self._started:
            raise RuntimeError("PipelineRunner not started")

        await self._coordinator.shutdown_stages()
        await asyncio.gather(*self._stage_tasks)

        if self._completion_task is not None:
            self._completion_task.cancel()
            try:
                await self._completion_task
            except asyncio.CancelledError:
                pass

        await self._coordinator.stop()
        self._started = False

    async def run(self) -> None:
        await self.start()
        await self.wait()
