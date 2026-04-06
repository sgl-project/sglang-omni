# SPDX-License-Identifier: Apache-2.0
"""Runner for coordinator and stages."""

from __future__ import annotations

import asyncio
from typing import Iterable

from sglang_omni.config.compiler import (
    IpcNamespaceLock,
    _acquire_ipc_namespace_lock,
    compile_pipeline,
)
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.pipeline import Coordinator, Stage


class PipelineRunner:
    """Manage coordinator and stage lifecycles."""

    def __init__(
        self,
        coordinator: Coordinator,
        stages: Iterable[Stage],
        *,
        ipc_namespace_lock: IpcNamespaceLock | None = None,
    ):
        self._coordinator = coordinator
        self._stages = list(stages)
        self._completion_task: asyncio.Task[None] | None = None
        self._stage_tasks: list[asyncio.Task[None]] = []
        self._ipc_namespace_lock = ipc_namespace_lock
        self._started = False

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("PipelineRunner already started")

        try:
            await self._coordinator.start()
            self._completion_task = asyncio.create_task(
                self._coordinator.run_completion_loop()
            )
            self._stage_tasks = [
                asyncio.create_task(stage.run()) for stage in self._stages
            ]
            self._started = True
        except Exception:
            if self._ipc_namespace_lock is not None:
                self._ipc_namespace_lock.close()
                self._ipc_namespace_lock = None
            raise

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

        try:
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
        finally:
            if self._ipc_namespace_lock is not None:
                self._ipc_namespace_lock.close()
                self._ipc_namespace_lock = None

    async def run(self) -> None:
        await self.start()
        await self.wait()


def build_pipeline_runner(
    config: PipelineConfig,
) -> tuple[Coordinator, list[Stage], PipelineRunner]:
    """Build a single-process pipeline runtime with IPC namespace reservation."""
    ipc_namespace_lock = _acquire_ipc_namespace_lock(config)
    try:
        coordinator, stages = compile_pipeline(
            config,
            ipc_namespace=(
                ipc_namespace_lock.ipc_namespace if ipc_namespace_lock else None
            ),
        )
    except Exception:
        if ipc_namespace_lock is not None:
            ipc_namespace_lock.close()
        raise

    runner = PipelineRunner(
        coordinator,
        stages,
        ipc_namespace_lock=ipc_namespace_lock,
    )
    return coordinator, stages, runner
