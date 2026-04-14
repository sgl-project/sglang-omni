# SPDX-License-Identifier: Apache-2.0
"""Runner for coordinator and stages."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Iterable

from sglang_omni.config.compiler import IpcRuntimeDir, compile_pipeline_core
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.pipeline import Coordinator, Stage


class PipelineRunner:
    """Manage coordinator and stage lifecycles."""

    def __init__(
        self,
        coordinator: Coordinator,
        stages: Iterable[Stage],
        *,
        ipc_runtime_dir: IpcRuntimeDir | None = None,
    ):
        self._coordinator = coordinator
        self._stages = list(stages)
        self._completion_task: asyncio.Task[None] | None = None
        self._stage_tasks: list[asyncio.Task[None]] = []
        self._ipc_runtime_dir = ipc_runtime_dir
        self._started = False

    @property
    def coordinator(self) -> Coordinator:
        return self._coordinator

    @property
    def stages(self) -> list[Stage]:
        return self._stages

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
            await self._cancel_completion_task()
            with suppress(Exception):
                await self._coordinator.stop()
            self._stage_tasks.clear()
            if self._ipc_runtime_dir is not None:
                self._ipc_runtime_dir.close()
                self._ipc_runtime_dir = None
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
            return

        self._started = False
        shutdown_error: Exception | None = None

        try:
            await self._coordinator.shutdown_stages()
            await asyncio.gather(*self._stage_tasks)
        except Exception as exc:
            shutdown_error = exc

        try:
            await self._cancel_completion_task()
            await self._coordinator.stop()
        finally:
            self._stage_tasks.clear()
            if self._ipc_runtime_dir is not None:
                self._ipc_runtime_dir.close()
                self._ipc_runtime_dir = None

        if shutdown_error is not None:
            raise shutdown_error

    async def run(self) -> None:
        await self.start()
        await self.wait()

    async def _cancel_completion_task(self) -> None:
        if self._completion_task is not None:
            self._completion_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._completion_task
            self._completion_task = None


def build_pipeline_runner(
    config: PipelineConfig,
) -> PipelineRunner:
    """Build a single-process pipeline runtime with isolated IPC paths."""
    coordinator, stages, ipc_runtime_dir = compile_pipeline_core(config)

    runner = PipelineRunner(
        coordinator,
        stages,
        ipc_runtime_dir=ipc_runtime_dir,
    )
    return runner
