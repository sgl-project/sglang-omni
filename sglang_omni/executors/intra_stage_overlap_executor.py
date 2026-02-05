# SPDX-License-Identifier: Apache-2.0
"""IntraStageOverlapExecutor for CPU/GPU overlap within a single stage."""

from __future__ import annotations

import asyncio
import logging

from sglang_omni.executors.interface import Executor
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


class IntraStageOverlapExecutor(Executor):
    """Enable CPU/GPU overlap within a single stage (intra-stage optimization).

    Primary use case: Image/audio/video encoder stages where:
    - CPU: preprocessing (resize, normalize, etc.)
    - GPU: model forward pass (vision transformer, etc.)

    This allows Request N's GPU computation to overlap with Request N+1's
    CPU preprocessing, maximizing hardware utilization.

    Architecture:
        add_request() -> [CPU Executor] -> pump_loop -> [GPU Executor] -> get_result()
                                               |
                                          gpu_consumer (releases semaphore)

    The pump_loop runs as a background task, continuously moving completed
    CPU results to the GPU executor. A separate gpu_consumer task monitors
    GPU completion to release semaphore immediately, preventing delays.

    Args:
        cpu_executor: Executor for CPU-bound preprocessing (e.g., FrontendExecutor).
        gpu_executor: Executor for GPU-bound model inference (e.g., EngineExecutor).
        max_pending: Maximum requests allowed in GPU queue (backpressure control).
    """

    def __init__(
        self,
        cpu_executor: Executor,
        gpu_executor: Executor,
        *,
        max_pending: int = 4,
    ):
        self._cpu = cpu_executor
        self._gpu = gpu_executor
        self._max_pending = max_pending

        self._semaphore: asyncio.Semaphore | None = None
        self._pump_task: asyncio.Task[None] | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._results: asyncio.Queue[StagePayload] = asyncio.Queue()
        self._aborted: set[str] = set()
        self._running = False

    async def start(self) -> None:
        await self._cpu.start()
        await self._gpu.start()
        self._semaphore = asyncio.Semaphore(self._max_pending)
        self._running = True
        self._pump_task = asyncio.create_task(self._pump_loop())
        self._consumer_task = asyncio.create_task(self._gpu_consumer())

    async def stop(self) -> None:
        self._running = False

        # Cancel background tasks
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            self._pump_task = None

        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

        await self._gpu.stop()
        await self._cpu.stop()

    async def _pump_loop(self) -> None:
        while self._running:
            try:
                cpu_result = await self._cpu.get_result()
                request_id = cpu_result.request_id

                if request_id in self._aborted:
                    continue

                # Backpressure: wait if GPU queue is full
                assert self._semaphore is not None
                await self._semaphore.acquire()

                # Submit to GPU (non-blocking)
                await self._gpu.add_request(cpu_result)

            except asyncio.CancelledError:
                break

    async def _gpu_consumer(self) -> None:
        while self._running:
            try:
                # Wait for GPU completion
                result = await self._gpu.get_result()

                # Release semaphore immediately (key fix for defect 1)
                if self._semaphore is not None:
                    self._semaphore.release()

                # Queue result for user retrieval
                await self._results.put(result)

            except asyncio.CancelledError:
                break

    async def add_request(self, payload: StagePayload) -> None:
        if payload.request_id in self._aborted:
            return
        await self._cpu.add_request(payload)

    async def get_result(self) -> StagePayload:
        while True:
            result = await self._results.get()
            if result.request_id in self._aborted:
                continue
            return result

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        await self._cpu.abort(request_id)
        await self._gpu.abort(request_id)

    async def stream(self, request_id: str):
        stream_fn = getattr(self._gpu, "stream", None)
        if not callable(stream_fn):
            return
        async for item in stream_fn(request_id):
            if request_id in self._aborted:
                break
            yield item


def _build_executor(config: dict) -> Executor:
    from sglang_omni.config.imports import import_string

    factory_path = config.get("factory")
    if not isinstance(factory_path, str) or not factory_path:
        raise ValueError("Executor config missing 'factory'")
    args = config.get("args") or {}
    if not isinstance(args, dict):
        raise ValueError("Executor config 'args' must be a dict")

    factory = import_string(factory_path)
    if not callable(factory):
        raise TypeError(f"Executor factory is not callable: {factory_path}")
    executor = factory(**args)
    if not isinstance(executor, Executor):
        raise TypeError(f"Executor factory {factory_path} returned {type(executor)}")
    return executor


def create_intra_stage_overlap_executor(
    *,
    cpu_executor: dict,
    gpu_executor: dict,
    max_pending: int = 4,
) -> IntraStageOverlapExecutor:
    cpu = _build_executor(cpu_executor)
    gpu = _build_executor(gpu_executor)
    return IntraStageOverlapExecutor(
        cpu_executor=cpu,
        gpu_executor=gpu,
        max_pending=max_pending,
    )
