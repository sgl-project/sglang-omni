# SPDX-License-Identifier: Apache-2.0
"""Executor that runs multiple stage executors in sequence."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from sglang_omni.executors.interface import Executor
from sglang_omni.proto import StagePayload
from sglang_omni.utils import import_string


class FusedExecutor(Executor):
    """Execute a list of executors sequentially inside one worker.

    Notes:
    - Streaming is delegated to the last executor only.
    - Only adjacent stages should be fused to preserve correctness.
    """

    def __init__(self, executors: list[Executor]):
        if not executors:
            raise ValueError("FusedExecutor requires at least one executor")
        self._executors = executors
        self._pipeline_lock = asyncio.Lock()
        self._aborted: set[str] = set()

    async def start(self) -> None:
        for executor in self._executors:
            await executor.start()

    async def stop(self) -> None:
        for executor in reversed(self._executors):
            await executor.stop()

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return

        # Serialize intermediate stages so concurrent add_request calls
        # don't race on sub-executor get_result.  Only the last executor
        # (typically an engine) runs requests concurrently.
        current = payload
        async with self._pipeline_lock:
            for executor in self._executors[:-1]:
                await executor.add_request(current)
                current = await executor.get_result()

        await self._executors[-1].add_request(current)

    async def get_result(self) -> StagePayload:
        while True:
            result = await self._executors[-1].get_result()
            if result.request_id in self._aborted:
                continue
            return result

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        for executor in self._executors:
            await executor.abort(request_id)

    async def stream(self, request_id: str):
        last = self._executors[-1]
        stream_fn: Callable[[str], Any] | None = getattr(last, "stream", None)
        if not callable(stream_fn):
            return
        async for item in stream_fn(request_id):
            if request_id in self._aborted:
                break
            yield item


def create_fused_executor(*, executors: list[dict[str, Any]]) -> Executor:
    """Factory to build a fused executor from executor configs."""
    instances: list[Executor] = []
    for entry in executors:
        factory_path = entry.get("factory")
        if not isinstance(factory_path, str) or not factory_path:
            raise ValueError("Fused executor entry missing factory")
        args = entry.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("Fused executor entry args must be a dict")
        factory = import_string(factory_path)
        if not callable(factory):
            raise TypeError(f"Executor factory is not callable: {factory_path}")
        executor = factory(**args)
        if not isinstance(executor, Executor):
            raise TypeError(
                f"Executor factory {factory_path} returned {type(executor)}"
            )
        instances.append(executor)
    return FusedExecutor(instances)
