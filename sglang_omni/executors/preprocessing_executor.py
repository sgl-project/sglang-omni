# SPDX-License-Identifier: Apache-2.0
"""PreprocessingExecutor for CPU-bound worker roles."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from sglang_omni.executors.interface import Executor
from sglang_omni.proto import StagePayload


class PreprocessingExecutor(Executor):
    """Run synchronous or asynchronous CPU processing inside an async interface."""

    def __init__(
        self,
        processor: Callable[
            [StagePayload], StagePayload | Any | Awaitable[StagePayload | Any]
        ],
    ):
        self._processor = processor
        self._is_async = inspect.iscoroutinefunction(processor)
        self._results: asyncio.Queue[StagePayload] = asyncio.Queue()
        self._aborted: set[str] = set()

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return

        # Run processor - async or sync
        if self._is_async:
            result = await self._processor(payload)
        else:
            # Run synchronous processor in thread pool
            result = await asyncio.to_thread(self._processor, payload)

        if not isinstance(result, StagePayload):
            result = StagePayload(
                request_id=request_id,
                request=payload.request,
                data=result,
            )

        await self._results.put(result)

    async def get_result(self) -> StagePayload:
        while True:
            result = await self._results.get()
            if result.request_id in self._aborted:
                continue
            return result

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
