# SPDX-License-Identifier: Apache-2.0
"""FrontendExecutor for CPU-bound worker roles."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sglang_omni.executors.interface import Executor
from sglang_omni.proto import StagePayload


class FrontendExecutor(Executor):
    """Run synchronous CPU processing inside an async interface.

    Args:
        frontend: Callable that processes StagePayload (CPU-bound).
        use_thread_pool: If True, run frontend in thread pool to avoid blocking
            the event loop. Enables CPU/GPU overlap when used with pipeline.
        max_workers: Number of threads for CPU processing (only if use_thread_pool=True).
    """

    def __init__(
        self,
        frontend: Callable[[StagePayload], StagePayload | Any],
        *,
        use_thread_pool: bool = False,
        max_workers: int = 4,
    ):
        self._frontend = frontend
        self._results: asyncio.Queue[StagePayload] = asyncio.Queue()
        self._aborted: set[str] = set()
        self._use_thread_pool = use_thread_pool
        self._thread_pool: ThreadPoolExecutor | None = None
        self._max_workers = max_workers

    async def start(self) -> None:
        if self._use_thread_pool:
            self._thread_pool = ThreadPoolExecutor(max_workers=self._max_workers)

    async def stop(self) -> None:
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return

        if self._thread_pool is not None:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._thread_pool,
                self._frontend,
                payload,
            )
        else:
            result = self._frontend(payload)

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
