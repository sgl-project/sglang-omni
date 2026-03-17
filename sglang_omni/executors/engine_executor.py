# SPDX-License-Identifier: Apache-2.0
"""EngineExecutor bridges worker payloads to OmniEngine."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from sglang_omni.engines.base import Engine
from sglang_omni.executors.interface import Executor
from sglang_omni.proto import StagePayload


class EngineExecutor(Executor):
    """Wrap an Engine with worker-facing StagePayload I/O.

    Results are returned in completion order, not submission order,
    so callers that pipeline multiple requests get the fastest result first.
    """

    def __init__(
        self,
        engine: Engine,
        request_builder: Callable[[StagePayload], Any],
        result_builder: Callable[[StagePayload, Any], StagePayload] | None = None,
        stream_builder: Callable[[StagePayload | None, Any], Any] | None = None,
    ):
        self._engine = engine
        self._request_builder = request_builder
        self._result_builder = result_builder or self._default_result_builder
        self._stream_builder = stream_builder or self._default_stream_builder
        self._stream_queue: Any | None = (
            None  # Set by compiler for stream-receiving stages
        )
        self._stream_fn: Callable | None = (
            None  # Set by compiler for stream-sending stages
        )
        self._done: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task[StagePayload]] = {}
        self._payloads: dict[str, StagePayload] = {}
        self._submit_times: dict[str, float] = {}
        self._aborted: set[str] = set()

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return

        # Pre-fetch chunks from stream queue (async) before calling sync request_builder
        if self._stream_queue is not None:
            chunks = []
            while True:
                item = await self._stream_queue.get(request_id)
                if item is None:  # EOS
                    break
                chunks.append(item)
            payload.prefetched_chunks = chunks
        else:
            payload.prefetched_chunks = None

        self._payloads[request_id] = payload
        engine_input = self._request_builder(payload)
        await self._engine.add_request(request_id, engine_input)
        self._submit_times[request_id] = time.perf_counter()

        task = asyncio.create_task(self._await_result(payload))
        self._tasks[request_id] = task
        task.add_done_callback(lambda _t: self._done.put_nowait(request_id))

    async def start(self) -> None:
        start = getattr(self._engine, "start", None)
        if callable(start):
            await start()

    async def stop(self) -> None:
        stop = getattr(self._engine, "stop", None)
        if callable(stop):
            await stop()

    def set_stream_fn(self, fn) -> None:
        """Set the streaming output callback."""
        self._stream_fn = fn

    def set_feedback_mailbox(self, mailbox: Any) -> None:
        """Attach a feedback mailbox to engines that support WAITING_FEEDBACK."""
        if hasattr(self._engine, "_feedback_mailbox"):
            self._engine._feedback_mailbox = mailbox

    async def get_result(self) -> StagePayload:
        while True:
            request_id = await self._done.get()
            if request_id in self._aborted:
                self._tasks.pop(request_id, None)
                self._payloads.pop(request_id, None)
                continue
            task = self._tasks.pop(request_id, None)
            if task is None:
                continue
            self._payloads.pop(request_id, None)
            try:
                return await task
            except Exception as e:
                e.request_id = request_id
                raise

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        self._payloads.pop(request_id, None)
        task = self._tasks.pop(request_id, None)
        if task is not None:
            task.cancel()
        await self._engine.abort(request_id)

    async def stream(self, request_id: str):
        stream_fn = getattr(self._engine, "stream", None)
        if not callable(stream_fn):
            return
        payload = self._payloads.get(request_id)
        async for item in stream_fn(request_id):
            if request_id in self._aborted:
                break
            yield self._stream_builder(payload, item)

    async def _await_result(self, payload: StagePayload) -> StagePayload:
        request_id = payload.request_id
        result = await self._engine.get_result(request_id)
        t_submit = self._submit_times.pop(request_id, None)
        engine_time_s = time.perf_counter() - t_submit if t_submit else None

        output = self._result_builder(payload, result)
        if not isinstance(output, StagePayload):
            output = StagePayload(
                request_id=request_id,
                request=payload.request,
                data=output,
            )
        if engine_time_s is not None and isinstance(output.data, dict):
            output.data["engine_time_s"] = engine_time_s
        return output

    @staticmethod
    def _default_result_builder(payload: StagePayload, result: Any) -> StagePayload:
        if not isinstance(payload.data, dict):
            payload.data = {"model_output": result}
            return payload

        payload.data["model_output"] = result
        return payload

    @staticmethod
    def _default_stream_builder(payload: StagePayload | None, item: Any) -> Any:
        if isinstance(item, dict):
            return item
        if isinstance(item, tuple) and item:
            item = item[0]
        if isinstance(item, int):
            return {"token_ids": [item]}
        return {"data": item}
