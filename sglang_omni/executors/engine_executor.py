# SPDX-License-Identifier: Apache-2.0
"""EngineExecutor bridges worker payloads to OmniEngine."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from typing import Any

from sglang_omni.engines.base import Engine
from sglang_omni.executors.interface import Executor
from sglang_omni.proto import StagePayload


class EngineExecutor(Executor):
    """Wrap an Engine with worker-facing StagePayload I/O."""

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
        self._pending: deque[str] = deque()
        self._payloads: dict[str, StagePayload] = {}
        self._aborted: set[str] = set()

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._aborted:
            return

        self._pending.append(request_id)
        self._payloads[request_id] = payload
        engine_input = self._request_builder(payload)
        await self._engine.add_request(request_id, engine_input)

    async def get_result(self) -> StagePayload:
        while self._pending:
            request_id = self._pending.popleft()
            if request_id in self._aborted:
                self._payloads.pop(request_id, None)
                continue

            payload = self._payloads.pop(request_id, None)
            if payload is None:
                raise KeyError(f"Missing payload for request_id={request_id}")

            result = await self._engine.get_result(request_id)
            output = self._result_builder(payload, result)
            if not isinstance(output, StagePayload):
                output = StagePayload(
                    request_id=request_id,
                    request=payload.request,
                    data=output,
                )
            return output

        await asyncio.sleep(0)
        raise RuntimeError("No pending requests for get_result")

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        self._payloads.pop(request_id, None)
        try:
            self._pending.remove(request_id)
        except ValueError:
            pass
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
