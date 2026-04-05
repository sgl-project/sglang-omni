from __future__ import annotations

import asyncio

from sglang_omni.executors.engine_executor import EngineExecutor
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeEngine:
    def __init__(self):
        self.stream_calls = 0

    async def add_request(self, request_id: str, data):
        del request_id, data

    async def get_result(self, request_id: str):
        del request_id
        return {}

    async def abort(self, request_id: str):
        del request_id

    async def stream(self, request_id: str):
        del request_id
        self.stream_calls += 1
        if False:
            yield {}


def test_engine_executor_stream_skips_non_stream_requests() -> None:
    async def _run() -> tuple[list[dict], int]:
        engine = _FakeEngine()
        executor = EngineExecutor(
            engine=engine,
            request_builder=lambda payload: payload.data,
            stream_builder=lambda payload, item: {"ok": True},
        )
        request_id = "req-non-stream"
        executor._payloads[request_id] = StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="hello", params={"stream": False}),
            data={},
        )
        chunks = [chunk async for chunk in executor.stream(request_id)]
        return chunks, engine.stream_calls

    chunks, stream_calls = asyncio.run(_run())
    assert chunks == []
    assert stream_calls == 0


def test_engine_executor_stream_skips_truthy_non_bool_stream_flag() -> None:
    async def _run() -> tuple[list[dict], int]:
        engine = _FakeEngine()
        executor = EngineExecutor(
            engine=engine,
            request_builder=lambda payload: payload.data,
            stream_builder=lambda payload, item: {"ok": True},
        )
        request_id = "req-truthy-stream"
        executor._payloads[request_id] = StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="hello", params={"stream": 1}),
            data={},
        )
        chunks = [chunk async for chunk in executor.stream(request_id)]
        return chunks, engine.stream_calls

    chunks, stream_calls = asyncio.run(_run())
    assert chunks == []
    assert stream_calls == 0


def test_engine_executor_stream_runs_for_stream_requests() -> None:
    class _EngineWithOneChunk(_FakeEngine):
        async def stream(self, request_id: str):
            del request_id
            self.stream_calls += 1
            yield {"audio_data": [1, 2, 3]}

    async def _run() -> tuple[list[dict], int]:
        engine = _EngineWithOneChunk()
        executor = EngineExecutor(
            engine=engine,
            request_builder=lambda payload: payload.data,
            stream_builder=lambda payload, item: item,
        )
        request_id = "req-stream"
        executor._payloads[request_id] = StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="hello", params={"stream": True}),
            data={},
        )
        chunks = [chunk async for chunk in executor.stream(request_id)]
        return chunks, engine.stream_calls

    chunks, stream_calls = asyncio.run(_run())
    assert chunks == [{"audio_data": [1, 2, 3]}]
    assert stream_calls == 1
