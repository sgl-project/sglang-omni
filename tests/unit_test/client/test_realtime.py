# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sglang_omni.client import Client, GenerateRequest
from sglang_omni.client.realtime import open_realtime
from sglang_omni.proto import CompleteMessage, StreamMessage


class _RealtimeStreamStub:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.session_id = "session-1"
        self.turn_id = "turn-1"
        self.input_stage = "tts_engine"
        self.updates: list[object] = []
        self.closed = False
        self._messages = iter(
            [
                CompleteMessage(
                    request_id=request_id,
                    from_stage="output",
                    success=True,
                    result={"ok": True},
                )
            ]
        )

    def __aiter__(self) -> "_RealtimeStreamStub":
        return self

    async def __anext__(self) -> CompleteMessage:
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def send_input(self, message: object) -> None:
        self.updates.append(message)

    async def aclose(self) -> None:
        self.closed = True


class _CoordinatorStub:
    def __init__(self) -> None:
        self.realtime: list[dict[str, Any]] = []
        self.realtime_stream: _RealtimeStreamStub | None = None

    async def open_realtime(
        self,
        request_id: str,
        request: object,
        *,
        session_id: str,
        turn_id: str,
        input_stage: str,
    ) -> _RealtimeStreamStub:
        self.realtime.append(
            {
                "request_id": request_id,
                "request": request,
                "session_id": session_id,
                "turn_id": turn_id,
                "input_stage": input_stage,
            }
        )
        self.realtime_stream = _RealtimeStreamStub(request_id)
        self.realtime_stream.session_id = session_id
        self.realtime_stream.turn_id = turn_id
        self.realtime_stream.input_stage = input_stage
        return self.realtime_stream


def test_client_open_realtime_returns_turn_scoped_handle() -> None:
    async def _run() -> None:
        coordinator = _CoordinatorStub()
        client = Client(coordinator)

        handle = await open_realtime(
            client,
            GenerateRequest(prompt="hello", stream=True),
            request_id="request-1",
            session_id="session-1",
            turn_id="turn-1",
            input_stage="tts_engine",
        )

        assert len(coordinator.realtime) == 1
        opened = coordinator.realtime[0]
        assert opened["request_id"] == "request-1"
        assert opened["session_id"] == "session-1"
        assert opened["turn_id"] == "turn-1"
        assert opened["input_stage"] == "tts_engine"
        assert handle.request_id == "request-1"
        assert handle.session_id == "session-1"
        assert handle.turn_id == "turn-1"
        assert handle.input_stage == "tts_engine"

        update = object()
        result = await handle.send_input(update)
        assert coordinator.realtime_stream is not None
        assert coordinator.realtime_stream.updates == [update]
        assert result is None

        chunks = [chunk async for chunk in handle]
        assert chunks[0].request_id == "request-1"
        assert chunks[0].finish_reason == "stop"

        await handle.aclose()
        assert coordinator.realtime_stream.closed is True

    asyncio.run(_run())


def test_client_open_realtime_converts_stream_messages() -> None:
    async def _run() -> None:
        coordinator = _CoordinatorStub()
        client = Client(coordinator)
        handle = await open_realtime(
            client,
            GenerateRequest(prompt="hello", stream=True),
            request_id="request-1",
            session_id="session-1",
            turn_id="turn-1",
            input_stage="tts_engine",
        )
        assert coordinator.realtime_stream is not None
        coordinator.realtime_stream._messages = iter(
            [
                StreamMessage(
                    request_id="request-1",
                    from_stage="vocoder",
                    chunk={"text": "delta"},
                    modality="text",
                    chunk_id=0,
                )
            ]
        )

        chunk = await handle.__anext__()

        assert chunk.request_id == "request-1"
        assert chunk.text == "delta"
        assert chunk.stage_name == "vocoder"
        assert chunk.modality == "text"

    asyncio.run(_run())


def test_client_open_realtime_requires_streaming_output() -> None:
    async def _run() -> None:
        client = Client(_CoordinatorStub())
        with pytest.raises(ValueError, match="request.stream=True"):
            await open_realtime(
                client,
                GenerateRequest(prompt="hello", stream=False),
                session_id="session-1",
                turn_id="turn-1",
                input_stage="tts_engine",
            )

    asyncio.run(_run())
