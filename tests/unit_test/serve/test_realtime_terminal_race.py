# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.serve.realtime import session as session_module
from sglang_omni.serve.realtime.session import RealtimeSession


class FakeVAD:
    def __init__(self, _config: object | None = None) -> None: ...

    def reset(self) -> None: ...


class BlockingWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.terminal_started = asyncio.Event()
        self.release_terminal = asyncio.Event()

    async def send_text(self, payload: str) -> None:
        event = json.loads(payload)
        if event["type"] == "response.text.done":
            self.terminal_started.set()
            await self.release_terminal.wait()
        self.events.append(event)


class StreamingClient:
    def __init__(self) -> None:
        self.aborted: list[str] = []

    async def completion_stream(
        self, _request: Any, *, request_id: str, audio_format: str = "wav"
    ) -> AsyncIterator[CompletionStreamChunk]:
        del request_id, audio_format
        yield CompletionStreamChunk(
            request_id="request",
            modality="text",
            text="answer",
        )
        yield CompletionStreamChunk(
            request_id="request",
            modality="audio",
            audio_b64="AQI=",
        )

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


@pytest.mark.asyncio
async def test_completion_wins_if_cancel_arrives_during_terminal_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "StreamingVAD", FakeVAD)
    websocket = BlockingWebSocket()
    client = StreamingClient()
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]
    response = asyncio.create_task(session.run_response("audio"))
    await websocket.terminal_started.wait()

    await session.cancel_active_response("turn_detected")
    websocket.release_terminal.set()

    assert (await response).text == "answer"
    assert client.aborted == []
    done = next(
        event["response"]
        for event in websocket.events
        if event["type"] == "response.done"
    )
    assert done["status"] == "completed"
