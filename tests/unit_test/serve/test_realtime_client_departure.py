# SPDX-License-Identifier: Apache-2.0
"""A client that vanishes must not leave its realtime session behind."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from starlette.types import Message
from starlette.websockets import WebSocket

from sglang_omni.client.client import Client
from sglang_omni.config import RealtimeTranscriptionConfig
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime import session as session_module
from sglang_omni.serve.realtime.manager import RealtimeSessionManager
from sglang_omni.serve.realtime.task_cleanup import close_websocket
from tests.unit_test.fixtures.realtime_fakes import MODEL_NAME, HealthCoordinator

SESSION_UPDATE = json.dumps({"type": "session.update", "session": {}})
INVALID_JSON = "not json"


class FakeVAD:
    def __init__(self, config: object | None = None) -> None: ...

    def reset(self) -> None: ...


class IdleStrategy:
    def create_state(self, *, model_name: str, language: str | None) -> None:
        return None


class VanishingPeer:
    """ASGI peer whose socket fails the way uvicorn reports a closed connection."""

    def __init__(self, *, frames: list[str], failing_message_type: str) -> None:
        self.incoming: list[Message] = [
            {"type": "websocket.connect"},
            *({"type": "websocket.receive", "text": frame} for frame in frames),
        ]
        self.failing_message_type = failing_message_type
        self.sent_frame_count = 0

    async def receive(self) -> Message:
        if self.incoming:
            return self.incoming.pop(0)
        else:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def send(self, message: Message) -> None:
        if message["type"] == "websocket.send":
            self.sent_frame_count += 1
        else:
            pass
        # note (Qing Guo): The first frame reaches the client, so the failure is the reply to its last frame.
        if message["type"] == self.failing_message_type and (
            message["type"] != "websocket.send" or self.sent_frame_count > 1
        ):
            raise OSError("client vanished")
        else:
            pass


def websocket_scope(query_string: bytes = b"") -> dict[str, object]:
    return {
        "type": "websocket",
        "path": "/v1/realtime",
        "raw_path": b"/v1/realtime",
        "root_path": "",
        "scheme": "ws",
        "query_string": query_string,
        "headers": [],
        "subprotocols": [],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
        "asgi": {"version": "3.0"},
    }


@pytest.fixture
def realtime_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(session_module, "StreamingVAD", FakeVAD)
    monkeypatch.setattr("sglang_omni.audio.smart_turn.load_smart_turn", lambda: None)
    return create_app(
        Client(HealthCoordinator(True)),
        model_name=MODEL_NAME,
        enable_realtime=True,
        realtime_transcription=RealtimeTranscriptionConfig(strategy_cls=IdleStrategy),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query_string", "frame"),
    [(b"", SESSION_UPDATE), (b"intent=transcription", INVALID_JSON)],
    ids=["conversation", "transcription"],
)
async def test_client_vanishing_during_a_send_releases_its_session(
    realtime_app: FastAPI, query_string: bytes, frame: str
) -> None:
    peer = VanishingPeer(frames=[frame], failing_message_type="websocket.send")

    await realtime_app(websocket_scope(query_string), peer.receive, peer.send)

    assert peer.sent_frame_count == 2
    assert not realtime_app.state.realtime_manager.sessions


@pytest.mark.asyncio
async def test_close_websocket_tolerates_a_client_vanishing_mid_close() -> None:
    peer = VanishingPeer(frames=[], failing_message_type="websocket.close")
    websocket = WebSocket(websocket_scope(), peer.receive, peer.send)
    await websocket.accept()

    await close_websocket(websocket)


@pytest.mark.asyncio
async def test_manager_forgets_a_session_whose_teardown_failed() -> None:
    class FailingSession:
        async def teardown(self) -> None:
            raise RuntimeError("teardown failed")

    manager = RealtimeSessionManager(
        client=Client(HealthCoordinator(True)), model_name=MODEL_NAME
    )
    manager.sessions["session"] = FailingSession()

    with pytest.raises(RuntimeError, match="teardown failed"):
        await manager.close("session")

    assert not manager.sessions
