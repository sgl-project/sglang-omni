# SPDX-License-Identifier: Apache-2.0
"""Exercise VoiceChat framing through the mounted shared WebSocket."""

import asyncio
import base64
from collections.abc import AsyncIterator
from dataclasses import replace

from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from sglang_omni.models.nemotron_voicechat.realtime import deployment
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import (
    OutputChunk,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
)
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.schema import JsonObject, JsonValue


class RecordingSessionClient:
    def __init__(self) -> None:
        self.chunks: list[TimedChunk] = []
        self.closed_sessions = 0
        self.output_queue: asyncio.Queue[OutputChunk | None] = asyncio.Queue()
        self.identity: SessionIdentity | None = None

    async def open_session(
        self,
        request: OmniRequest,
        *,
        stages: list[str],
        limits: SessionLimits,
        session_id: str,
    ) -> SessionIdentity:
        assert stages == ["perception", "thinker", "talker", "code2wav"]
        self.identity = SessionIdentity(session_id)
        self.output_queue = asyncio.Queue()
        return self.identity

    async def append_session(
        self, identity: SessionIdentity, chunk: TimedChunk
    ) -> None:
        assert identity == self.identity
        self.chunks.append(chunk)
        output = OutputChunk(
            identity,
            chunk.seq,
            chunk.seq,
            "audio",
            chunk.t_start_ms,
            chunk.duration_ms,
            {
                "pcm": b"\0\0" * (1764 if chunk.payload else 256),
                "text": "hello",
                "eos": chunk.eos,
            },
            "voicechat",
            chunk.eos,
        )
        await self.output_queue.put(output)
        await self.output_queue.put(replace(output, payload=None, kind="input_done"))

    async def session_outputs(
        self, identity: SessionIdentity
    ) -> AsyncIterator[OutputChunk]:
        while True:
            output = await self.output_queue.get()
            if output is None:
                break
            else:
                yield output

    async def close_session(self, identity: SessionIdentity) -> None:
        self.closed_sessions += 1
        await self.output_queue.put(None)


def send_event(
    websocket: WebSocketTestSession, event_type: str, **fields: JsonValue
) -> None:
    websocket.send_json({"type": event_type, "event_id": event_type, **fields})


def receive_until(websocket: WebSocketTestSession, event_type: str) -> list[JsonObject]:
    events: list[JsonObject] = []
    while not events or events[-1]["type"] != event_type:
        event = websocket.receive_json()
        assert event["type"] != "error", event
        events.append(event)
    return events


def append_audio(
    websocket: WebSocketTestSession, sequence: int, sample_count: int, start_ms: float
) -> None:
    send_event(
        websocket,
        "input_audio_buffer.append",
        audio=base64.b64encode(b"\0\0" * sample_count).decode(),
        sglang={"seq": sequence, "t_start_ms": start_ms},
    )


def test_native_websocket_partial_tail_continuation_and_close() -> None:
    client = RecordingSessionClient()
    app = create_app(
        client, model_name="nemotron-voicechat", realtime_deployment=deployment(client)
    )
    with TestClient(app).websocket_connect("/v1/realtime") as websocket:
        receive_until(websocket, "session.created")
        send_event(
            websocket, "session.update", session={"output_modalities": ["audio"]}
        )
        update = receive_until(websocket, "session.updated")[-1]
        assert update["session"]["audio"]["output"]["format"]["rate"] == 22050
        append_audio(websocket, 0, 640, 0)
        receive_until(websocket, "sglang.input_audio.accepted")
        assert client.chunks == []
        append_audio(websocket, 1, 640, 40)
        first = receive_until(websocket, "sglang.unit.done")
        assert len(client.chunks) == 1 and len(client.chunks[0].payload) == 2560
        assert any(event["type"] == "response.output_audio.delta" for event in first)
        append_audio(websocket, 2, 100, 80)
        send_event(websocket, "sglang.input_audio.end")
        events = receive_until(websocket, "sglang.input_audio.drained")
        assert events[-1]["consumed_ms"] == 86.25
        assert events[-1]["padding_ms"] == 73.75
        assert client.chunks[-1].eos and len(client.chunks[-1].payload) == 2560
        assert client.chunks[-1].duration_ms == 6.25
        assert any(event["type"] == "response.done" for event in events)
        send_event(websocket, "session.close")
        receive_until(websocket, "session.closed")
    assert client.closed_sessions == 1


def test_native_websocket_disconnect_releases_session() -> None:
    client = RecordingSessionClient()
    app = create_app(
        client, model_name="nemotron-voicechat", realtime_deployment=deployment(client)
    )
    with TestClient(app).websocket_connect("/v1/realtime") as websocket:
        receive_until(websocket, "session.created")
        send_event(websocket, "session.update", session={})
        receive_until(websocket, "session.updated")
    assert client.closed_sessions == 1


def test_native_websocket_reconnect_accepts_audio_after_session_close() -> None:
    client = RecordingSessionClient()
    application = create_app(
        client, model_name="nemotron-voicechat", realtime_deployment=deployment(client)
    )
    with TestClient(application) as http_client:
        for conversation_index in range(3):
            with http_client.websocket_connect("/v1/realtime") as websocket:
                receive_until(websocket, "session.created")
                send_event(
                    websocket,
                    "session.update",
                    session={"output_modalities": ["audio"]},
                )
                receive_until(websocket, "session.updated")
                append_audio(websocket, 0, 1280, 0)
                events = receive_until(websocket, "sglang.unit.done")
                audio_events = [
                    event
                    for event in events
                    if event["type"] == "response.output_audio.delta"
                ]
                assert len(audio_events) == 1
                assert base64.b64decode(audio_events[0]["delta"]) == bytes(1764 * 2)
                send_event(websocket, "session.close")
                receive_until(websocket, "session.closed")
                assert websocket.receive()["type"] == "websocket.close"
            assert client.closed_sessions == conversation_index + 1
