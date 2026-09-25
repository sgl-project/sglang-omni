# SPDX-License-Identifier: Apache-2.0
"""Exercise VoiceChat framing through the mounted shared WebSocket."""

import asyncio
import base64
from dataclasses import replace

from fastapi.testclient import TestClient

from sglang_omni.models.nemotron_voicechat.realtime import deployment
from sglang_omni.proto.session import OutputChunk, SessionIdentity
from sglang_omni.serve.openai_api import create_app


class Client:
    def __init__(self):
        self.chunks = []
        self.closed = 0
        self.queue = asyncio.Queue()
        self.identity = None

    async def open_session(self, request, *, stages, limits, session_id):
        assert stages == ["perception", "thinker", "talker", "code2wav"]
        self.identity = SessionIdentity(session_id)
        return self.identity

    async def append_session(self, identity, chunk):
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
        await self.queue.put(output)
        await self.queue.put(replace(output, payload=None, kind="input_done"))

    async def session_outputs(self, identity):
        while True:
            output = await self.queue.get()
            if output is None:
                break
            yield output

    async def close_session(self, identity):
        self.closed += 1
        await self.queue.put(None)


def send(websocket, event_type, **fields):
    websocket.send_json({"type": event_type, "event_id": event_type, **fields})


def until(websocket, event_type):
    events = []
    while not events or events[-1]["type"] != event_type:
        event = websocket.receive_json()
        assert event["type"] != "error", event
        events.append(event)
    return events


def append(websocket, sequence, samples, start):
    send(
        websocket,
        "input_audio_buffer.append",
        audio=base64.b64encode(b"\0\0" * samples).decode(),
        sglang={"seq": sequence, "t_start_ms": start},
    )


def test_native_websocket_partial_tail_continuation_and_close():
    client = Client()
    app = create_app(
        client, model_name="nemotron-voicechat", realtime_deployment=deployment(client)
    )
    with TestClient(app).websocket_connect("/v1/realtime") as websocket:
        until(websocket, "session.created")
        send(websocket, "session.update", session={"output_modalities": ["audio"]})
        update = until(websocket, "session.updated")[-1]
        assert update["session"]["audio"]["output"]["format"]["rate"] == 22050
        append(websocket, 0, 640, 0)
        until(websocket, "sglang.input_audio.accepted")
        assert client.chunks == []
        append(websocket, 1, 640, 40)
        first = until(websocket, "sglang.unit.done")
        assert len(client.chunks) == 1 and len(client.chunks[0].payload) == 2560
        assert any(event["type"] == "response.output_audio.delta" for event in first)
        append(websocket, 2, 100, 80)
        send(websocket, "sglang.input_audio.end")
        events = until(websocket, "sglang.input_audio.drained")
        assert events[-1]["consumed_ms"] == 86.25
        assert events[-1]["padding_ms"] == 73.75
        assert client.chunks[-1].eos and len(client.chunks[-1].payload) == 2560
        assert client.chunks[-1].duration_ms == 6.25
        assert any(event["type"] == "response.done" for event in events)
        send(websocket, "session.close")
        until(websocket, "session.closed")
    assert client.closed == 1


def test_native_websocket_disconnect_releases_session():
    client = Client()
    app = create_app(
        client, model_name="nemotron-voicechat", realtime_deployment=deployment(client)
    )
    with TestClient(app).websocket_connect("/v1/realtime") as websocket:
        until(websocket, "session.created")
        send(websocket, "session.update", session={})
        until(websocket, "session.updated")
    assert client.closed == 1
