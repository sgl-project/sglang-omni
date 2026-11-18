# SPDX-License-Identifier: Apache-2.0
"""The /v1/realtime WebSocket every native realtime session runs on: framing, event numbering and the client events' shared checks."""

from __future__ import annotations

import base64
import json

import pytest
from native_server import REQUEST_TIMEOUT_SECONDS, SupervisedServer
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

PCM16_SAMPLE_COUNT = 160


def receive_event(websocket: ClientConnection) -> dict[str, object]:
    return json.loads(websocket.recv(timeout=REQUEST_TIMEOUT_SECONDS))


def append_event(pcm: bytes) -> str:
    return json.dumps(
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
    )


def test_a_session_numbers_its_events_and_ends_after_done(
    asr_server: SupervisedServer,
) -> None:
    with connect(f"ws://127.0.0.1:{asr_server.port}/v1/realtime") as websocket:
        websocket.send(
            json.dumps({"type": "session.update", "session": {"turn_detection": None}})
        )
        updated = receive_event(websocket)
        assert updated["type"] == "transcription_session.updated"
        assert updated["event_index"] == 1
        assert isinstance(updated["event_id"], str)
        websocket.send(append_event(b"\0\0" * PCM16_SAMPLE_COUNT))
        websocket.send(append_event(b"\0\0" * PCM16_SAMPLE_COUNT))
        assert websocket.ping().wait(timeout=REQUEST_TIMEOUT_SECONDS)
        # A client may split one event over continuation frames.
        websocket.send(iter(['{"type": "transcrip', 'tion.done"}']))
        completed = receive_event(websocket)
        assert completed["type"] == "transcription.completed"
        assert completed["text"] == str(2 * PCM16_SAMPLE_COUNT)
        assert completed["event_index"] == 2
        with pytest.raises(ConnectionClosed):
            websocket.recv(timeout=REQUEST_TIMEOUT_SECONDS)


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("not json", "invalid_json"),
        (json.dumps({"type": "session.update", "session": {"turn_detection": {"type": "server_vad"}}}), "unsupported_session"),
        (json.dumps({"type": "session.update"}), "unsupported_session"),
        (json.dumps({"type": "input_audio_buffer.append", "audio": "not base64!"}), "invalid_audio"),
        (append_event(b"\0\0\0"), "invalid_audio"),
        (append_event(b""), "invalid_audio"),
    ],
)  # fmt: skip
def test_an_invalid_event_is_reported_and_the_session_continues(
    asr_server: SupervisedServer, message: str, code: str
) -> None:
    with connect(f"ws://127.0.0.1:{asr_server.port}/v1/realtime") as websocket:
        websocket.send(message)
        error = receive_event(websocket)
        assert error["type"] == "error"
        assert error["error"]["code"] == code
        websocket.send(json.dumps({"type": "transcription.done"}))
        assert receive_event(websocket)["text"] == "0"
