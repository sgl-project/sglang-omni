# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang_omni.realtime.backend import MockResponseBackend
from sglang_omni.serve.realtime_ws_api import create_realtime_ws_router


def _make_app(
    *,
    output_modalities: tuple[str, ...] = ("text", "audio"),
    audio_mode: str = "playback",
) -> FastAPI:
    app = FastAPI()

    def backend_factory(model_name: str, max_new_tokens: int) -> MockResponseBackend:
        del max_new_tokens
        return MockResponseBackend(
            model=model_name,
            output_modalities=output_modalities,
            response_text="Mock websocket response.",
            audio_mode=audio_mode,
            inter_chunk_delay_s=0.0,
            chunk_duration_s=0.05,
            total_duration_s=0.1,
        )

    app.include_router(
        create_realtime_ws_router(
            model_name="mock-realtime-ws",
            backend_factory=backend_factory,
        )
    )
    return app


def test_realtime_ws_emits_session_created_event():
    client = TestClient(_make_app())

    with client.websocket_connect("/v1/realtime/ws?model=demo-model") as websocket:
        event = json.loads(websocket.receive_text())

    assert event["type"] == "session.created"
    assert event["model"] == "demo-model"
    assert event["transport"] == {"type": "websocket"}
    assert event["audio"]["input_encoding"] == "pcm16le"
    assert event["audio"]["output_sample_rate"] == 24000


def test_realtime_ws_supports_text_only_turns():
    client = TestClient(_make_app(output_modalities=("text",)))

    with client.websocket_connect("/v1/realtime/ws") as websocket:
        created = json.loads(websocket.receive_text())
        assert created["type"] == "session.created"

        websocket.send_text(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {"role": "user", "content": "hello there"},
                }
            )
        )
        websocket.send_text(json.dumps({"type": "response.create"}))

        seen_types: list[str] = []
        done_event = None
        for _ in range(16):
            event = json.loads(websocket.receive_text())
            seen_types.append(event["type"])
            if event["type"] == "response.done":
                done_event = event
                break

    assert "conversation.item.created" in seen_types
    assert "turn.prepared" in seen_types
    assert "response.created" in seen_types
    assert "response.output_text.delta" in seen_types
    assert done_event is not None
    assert done_event["text"] == "Mock websocket response."


def test_realtime_ws_accepts_binary_pcm_audio_in_manual_mode():
    client = TestClient(_make_app(output_modalities=("text",)))

    with client.websocket_connect("/v1/realtime/ws") as websocket:
        created = json.loads(websocket.receive_text())
        assert created["type"] == "session.created"

        websocket.send_text(
            json.dumps(
                {
                    "type": "input_audio_format",
                    "sample_rate": 16000,
                    "encoding": "pcm16le",
                }
            )
        )
        updated = json.loads(websocket.receive_text())
        assert updated["type"] == "input_audio_format.updated"
        assert updated["sample_rate"] == 16000

        websocket.send_text(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"input_mode": "manual"}},
                }
            )
        )
        mode_event = json.loads(websocket.receive_text())
        assert mode_event["type"] == "session.updated"
        assert mode_event["session"]["audio"]["input_mode"] == "manual"

        websocket.send_text(json.dumps({"type": "input_audio_buffer.start"}))
        manual_started = json.loads(websocket.receive_text())
        assert manual_started["type"] == "input_audio_buffer.manual_started"

        pcm = (np.sin(np.linspace(0.0, np.pi * 8.0, 1600)) * 12000.0).astype("<i2")
        websocket.send_bytes(pcm.tobytes())
        websocket.send_text(json.dumps({"type": "input_audio_buffer.commit"}))

        seen_types: list[str] = []
        committed = None
        done_event = None
        for _ in range(20):
            event = json.loads(websocket.receive_text())
            seen_types.append(event["type"])
            if event["type"] == "input_audio_buffer.manual_committed":
                committed = event
            if event["type"] == "response.done":
                done_event = event
                break

    assert committed is not None
    assert committed["empty"] is False
    assert committed["sample_count"] == int(pcm.size)
    assert "response.created" in seen_types
    assert "response.output_text.delta" in seen_types
    assert done_event is not None
