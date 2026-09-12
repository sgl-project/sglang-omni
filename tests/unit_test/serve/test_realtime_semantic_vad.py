# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.serve.realtime.events import SessionUpdate, TurnDetection
from sglang_omni.serve.realtime.session import RealtimeSession
from sglang_omni.serve.realtime.turn_detector import TurnDetectorBuild


class FakeDetector:
    def __init__(self) -> None:
        self.reset_calls = 0

    def process(self, pcm_bytes: bytes) -> list:
        return []

    def reset(self) -> None:
        self.reset_calls += 1


def _event(turn_detection: dict) -> SessionUpdate:
    return SessionUpdate.model_validate(
        {
            "type": "session.update",
            "session": {"turn_detection": turn_detection},
        }
    )


def _session(
    initial: FakeDetector,
    *,
    smart_turn_model=MagicMock(),
) -> RealtimeSession:
    with patch(
        "sglang_omni.serve.realtime.session.StreamingVAD",
        return_value=initial,
    ):
        session = RealtimeSession(
            MagicMock(),
            client=MagicMock(),
            model_name="model",
            session_id="session",
            smart_turn_model=smart_turn_model,
        )
    session.send = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_semantic_update_replaces_detector_and_clears_pending_audio():
    initial, semantic = FakeDetector(), FakeDetector()
    session = _session(initial)
    session.audio_buffer.buf.extend(b"\x01\x00" * 32)
    build = TurnDetectorBuild(semantic, {"type": "semantic_vad", "eagerness": "medium"})
    with patch(
        "sglang_omni.serve.realtime.session.build_turn_detector",
        return_value=build,
    ):
        await session.handle_session_update(
            _event({"type": "semantic_vad", "eagerness": "medium"})
        )

    assert session.vad is semantic and initial.reset_calls == 1
    assert session.audio_buffer.is_empty()
    assert session.buffer_origin_samples == 32
    assert session.session_object.turn_detection.type == "semantic_vad"
    assert [call.args[0]["type"] for call in session.send.await_args_list] == [
        "input_audio_buffer.cleared",
        "session.updated",
    ]


@pytest.mark.asyncio
async def test_unavailable_semantic_model_reports_effective_server_vad():
    initial, unused = FakeDetector(), FakeDetector()
    session = _session(initial, smart_turn_model=None)
    session.audio_buffer.buf.extend(b"\x01\x00")
    build = TurnDetectorBuild(
        unused,
        {"type": "server_vad", "eagerness": None},
    )
    with patch(
        "sglang_omni.serve.realtime.session.build_turn_detector",
        return_value=build,
    ):
        await session.handle_session_update(
            _event({"type": "semantic_vad", "eagerness": "high"})
        )

    assert session.vad is initial and initial.reset_calls == 0
    assert not session.audio_buffer.is_empty()
    assert session.session_object.turn_detection.type == "server_vad"
    sent = session.send.await_args.args[0]
    assert sent["type"] == "session.updated"
    assert sent["session"]["turn_detection"]["type"] == "server_vad"


@pytest.mark.asyncio
async def test_invalid_semantic_update_preserves_live_state():
    initial = FakeDetector()
    session = _session(initial)
    session.audio_buffer.buf.extend(b"\x01\x00")

    await session.handle_session_update(
        _event(
            {
                "type": "semantic_vad",
                "eagerness": "medium",
                "silence_duration_ms": 500,
            }
        )
    )
    assert session.vad is initial and initial.reset_calls == 0
    assert not session.audio_buffer.is_empty()
    assert session.session_object.turn_detection is None
    error = session.send.await_args.args[0]
    assert error["type"] == "error"
    assert error["error"]["code"] == "invalid_turn_detection"


@pytest.mark.asyncio
async def test_same_type_partial_update_preserves_semantic_type():
    initial, replacement = FakeDetector(), FakeDetector()
    session = _session(initial)
    session.session_object.turn_detection = TurnDetection(
        type="semantic_vad", eagerness="low"
    )
    build = TurnDetectorBuild(
        replacement, {"type": "semantic_vad", "eagerness": "high"}
    )
    with patch(
        "sglang_omni.serve.realtime.session.build_turn_detector",
        return_value=build,
    ) as builder:
        await session.handle_session_update(_event({"eagerness": "high"}))
    assert builder.call_args.args[0] == {
        "type": "semantic_vad",
        "eagerness": "high",
    }
    assert session.vad is replacement


@pytest.mark.asyncio
async def test_detector_build_failure_preserves_audio_and_state():
    initial = FakeDetector()
    session = _session(initial)
    session.audio_buffer.buf.extend(b"\x01\x00")
    loop = asyncio.get_running_loop()
    with (
        patch(
            "sglang_omni.serve.realtime.session.build_turn_detector",
            side_effect=RuntimeError("failed"),
        ),
        patch.object(loop, "call_exception_handler"),
    ):
        await session.handle_session_update(_event({"type": "semantic_vad"}))
    assert session.vad is initial and initial.reset_calls == 0
    assert not session.audio_buffer.is_empty()
    assert session.session_object.turn_detection is None
    assert session.send.await_args.args[0]["error"]["code"] == (
        "turn_detection_initialization_failed"
    )


@pytest.mark.asyncio
async def test_session_created_advertises_turn_detection_capabilities():
    websocket = MagicMock()
    websocket.application_state = WebSocketState.CONNECTED
    websocket.send_text = AsyncMock()
    websocket.receive = AsyncMock(return_value={"type": "websocket.disconnect"})
    initial = FakeDetector()
    with patch(
        "sglang_omni.serve.realtime.session.StreamingVAD",
        return_value=initial,
    ):
        session = RealtimeSession(
            websocket,
            client=MagicMock(),
            model_name="model",
            session_id="session",
            smart_turn_model=MagicMock(),
        )

    await session.run()

    created = json.loads(websocket.send_text.await_args.args[0])
    assert created["type"] == "session.created"
    assert created["session"]["capabilities"]["turn_detection"] == [
        "server_vad",
        "semantic_vad",
    ]
