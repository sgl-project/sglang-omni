# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.serve.realtime import session as session_module
from sglang_omni.serve.realtime.events import TurnDetection
from sglang_omni.serve.realtime.session import RealtimeSession
from sglang_omni.serve.realtime.vad import Emit, VADEvent


class FakeVAD:
    def __init__(self, _config: object | None = None) -> None: ...

    def reset(self) -> None: ...


class RecordingWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    async def send_text(self, _payload: str) -> None: ...


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interrupt_response", "expected_calls"),
    [(None, 1), (False, 0)],
)
async def test_speech_start_honors_interrupt_response(
    interrupt_response: bool | None,
    expected_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "StreamingVAD", FakeVAD)
    session = RealtimeSession(
        RecordingWebSocket(),  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]
    session.session_object.turn_detection = TurnDetection(
        type="server_vad",
        interrupt_response=interrupt_response,
    )
    session.active_response_has_audio = True
    session.cancel_active_response = AsyncMock()  # type: ignore[method-assign]

    await session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))

    assert session.cancel_active_response.await_count == expected_calls


@pytest.mark.asyncio
async def test_partial_turn_detection_update_preserves_interrupt_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "StreamingVAD", FakeVAD)
    session = RealtimeSession(
        RecordingWebSocket(),  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )

    await session.dispatch(
        {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "turn_detection": {
                    "type": "server_vad",
                    "interrupt_response": False,
                },
            },
        }
    )
    await session.dispatch(
        {
            "type": "session.update",
            "session": {"turn_detection": {"threshold": 0.7}},
        }
    )

    turn_detection = session.session_object.turn_detection
    assert turn_detection is not None
    assert turn_detection.threshold == 0.7
    assert turn_detection.interrupt_response is False

    session.active_response_has_audio = True
    session.cancel_active_response = AsyncMock()  # type: ignore[method-assign]
    await session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))
    session.cancel_active_response.assert_not_awaited()
