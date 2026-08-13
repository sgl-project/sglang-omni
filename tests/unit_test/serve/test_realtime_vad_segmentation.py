# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.serve.realtime import session as session_module
from sglang_omni.serve.realtime.events import InputAudioBufferAppend
from sglang_omni.serve.realtime.session import RealtimeSession
from sglang_omni.serve.realtime.vad import Emit, VADEvent


class RecordingWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send_text(self, payload: str) -> None:
        self.events.append(json.loads(payload))

    async def close(self) -> None:
        self.application_state = WebSocketState.DISCONNECTED
        self.client_state = WebSocketState.DISCONNECTED


class StopStartVAD:
    def __init__(self) -> None:
        self.reset_calls = 0

    def process(self, _pcm: bytes) -> list[Emit]:
        return [
            Emit(VADEvent.SPEECH_STOPPED, 512),
            Emit(VADEvent.SPEECH_STARTED, 1024),
        ]

    def reset(self) -> None:
        self.reset_calls += 1


@pytest.mark.asyncio
async def test_stop_then_start_in_one_append_retains_the_next_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vad = StopStartVAD()
    monkeypatch.setattr(session_module, "StreamingVAD", lambda _config: vad)
    websocket = RecordingWebSocket()
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]
    session.utterance_start_byte = 0
    session.utterance_item_id = "first-item"
    response_started = asyncio.Event()

    async def run_turn(_item_id: str, _payload: str) -> None:
        response_started.set()

    session.run_turn = AsyncMock(side_effect=run_turn)  # type: ignore[method-assign]
    pcm = bytes(4096)

    await session.handle_audio_append(
        InputAudioBufferAppend(
            type="input_audio_buffer.append",
            audio=base64.b64encode(pcm).decode(),
        )
    )

    assert session.audio_buffer.num_bytes == 3072
    assert session.buffer_origin_samples == 512
    assert vad.reset_calls == 0
    assert session.utterance_start_byte == 1024
    assert not session.speech_idle.is_set()
    stopped = next(
        event
        for event in websocket.events
        if event["type"] == "input_audio_buffer.speech_stopped"
    )
    started = next(
        event
        for event in websocket.events
        if event["type"] == "input_audio_buffer.speech_started"
    )
    assert stopped["audio_end_ms"] == 32
    assert started["audio_start_ms"] == 64

    await session.handle_vad_emit(Emit(VADEvent.SPEECH_STOPPED, 1536))
    await asyncio.wait_for(response_started.wait(), 1)
    await session.teardown()
