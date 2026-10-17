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
from sglang_omni.serve.realtime.vad import Emit, VADConfig, VADEvent


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

    def process(self, pcm: bytes) -> list[Emit]:
        return [
            Emit(VADEvent.SPEECH_STOPPED, 512),
            Emit(VADEvent.SPEECH_STARTED, 1024),
        ]

    def reset(self) -> None:
        self.reset_calls += 1


class SilentVAD:
    config = VADConfig()

    def process(self, pcm: bytes) -> list[Emit]:
        return []

    def reset(self) -> None:
        pass


class EndlessSpeechVAD(SilentVAD):
    def __init__(self) -> None:
        self.started = False

    def process(self, pcm: bytes) -> list[Emit]:
        if self.started:
            return []
        else:
            self.started = True
            return [Emit(VADEvent.SPEECH_STARTED, 0)]


@pytest.mark.asyncio
@pytest.mark.parametrize("speaking", [False, True])
async def test_seventy_seconds_without_a_committed_turn_keep_the_session(
    monkeypatch: pytest.MonkeyPatch, speaking: bool
) -> None:
    vad = EndlessSpeechVAD() if speaking else SilentVAD()
    monkeypatch.setattr(session_module, "StreamingVAD", lambda config: vad)
    websocket = RecordingWebSocket()
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    committed_bytes: list[int] = []

    async def record_turn(turn: tuple[str, str]) -> None:
        payload = turn[1]
        committed_bytes.append(len(base64.b64decode(payload.split(",", 1)[1])) - 44)

    session.response_queue.put = record_turn  # type: ignore[method-assign]
    chunk = base64.b64encode(bytes(3200)).decode()

    for _ in range(700):
        await session.handle_audio_append(
            InputAudioBufferAppend(type="input_audio_buffer.append", audio=chunk)
        )

    errors = [event["error"] for event in websocket.events if event["type"] == "error"]
    assert errors == []
    if speaking:
        # A turn as long as the buffer is committed whole and continues as a new turn.
        assert committed_bytes == [600 * 3200]
        types = [event["type"] for event in websocket.events]
        assert types[types.index("input_audio_buffer.committed") + 1] == (
            "input_audio_buffer.speech_started"
        )
        assert session.audio_buffer.num_bytes == 100 * 3200
    else:
        assert committed_bytes == []
        assert session.audio_buffer.num_bytes <= 2 * (300 * 16 + 2 * 512)
    # One append larger than the whole buffer still fails closed.
    await session.handle_audio_append(
        InputAudioBufferAppend(
            type="input_audio_buffer.append",
            audio=base64.b64encode(bytes(1920002)).decode(),
        )
    )
    errors = [event["error"] for event in websocket.events if event["type"] == "error"]
    assert [error["code"] for error in errors] == ["audio_buffer_overflow"]
    assert websocket.application_state is WebSocketState.CONNECTED
    await session.teardown()


@pytest.mark.asyncio
async def test_stop_then_start_in_one_append_retains_the_next_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vad = StopStartVAD()
    monkeypatch.setattr(session_module, "StreamingVAD", lambda config: vad)
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

    async def run_turn(_item_id: str, payload: str) -> None:
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
