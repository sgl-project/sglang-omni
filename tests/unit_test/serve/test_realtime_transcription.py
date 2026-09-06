# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client import CompletionResult, GenerateRequest
from sglang_omni.config import AudioChunkingConfig, RealtimeTranscriptionConfig
from sglang_omni.serve.realtime import transcription_session as session_module
from sglang_omni.serve.realtime.transcription_session import (
    RealtimeTranscriptionSession,
)
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


class FakeVAD:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.config = VADConfig()

    def process(self, _pcm: bytes) -> list[Any]:
        return []

    def reset(self) -> None:
        self.reset_calls += 1


class StartOnNextAppendVAD(FakeVAD):
    def __init__(self) -> None:
        super().__init__()
        self.should_start = True

    def process(self, _pcm: bytes) -> list[Emit]:
        if not self.should_start:
            return []
        self.should_start = False
        return [Emit(VADEvent.SPEECH_STARTED, 0)]

    def reset(self) -> None:
        super().reset()
        self.should_start = True


class FakeStrategy:
    def create_state(self, **settings: Any) -> object:
        return settings

    def build_decode_request(self, **_: Any) -> GenerateRequest:
        return GenerateRequest(prompt="audio", stream=False)

    def update_hypothesis(
        self,
        *,
        generated_text: str,
        language: str | None,
        state: object,
    ) -> str:
        assert isinstance(state, dict)
        state["language"] = language
        return generated_text


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []
        self.aborted: list[str] = []

    async def completion(
        self, _request: GenerateRequest, *, request_id: str
    ) -> CompletionResult:
        self.calls.append(request_id)
        text = self.outputs.pop(0) if self.outputs else f"text-{len(self.calls)}"
        return CompletionResult(request_id=request_id, text=text, language="English")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class BlockingClient(FakeClient):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def completion(
        self, _request: GenerateRequest, *, request_id: str
    ) -> CompletionResult:
        self.calls.append(request_id)
        if len(self.calls) == 1:
            self.started.set()
            await self.release.wait()
        return CompletionResult(
            request_id=request_id,
            text=f"text-{len(self.calls)}",
            language="English",
        )


class BlockingSecondClient(FakeClient):
    def __init__(self) -> None:
        super().__init__([])
        self.second_started = asyncio.Event()

    async def completion(
        self, _request: GenerateRequest, *, request_id: str
    ) -> CompletionResult:
        self.calls.append(request_id)
        if len(self.calls) == 2:
            self.second_started.set()
            await asyncio.Event().wait()
        return CompletionResult(
            request_id=request_id,
            text=f"text-{len(self.calls)}",
            language="English",
        )


def _pcm(seconds: float, amplitude: int = 1000) -> bytes:
    samples = int(16000 * seconds)
    return amplitude.to_bytes(2, "little", signed=True) * samples


def _audio_event(pcm: bytes) -> dict[str, Any]:
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode(),
    }


async def _session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outputs: list[str] | None = None,
    max_audio_clip_s: float = 60.0,
) -> tuple[RealtimeTranscriptionSession, RecordingWebSocket, FakeClient]:
    monkeypatch.setattr(session_module, "StreamingVAD", lambda _config: FakeVAD())
    websocket = RecordingWebSocket()
    client = FakeClient(outputs or [])
    session = RealtimeTranscriptionSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-asr",
        capability=RealtimeTranscriptionConfig(
            strategy_cls=FakeStrategy,
            decode_interval_ms=2000,
        ),
        audio_chunking=AudioChunkingConfig(max_audio_clip_s=max_audio_clip_s),
        strategy=FakeStrategy(),
        session_id="sess-test",
    )
    await session.dispatch(
        {
            "type": "session.update",
            "session": {"turn_detection": None},
        }
    )
    return session, websocket, client


@pytest.mark.asyncio
async def test_partial_is_replaced_by_one_final_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await _session(
        monkeypatch, outputs=["hello wor", "hello world"]
    )
    await session.dispatch(_audio_event(_pcm(2.0)))
    for _ in range(10):
        await asyncio.sleep(0)
        if any(event["type"] == "transcription.segment" for event in websocket.events):
            break

    await session.dispatch({"type": "input_audio_buffer.commit"})
    await session.dispatch({"type": "transcription.done"})

    hypotheses = [
        event for event in websocket.events if event["type"] == "transcription.segment"
    ]
    assert [(event["text"], event["is_final"]) for event in hypotheses] == [
        ("hello wor", False),
        ("hello world", True),
    ]
    indexes = [event["event_index"] for event in websocket.events]
    assert indexes == sorted(indexes) and len(indexes) == len(set(indexes))
    completed = websocket.events[-1]
    assert completed["type"] == "transcription.completed"
    assert completed["text"] == "hello world"
    assert session._decode_worker_task.done()


@pytest.mark.asyncio
async def test_audio_during_decode_coalesces_to_one_followup_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _websocket, _client = await _session(monkeypatch)
    client = BlockingClient()
    session.client = client  # type: ignore[assignment]

    await session.dispatch(_audio_event(_pcm(2.0)))
    await client.started.wait()
    await session.dispatch(_audio_event(_pcm(2.0)))
    client.release.set()
    for _ in range(20):
        await asyncio.sleep(0)
        if len(client.calls) == 2:
            break

    assert len(client.calls) == 2
    await session.teardown()


@pytest.mark.asyncio
async def test_teardown_aborts_inflight_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await _session(monkeypatch)
    client = BlockingClient()
    session.client = client  # type: ignore[assignment]

    await session.dispatch(_audio_event(_pcm(2.0)))
    await client.started.wait()
    await session.teardown()

    assert client.aborted == [client.calls[0]]
    assert session._decode_worker_task.done()
    assert websocket.client_state == WebSocketState.DISCONNECTED


@pytest.mark.asyncio
async def test_clear_aborts_active_segment_and_session_remains_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vad = StartOnNextAppendVAD()
    monkeypatch.setattr(session_module, "StreamingVAD", lambda _config: vad)
    websocket = RecordingWebSocket()
    client = BlockingSecondClient()
    strategy = FakeStrategy()
    session = RealtimeTranscriptionSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-asr",
        capability=RealtimeTranscriptionConfig(
            strategy_cls=FakeStrategy,
            decode_interval_ms=2000,
        ),
        audio_chunking=AudioChunkingConfig(),
        strategy=strategy,
        session_id="sess-clear",
    )

    await session.dispatch(_audio_event(_pcm(2.0)))
    for _ in range(10):
        await asyncio.sleep(0)
        if any(event["type"] == "transcription.segment" for event in websocket.events):
            break
    assert session.active_segment is not None
    cleared_state = session.active_segment.strategy_state

    await session.dispatch(_audio_event(_pcm(2.0)))
    await client.second_started.wait()
    await session.dispatch({"type": "input_audio_buffer.clear"})

    assert client.aborted == [client.calls[1]]
    assert session.audio_buffer.is_empty()
    assert session.active_segment is None
    assert vad.reset_calls == 1
    assert not session._decode_worker_task.done()
    assert not session._pending_finals
    assert not session._final_waiters
    assert websocket.events[-1]["type"] == "input_audio_buffer.cleared"
    assert not any(
        event["type"] == "transcription.segment"
        and event["segment_id"] == 0
        and event["is_final"]
        for event in websocket.events
    )

    await session.dispatch(_audio_event(_pcm(1.0)))
    assert session.active_segment is not None
    assert session.active_segment.strategy_state is not cleared_state
    await session.dispatch({"type": "input_audio_buffer.commit"})
    await session.dispatch({"type": "transcription.done"})

    finals = [
        event
        for event in websocket.events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert [(event["segment_id"], event["text"]) for event in finals] == [(1, "text-3")]
    assert websocket.events[-1]["type"] == "transcription.completed"
    assert websocket.events[-1]["text"] == "text-3"


@pytest.mark.asyncio
async def test_silent_final_does_not_reach_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, client = await _session(monkeypatch)

    await session.dispatch(_audio_event(_pcm(0.5, amplitude=0)))
    await session.dispatch({"type": "input_audio_buffer.commit"})
    await session.dispatch({"type": "transcription.done"})

    assert client.calls == []
    assert websocket.events[-2]["type"] == "transcription.segment"
    assert websocket.events[-2]["text"] == ""
    assert websocket.events[-2]["is_final"] is True
    assert websocket.events[-1]["type"] == "transcription.completed"
    assert websocket.events[-1]["text"] == ""


@pytest.mark.asyncio
async def test_hard_limit_finalizes_in_audio_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _client = await _session(monkeypatch, max_audio_clip_s=1.0)
    await session.dispatch(_audio_event(_pcm(2.25)))
    await session.dispatch({"type": "transcription.done"})

    finals = [
        event
        for event in websocket.events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert [event["segment_id"] for event in finals] == [0, 1, 2]
    assert websocket.events[-1]["type"] == "transcription.completed"
    assert websocket.events[-1]["text"] == "text-1 text-2 text-3"


@pytest.mark.asyncio
async def test_vad_idle_silence_keeps_buffer_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "StreamingVAD", lambda _config: FakeVAD())
    websocket = RecordingWebSocket()
    session = RealtimeTranscriptionSession(
        websocket,  # type: ignore[arg-type]
        client=FakeClient([]),  # type: ignore[arg-type]
        model_name="qwen3-asr",
        capability=RealtimeTranscriptionConfig(
            strategy_cls=FakeStrategy,
            decode_interval_ms=2000,
        ),
        audio_chunking=AudioChunkingConfig(max_audio_clip_s=1.0),
        strategy=FakeStrategy(),
        session_id="sess-idle",
    )

    # Server VAD never reports speech, so no segment starts and _queue_final
    # never drains the buffer. Streaming past max_audio_clip_s + 4s of audio
    # must still not raise BufferOverflow.
    for _ in range(8):
        await session.dispatch(_audio_event(_pcm(1.0, amplitude=0)))

    assert session.active_segment is None
    assert not [event for event in websocket.events if event["type"] == "error"]
    assert session.audio_buffer.num_bytes < session.audio_buffer.max_bytes
    await session.teardown()
