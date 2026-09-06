# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.serve.realtime import session as session_module
from sglang_omni.serve.realtime.events import ResponseCancel
from sglang_omni.serve.realtime.session import RealtimeSession
from sglang_omni.serve.realtime.vad import Emit, VADEvent


class FakeVAD:
    def __init__(self, _config: object | None = None) -> None: ...

    def reset(self) -> None: ...


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


class BlockingSpeechStartedWebSocket(RecordingWebSocket):
    def __init__(self) -> None:
        super().__init__()
        self.speech_started_send = asyncio.Event()
        self.release_speech_started = asyncio.Event()

    async def send_text(self, payload: str) -> None:
        event = json.loads(payload)
        if event["type"] == "input_audio_buffer.speech_started":
            self.speech_started_send.set()
            await self.release_speech_started.wait()
        self.events.append(event)


class FailingSpeechStartedWebSocket(RecordingWebSocket):
    async def send_text(self, payload: str) -> None:
        event = json.loads(payload)
        if event["type"] == "input_audio_buffer.speech_started":
            raise RuntimeError("speech_started send failed")
        self.events.append(event)


class BlockingSecondSpeechStartedWebSocket(RecordingWebSocket):
    def __init__(self) -> None:
        super().__init__()
        self.speech_started_count = 0
        self.second_speech_started_send = asyncio.Event()
        self.release_second_speech_started = asyncio.Event()

    async def send_text(self, payload: str) -> None:
        event = json.loads(payload)
        if event["type"] == "input_audio_buffer.speech_started":
            self.speech_started_count += 1
            if self.speech_started_count == 2:
                self.second_speech_started_send.set()
                await self.release_second_speech_started.wait()
        self.events.append(event)


StreamFactory = Callable[[], AsyncIterator[CompletionStreamChunk]]


class ScriptedClient:
    def __init__(self, streams: list[StreamFactory]) -> None:
        self.streams = list(streams)
        self.requests: list[str] = []
        self.aborted: list[str] = []
        self.abort_hook: Callable[[], Any] | None = None

    async def completion_stream(
        self, _request: Any, *, request_id: str, audio_format: str = "wav"
    ) -> AsyncIterator[CompletionStreamChunk]:
        del audio_format
        self.requests.append(request_id)
        async for chunk in self.streams.pop(0)():
            yield chunk

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)
        if self.abort_hook is not None:
            await self.abort_hook()


def _chunk(
    *, text: str = "", modality: str = "text", finish_reason: str | None = None
) -> CompletionStreamChunk:
    return CompletionStreamChunk(
        request_id="request",
        modality=modality,
        text=text,
        audio_b64="AQI=" if modality == "audio" else None,
        finish_reason=finish_reason,
    )


def _session(
    monkeypatch: pytest.MonkeyPatch, streams: list[StreamFactory]
) -> tuple[RealtimeSession, RecordingWebSocket, ScriptedClient]:
    monkeypatch.setattr(session_module, "StreamingVAD", FakeVAD)
    websocket = RecordingWebSocket()
    client = ScriptedClient(streams)
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]
    return session, websocket, client


@pytest.mark.asyncio
async def test_abort_finishes_before_transcription_and_omits_cancelled_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_started = asyncio.Event()
    abort_started = asyncio.Event()
    allow_abort = asyncio.Event()
    abort_done = asyncio.Event()
    transcription_started = asyncio.Event()

    async def response() -> AsyncIterator[CompletionStreamChunk]:
        response_started.set()
        yield _chunk(text="partial")
        yield _chunk(modality="audio")
        await abort_done.wait()
        raise RuntimeError("aborted stream")

    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        assert abort_done.is_set()
        transcription_started.set()
        yield _chunk(text="original question")
        yield _chunk(finish_reason="stop")

    session, websocket, client = _session(monkeypatch, [response, transcription])

    async def abort() -> None:
        abort_started.set()
        await allow_abort.wait()
        abort_done.set()

    client.abort_hook = abort
    turn = asyncio.create_task(session.run_turn("item-user", "audio"))
    await response_started.wait()
    cancel = asyncio.create_task(
        session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))
    )
    await abort_started.wait()
    await asyncio.sleep(0)
    dispatch_returned_before_abort = cancel.done()
    assert not transcription_started.is_set()
    allow_abort.set()
    await cancel
    await turn

    assert dispatch_returned_before_abort
    assert [item.text for item in session.conversation] == ["original question"]
    types = [event["type"] for event in websocket.events]
    assert types.count("response.text.done") == 1
    assert types.count("response.audio.done") == 1
    assert types.count("response.done") == 1
    response = next(
        e["response"] for e in websocket.events if e["type"] == "response.done"
    )
    assert response["status_details"]["reason"] == "turn_detected"


@pytest.mark.asyncio
async def test_barge_in_aborts_before_speech_started_send_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "StreamingVAD", FakeVAD)
    response_started = asyncio.Event()
    abort_started = asyncio.Event()
    abort_done = asyncio.Event()

    async def response() -> AsyncIterator[CompletionStreamChunk]:
        response_started.set()
        yield _chunk(text="partial")
        yield _chunk(modality="audio")
        await abort_done.wait()
        raise RuntimeError("aborted stream")

    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text="question")
        yield _chunk(finish_reason="stop")

    websocket = BlockingSpeechStartedWebSocket()
    client = ScriptedClient([response, transcription])
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]

    async def abort() -> None:
        abort_started.set()
        abort_done.set()

    client.abort_hook = abort
    turn = asyncio.create_task(session.run_turn("item-user", "audio"))
    await response_started.wait()
    speech_started = asyncio.create_task(
        session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))
    )
    await websocket.speech_started_send.wait()
    await asyncio.wait_for(abort_started.wait(), 0.1)
    await asyncio.sleep(0)

    assert not speech_started.done()
    assert not any(event["type"] == "response.done" for event in websocket.events)

    websocket.release_speech_started.set()
    await speech_started
    await turn

    types = [event["type"] for event in websocket.events]
    assert types.index("input_audio_buffer.speech_started") < types.index(
        "response.done"
    )


@pytest.mark.asyncio
async def test_speech_started_send_failure_does_not_deadlock_cancelled_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "StreamingVAD", FakeVAD)
    response_started = asyncio.Event()

    async def response() -> AsyncIterator[CompletionStreamChunk]:
        response_started.set()
        yield _chunk(text="partial")
        yield _chunk(modality="audio")
        await asyncio.Event().wait()

    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text="question")
        yield _chunk(finish_reason="stop")

    websocket = FailingSpeechStartedWebSocket()
    client = ScriptedClient([response, transcription])
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]
    turn = asyncio.create_task(session.run_turn("item-user", "audio"))
    await response_started.wait()

    with pytest.raises(RuntimeError, match="speech_started send failed"):
        await session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))

    await asyncio.wait_for(turn, 1)
    response = next(
        event["response"]
        for event in websocket.events
        if event["type"] == "response.done"
    )
    assert response["status_details"]["reason"] == "turn_detected"


@pytest.mark.asyncio
async def test_cancelling_speech_started_handler_releases_terminal_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "StreamingVAD", FakeVAD)
    response_started = asyncio.Event()

    async def response() -> AsyncIterator[CompletionStreamChunk]:
        response_started.set()
        yield _chunk(text="partial")
        yield _chunk(modality="audio")
        await asyncio.Event().wait()

    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text="question")
        yield _chunk(finish_reason="stop")

    websocket = BlockingSpeechStartedWebSocket()
    client = ScriptedClient([response, transcription])
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]
    turn = asyncio.create_task(session.run_turn("item-user", "audio"))
    await response_started.wait()
    speech_started = asyncio.create_task(
        session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))
    )
    await websocket.speech_started_send.wait()

    speech_started.cancel()
    with pytest.raises(asyncio.CancelledError):
        await speech_started

    await asyncio.wait_for(turn, 1)
    response = next(
        event["response"]
        for event in websocket.events
        if event["type"] == "response.done"
    )
    assert response["status_details"]["reason"] == "turn_detected"


@pytest.mark.asyncio
async def test_repeated_pending_barge_in_keeps_first_terminal_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "StreamingVAD", FakeVAD)
    websocket = BlockingSecondSpeechStartedWebSocket()
    client = ScriptedClient([])
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]
    session.response_start_pending = True

    await session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))
    second_speech_started = asyncio.create_task(
        session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 1))
    )
    await websocket.second_speech_started_send.wait()

    response_output = await asyncio.wait_for(session.run_response("audio"), 0.1)

    assert response_output.text == ""
    response = next(
        event["response"]
        for event in websocket.events
        if event["type"] == "response.done"
    )
    assert response["status_details"]["reason"] == "turn_detected"
    assert not second_speech_started.done()

    websocket.release_second_speech_started.set()
    await second_speech_started


@pytest.mark.asyncio
async def test_explicit_cancel_before_speech_keeps_client_cancelled_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "StreamingVAD", FakeVAD)
    response_started = asyncio.Event()

    async def response() -> AsyncIterator[CompletionStreamChunk]:
        response_started.set()
        yield _chunk(text="partial")
        yield _chunk(modality="audio")
        await asyncio.Event().wait()

    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text="question")
        yield _chunk(finish_reason="stop")

    websocket = BlockingSpeechStartedWebSocket()
    client = ScriptedClient([response, transcription])
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]
    turn = asyncio.create_task(session.run_turn("item-user", "audio"))
    await response_started.wait()

    await session.handle_response_cancel(ResponseCancel(type="response.cancel"))
    speech_started = asyncio.create_task(
        session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))
    )
    await websocket.speech_started_send.wait()
    await asyncio.wait_for(turn, 1)

    response = next(
        event["response"]
        for event in websocket.events
        if event["type"] == "response.done"
    )
    assert response["status_details"]["reason"] == "client_cancelled"
    assert not speech_started.done()

    websocket.release_speech_started.set()
    await speech_started


@pytest.mark.asyncio
async def test_speech_cancel_before_explicit_cancel_keeps_turn_detected_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "StreamingVAD", FakeVAD)
    response_started = asyncio.Event()

    async def response() -> AsyncIterator[CompletionStreamChunk]:
        response_started.set()
        yield _chunk(text="partial")
        yield _chunk(modality="audio")
        await asyncio.Event().wait()

    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text="question")
        yield _chunk(finish_reason="stop")

    websocket = BlockingSpeechStartedWebSocket()
    client = ScriptedClient([response, transcription])
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]
    turn = asyncio.create_task(session.run_turn("item-user", "audio"))
    await response_started.wait()

    speech_started = asyncio.create_task(
        session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))
    )
    await websocket.speech_started_send.wait()
    await session.handle_response_cancel(ResponseCancel(type="response.cancel"))

    websocket.release_speech_started.set()
    await speech_started
    await asyncio.wait_for(turn, 1)

    response = next(
        event["response"]
        for event in websocket.events
        if event["type"] == "response.done"
    )
    assert response["status_details"]["reason"] == "turn_detected"


@pytest.mark.asyncio
async def test_failed_abort_cancels_local_response_before_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_started = asyncio.Event()
    release_response = asyncio.Event()
    transcription_started = asyncio.Event()

    async def response() -> AsyncIterator[CompletionStreamChunk]:
        response_started.set()
        yield _chunk(text="partial")
        yield _chunk(modality="audio")
        await release_response.wait()
        yield _chunk(finish_reason="stop")

    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        transcription_started.set()
        yield _chunk(text="original question")
        yield _chunk(finish_reason="stop")

    session, websocket, client = _session(monkeypatch, [response, transcription])

    async def abort() -> None:
        raise RuntimeError("abort failed")

    client.abort_hook = abort
    turn = asyncio.create_task(session.run_turn("item-user", "audio"))
    await response_started.wait()
    await session.dispatch({"type": "response.cancel"})

    try:
        await asyncio.wait_for(transcription_started.wait(), 0.05)
        transcription_started_before_stream_end = True
    except TimeoutError:
        transcription_started_before_stream_end = False
    terminal_count_before_stream_end = sum(
        event["type"] == "response.done" for event in websocket.events
    )

    release_response.set()
    await turn

    assert transcription_started_before_stream_end
    assert terminal_count_before_stream_end == 1
    response_done = next(
        event["response"]
        for event in websocket.events
        if event["type"] == "response.done"
    )
    assert response_done["status_details"]["reason"] == "client_cancelled"


@pytest.mark.asyncio
async def test_response_cancel_during_transcription_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcription_started = asyncio.Event()
    release_transcription = asyncio.Event()

    async def response() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text="answer")
        yield _chunk(finish_reason="stop")
        yield _chunk(modality="audio")
        yield _chunk(modality="audio", finish_reason="stop")

    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        transcription_started.set()
        await release_transcription.wait()
        yield _chunk(text="question")
        yield _chunk(finish_reason="stop")

    session, _, client = _session(monkeypatch, [response, transcription])
    turn = asyncio.create_task(session.run_turn("item-user", "audio"))
    await transcription_started.wait()

    await session.dispatch({"type": "response.cancel"})

    assert client.aborted == []
    release_transcription.set()
    await turn


@pytest.mark.asyncio
async def test_speech_blocks_queued_response_until_speech_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _, _ = _session(monkeypatch, [])
    response_started = asyncio.Event()

    async def run_turn(_item_id: str, _audio: str) -> None:
        response_started.set()

    session.run_turn = AsyncMock(side_effect=run_turn)  # type: ignore[method-assign]
    await session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))
    await session.response_queue.put(("queued", "audio"))
    session.queue_drainer = asyncio.create_task(session.drain_queue())
    await asyncio.sleep(0)
    session.run_turn.assert_not_awaited()

    await session.handle_vad_emit(Emit(VADEvent.SPEECH_STOPPED, 0))
    await asyncio.wait_for(response_started.wait(), 1)
    session.run_turn.assert_awaited_once_with("queued", "audio")
    await session.teardown()


@pytest.mark.asyncio
async def test_new_speech_cancels_dequeued_turn_waiting_for_speech_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def first_stream() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text="question")
        yield _chunk(finish_reason="stop")
        yield _chunk(modality="audio")
        yield _chunk(modality="audio", finish_reason="stop")

    async def unexpected_second_stream() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text="unexpected second request")
        yield _chunk(finish_reason="stop")

    session, _, client = _session(monkeypatch, [first_stream, unexpected_second_stream])
    session.speech_idle.clear()
    await session.response_queue.put(("queued-user-item", "audio"))
    session.queue_drainer = asyncio.create_task(session.drain_queue())

    for _ in range(100):
        if session.response_queue.empty() and session.response_start_pending:
            break
        await asyncio.sleep(0)
    assert session.response_queue.empty()
    assert session.response_start_pending

    await session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))
    session.speech_idle.set()

    for _ in range(100):
        if session.conversation:
            break
        await asyncio.sleep(0)

    assert client.requests and len(client.requests) == 1
    assert [(item.role, item.text) for item in session.conversation] == [
        ("user", "question")
    ]
    response = next(
        event["response"]
        for event in session.websocket.events  # type: ignore[attr-defined]
        if event["type"] == "response.done"
    )
    assert response["status_details"]["reason"] == "turn_detected"
    await session.teardown()


@pytest.mark.asyncio
async def test_pending_response_is_cancelled_before_stream_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, client = _session(monkeypatch, [])
    session.response_start_pending = True
    await session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))

    assert (await session.run_response("audio")).text == ""
    assert client.requests == []
    response = next(
        e["response"] for e in websocket.events if e["type"] == "response.done"
    )
    assert response["status_details"]["reason"] == "turn_detected"


@pytest.mark.asyncio
async def test_teardown_cancels_turn_without_starting_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_started = asyncio.Event()

    async def response() -> AsyncIterator[CompletionStreamChunk]:
        response_started.set()
        await asyncio.Event().wait()
        yield _chunk()

    session, _, client = _session(monkeypatch, [response])
    turn = asyncio.create_task(session.run_turn("item-user", "audio"))
    session.active_task = turn
    await response_started.wait()

    await session.teardown()

    assert turn.cancelled()
    assert len(client.requests) == 1
    assert session.conversation == []
