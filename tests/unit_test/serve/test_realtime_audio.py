# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.serve.realtime.events import ResponseCancel, TurnDetection
from sglang_omni.serve.realtime.session import RealtimeSession
from sglang_omni.serve.realtime.vad import VADEvent


class RecordingWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self, timeline: list[str] | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self.timeline = timeline

    async def send_text(self, payload: str) -> None:
        event = json.loads(payload)
        self.events.append(event)
        if self.timeline is not None:
            self.timeline.append(event["type"])


def _chunk(
    *,
    modality: str = "text",
    text: str = "",
    audio_b64: str | None = None,
    finish_reason: str | None = None,
) -> CompletionStreamChunk:
    return CompletionStreamChunk(
        request_id="request",
        modality=modality,
        text=text,
        audio_b64=audio_b64,
        finish_reason=finish_reason,
    )


def _speech_started_emit() -> SimpleNamespace:
    return SimpleNamespace(event_type=VADEvent.SPEECH_STARTED, sample_offset=0)


async def _wait_for_conversation_size(
    session: RealtimeSession, expected_size: int
) -> None:
    async def wait() -> None:
        while len(session.conversation) < expected_size:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "modalities",
        "interrupt_response",
        "active_response_has_audio",
        "response_start_pending",
        "expected_cancel",
    ),
    [
        (["text", "audio"], None, True, False, True),
        (["text", "audio"], False, True, False, False),
        (["text"], None, False, False, False),
        (["text", "audio"], None, False, True, True),
    ],
)
async def test_speech_started_only_interrupts_audio_responses(
    modalities: list[str],
    interrupt_response: bool | None,
    active_response_has_audio: bool,
    response_start_pending: bool,
    expected_cancel: bool,
) -> None:
    timeline: list[str] = []
    session = RealtimeSession(
        RecordingWebSocket(timeline),  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = modalities
    if interrupt_response is not None:
        session.session_object.turn_detection = TurnDetection(
            interrupt_response=interrupt_response
        )
    session.active_response_has_audio = active_response_has_audio
    session.response_start_pending = response_start_pending

    async def record_cancel(reason: str) -> None:
        timeline.append(f"cancel:{reason}")

    session.cancel_active_response = record_cancel  # type: ignore[method-assign]

    await session.handle_vad_emit(_speech_started_emit())

    expected = ["input_audio_buffer.speech_started"]
    if expected_cancel:
        expected.append("cancel:turn_detected")
    assert timeline == expected


@pytest.mark.asyncio
async def test_explicit_cancel_uses_client_reason() -> None:
    session = RealtimeSession(
        RecordingWebSocket(),  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
        model_name="qwen3-omni",
    )
    reasons: list[str] = []

    async def record_cancel(reason: str) -> None:
        reasons.append(reason)

    session.cancel_active_response = record_cancel  # type: ignore[method-assign]

    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )

    assert reasons == ["client_cancelled"]


@pytest.mark.asyncio
async def test_barge_in_preserves_transcription_and_partial_history() -> None:
    response_started = asyncio.Event()

    class Client:
        def __init__(self) -> None:
            self.calls = 0
            self.aborted: list[str] = []

        async def completion_stream(
            self,
            request: Any,
            *,
            request_id: str,
            audio_format: str = "wav",
        ) -> AsyncIterator[CompletionStreamChunk]:
            del request, request_id, audio_format
            self.calls += 1
            if self.calls == 1:
                yield _chunk(text="partial answer")
                yield _chunk(modality="audio", audio_b64="AQI=")
                response_started.set()
                await asyncio.Future()
            else:
                yield _chunk(text="remember this")
                yield _chunk(finish_reason="stop")

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    websocket = RecordingWebSocket()
    client = Client()
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]

    turn_task = asyncio.create_task(
        session.run_turn("item-user", "data:audio/wav;base64,AAAA")
    )
    await asyncio.wait_for(response_started.wait(), timeout=1)
    await session.handle_vad_emit(_speech_started_emit())
    await asyncio.wait_for(turn_task, timeout=1)

    assert len(client.aborted) == 1
    assert [(item.role, item.text) for item in session.conversation] == [
        ("user", "remember this"),
        ("assistant", "partial answer"),
    ]

    event_types = [event["type"] for event in websocket.events]
    speech_started_index = event_types.index("input_audio_buffer.speech_started")
    audio_done_index = event_types.index("response.audio.done")
    response_done_index = event_types.index("response.done")
    assert speech_started_index < audio_done_index < response_done_index

    response_done = websocket.events[response_done_index]["response"]
    assert response_done["status"] == "cancelled"
    assert response_done["status_details"]["reason"] == "turn_detected"
    assert "conversation.item.input_audio_transcription.completed" in event_types


@pytest.mark.asyncio
async def test_response_cancel_during_transcription_is_noop() -> None:
    transcription_started = asyncio.Event()
    finish_transcription = asyncio.Event()

    class Client:
        def __init__(self) -> None:
            self.calls = 0
            self.aborted: list[str] = []

        async def completion_stream(
            self,
            request: Any,
            *,
            request_id: str,
            audio_format: str = "wav",
        ) -> AsyncIterator[CompletionStreamChunk]:
            del request, request_id, audio_format
            self.calls += 1
            if self.calls == 1:
                yield _chunk(text="answer")
                yield _chunk(finish_reason="stop")
            else:
                transcription_started.set()
                await finish_transcription.wait()
                yield _chunk(text="user question")
                yield _chunk(finish_reason="stop")

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    websocket = RecordingWebSocket()
    client = Client()
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
    )

    turn_task = asyncio.create_task(
        session.run_turn("item-user", "data:audio/wav;base64,AAAA")
    )
    await asyncio.wait_for(transcription_started.wait(), timeout=1)
    await session.handle_response_cancel(
        ResponseCancel.model_validate({"type": "response.cancel"})
    )
    finish_transcription.set()
    await asyncio.wait_for(turn_task, timeout=1)

    assert client.aborted == []
    assert [(item.role, item.text) for item in session.conversation] == [
        ("user", "user question"),
        ("assistant", "answer"),
    ]


@pytest.mark.asyncio
async def test_barge_in_allows_next_queued_turn_to_complete() -> None:
    first_response_started = asyncio.Event()

    class Client:
        def __init__(self) -> None:
            self.calls = 0
            self.aborted: list[str] = []

        async def completion_stream(
            self,
            request: Any,
            *,
            request_id: str,
            audio_format: str = "wav",
        ) -> AsyncIterator[CompletionStreamChunk]:
            del request, request_id, audio_format
            self.calls += 1
            if self.calls == 1:
                yield _chunk(text="partial first answer")
                yield _chunk(modality="audio", audio_b64="AQI=")
                first_response_started.set()
                await asyncio.Future()
            elif self.calls == 2:
                yield _chunk(text="first question")
                yield _chunk(finish_reason="stop")
            elif self.calls == 3:
                yield _chunk(text="second answer")
                yield _chunk(modality="audio", audio_b64="AwQ=")
                yield _chunk(finish_reason="stop")
            else:
                yield _chunk(text="second question")
                yield _chunk(finish_reason="stop")

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    websocket = RecordingWebSocket()
    client = Client()
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]
    session.response_queue.put_nowait(("item-first", "data:audio/wav;base64,AAAA"))
    session.response_queue.put_nowait(("item-second", "data:audio/wav;base64,BBBB"))

    drainer = asyncio.create_task(session.drain_queue())
    await asyncio.wait_for(first_response_started.wait(), timeout=1)
    await session.handle_vad_emit(_speech_started_emit())
    await _wait_for_conversation_size(session, 4)

    drainer.cancel()
    await asyncio.gather(drainer, return_exceptions=True)

    assert len(client.aborted) == 1
    assert [(item.role, item.text) for item in session.conversation] == [
        ("user", "first question"),
        ("assistant", "partial first answer"),
        ("user", "second question"),
        ("assistant", "second answer"),
    ]
    response_statuses = [
        (
            event["response"]["status"],
            event["response"]["status_details"]["reason"],
        )
        for event in websocket.events
        if event["type"] == "response.done"
    ]
    assert response_statuses == [
        ("cancelled", "turn_detected"),
        ("completed", "stop"),
    ]


@pytest.mark.asyncio
async def test_turn_cleanup_does_not_continue_to_transcription() -> None:
    response_started = asyncio.Event()

    class Client:
        def __init__(self) -> None:
            self.calls = 0
            self.aborted: list[str] = []

        async def completion_stream(
            self,
            request: Any,
            *,
            request_id: str,
            audio_format: str = "wav",
        ) -> AsyncIterator[CompletionStreamChunk]:
            del request, request_id, audio_format
            self.calls += 1
            response_started.set()
            await asyncio.Future()
            yield

        async def abort(self, request_id: str) -> None:
            self.aborted.append(request_id)

    client = Client()
    session = RealtimeSession(
        RecordingWebSocket(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
    )

    turn_task = asyncio.create_task(
        session.run_turn("item-user", "data:audio/wav;base64,AAAA")
    )
    session.active_task = turn_task
    await asyncio.wait_for(response_started.wait(), timeout=1)
    await session._cancel_and_abort(turn_task, session.active_request_id)

    assert turn_task.done()
    assert client.calls == 1
    assert len(client.aborted) == 1
    assert session.conversation == []
