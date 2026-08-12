# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.serve.realtime.vad import Emit, VADEvent
from tests.unit_test.serve.test_realtime_barge_in import _chunk, _session


def _assistant_item_id(events: list[dict]) -> str:
    response_done = next(event for event in events if event["type"] == "response.done")
    return response_done["response"]["output"][0]["id"]


def _response_stream() -> AsyncIterator[CompletionStreamChunk]:
    async def response() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text="full assistant answer")
        yield _chunk(finish_reason="stop")
        yield _chunk(modality="audio")
        yield _chunk(modality="audio", finish_reason="stop")

    return response()


def test_cancelled_assistant_item_tombstones_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _, _ = _session(monkeypatch, [])

    for index in range(65):
        session._remember_cancelled_assistant_item(f"assistant-{index}")

    assert list(session.cancelled_assistant_item_ids) == [
        f"assistant-{index}" for index in range(1, 65)
    ]


@pytest.mark.asyncio
async def test_truncate_before_history_append_omits_pending_assistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcription_started = asyncio.Event()
    release_transcription = asyncio.Event()

    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        transcription_started.set()
        await release_transcription.wait()
        yield _chunk(text="question")
        yield _chunk(finish_reason="stop")

    session, websocket, _ = _session(monkeypatch, [_response_stream, transcription])
    turn = asyncio.create_task(session.run_turn("user-item", "audio"))
    await transcription_started.wait()
    assistant_item_id = _assistant_item_id(websocket.events)

    await session.dispatch(
        {
            "type": "conversation.item.truncate",
            "item_id": assistant_item_id,
            "content_index": 0,
            "audio_end_ms": 240,
        }
    )
    release_transcription.set()
    await turn

    assert [(item.role, item.text) for item in session.conversation] == [
        ("user", "question")
    ]
    truncated = next(
        event
        for event in websocket.events
        if event["type"] == "conversation.item.truncated"
    )
    assert truncated["item_id"] == assistant_item_id
    assert truncated["audio_end_ms"] == 240


@pytest.mark.asyncio
async def test_truncate_after_history_append_removes_recorded_assistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text="question")
        yield _chunk(finish_reason="stop")

    session, websocket, _ = _session(monkeypatch, [_response_stream, transcription])
    await session.run_turn("user-item", "audio")
    assistant_item_id = _assistant_item_id(websocket.events)

    await session.dispatch(
        {
            "type": "conversation.item.truncate",
            "item_id": assistant_item_id,
            "content_index": 0,
            "audio_end_ms": 400,
        }
    )

    assert [(item.role, item.text) for item in session.conversation] == [
        ("user", "question")
    ]


@pytest.mark.asyncio
async def test_truncate_after_cancelled_response_is_acknowledged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late truncate is a no-op after cancellation already omitted the item."""
    response_started = asyncio.Event()
    transcription_started = asyncio.Event()
    release_transcription = asyncio.Event()

    async def response() -> AsyncIterator[CompletionStreamChunk]:
        response_started.set()
        yield _chunk(text="partial assistant answer")
        yield _chunk(modality="audio")
        await asyncio.Event().wait()

    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        transcription_started.set()
        await release_transcription.wait()
        yield _chunk(text="question")
        yield _chunk(finish_reason="stop")

    session, websocket, _ = _session(monkeypatch, [response, transcription])
    turn = asyncio.create_task(session.run_turn("user-item", "audio"))
    await response_started.wait()
    await session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))
    await transcription_started.wait()

    assistant_item_id = _assistant_item_id(websocket.events)
    event_count = len(websocket.events)
    await session.dispatch(
        {
            "type": "conversation.item.truncate",
            "item_id": assistant_item_id,
            "content_index": 0,
            "audio_end_ms": 10,
        }
    )
    truncate_events = websocket.events[event_count:]

    release_transcription.set()
    await turn

    assert [event["type"] for event in truncate_events] == [
        "conversation.item.truncated"
    ]
    assert [(item.role, item.text) for item in session.conversation] == [
        ("user", "question")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("item_id", "content_index", "expected_code"),
    [
        ("missing-item", 0, "item_not_found"),
        ("assistant-item", 1, "invalid_content_index"),
    ],
)
async def test_invalid_truncate_does_not_mutate_history(
    item_id: str,
    content_index: int,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text="question")
        yield _chunk(finish_reason="stop")

    session, websocket, _ = _session(monkeypatch, [_response_stream, transcription])
    await session.run_turn("user-item", "audio")
    assistant_item_id = _assistant_item_id(websocket.events)
    if item_id == "assistant-item":
        item_id = assistant_item_id
    history_before = list(session.conversation)

    await session.dispatch(
        {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": 10,
        }
    )

    assert session.conversation == history_before
    error = next(
        event for event in reversed(websocket.events) if event["type"] == "error"
    )
    assert error["error"]["code"] == expected_code
