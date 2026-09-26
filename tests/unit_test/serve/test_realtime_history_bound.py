# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import pytest

from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.serve.realtime.session import RealtimeSession
from tests.unit_test.serve.test_realtime_barge_in import make_chunk, make_session
from tests.unit_test.serve.test_realtime_history_truncation import (
    make_assistant_item_id,
)

StreamFactory = Callable[[], AsyncIterator[CompletionStreamChunk]]


def response_stream(text: str) -> StreamFactory:
    async def response() -> AsyncIterator[CompletionStreamChunk]:
        yield make_chunk(text=text)
        yield make_chunk(finish_reason="stop")
        yield make_chunk(modality="audio")
        yield make_chunk(modality="audio", finish_reason="stop")

    return response


def transcription_stream(text: str) -> StreamFactory:
    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        yield make_chunk(text=text)
        yield make_chunk(finish_reason="stop")

    return transcription


def turn_streams(turns: int) -> list[StreamFactory]:
    streams: list[StreamFactory] = []
    for index in range(turns):
        streams.append(response_stream(f"answer-{index}"))
        streams.append(transcription_stream(f"question-{index}"))
    return streams


async def run_turns(session: RealtimeSession, turns: int) -> None:
    for index in range(turns):
        await session.run_turn(f"user-item-{index}", "audio")


def history_items(session: RealtimeSession) -> list[tuple[str, str]]:
    return [(item.role, item.text) for item in session.conversation]


@pytest.mark.asyncio
async def test_default_history_is_unbounded(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _, _ = make_session(monkeypatch, turn_streams(3))

    await run_turns(session, 3)

    assert history_items(session) == [
        ("user", "question-0"),
        ("assistant", "answer-0"),
        ("user", "question-1"),
        ("assistant", "answer-1"),
        ("user", "question-2"),
        ("assistant", "answer-2"),
    ]


@pytest.mark.asyncio
async def test_bound_keeps_most_recent_complete_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _, _ = make_session(monkeypatch, turn_streams(3))
    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": 2}}
    )

    await run_turns(session, 3)

    assert history_items(session) == [
        ("user", "question-1"),
        ("assistant", "answer-1"),
        ("user", "question-2"),
        ("assistant", "answer-2"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_text", "transcript_text", "expected_history"),
    [
        ("", "question-1", [("user", "question-1")]),
        ("answer-1", "", [("assistant", "answer-1")]),
    ],
)
async def test_bound_keeps_single_sided_turn_intact(
    response_text: str,
    transcript_text: str,
    expected_history: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = [
        response_stream("answer-0"),
        transcription_stream("question-0"),
        response_stream(response_text),
        transcription_stream(transcript_text),
    ]
    session, _, _ = make_session(monkeypatch, streams)
    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": 1}}
    )

    await run_turns(session, 2)

    assert history_items(session) == expected_history
    assert {item.turn_id for item in session.conversation} == {"user-item-1"}


@pytest.mark.asyncio
async def test_new_bound_applies_to_existing_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _, _ = make_session(monkeypatch, turn_streams(3))
    await run_turns(session, 3)

    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": 1}}
    )

    assert history_items(session) == [
        ("user", "question-2"),
        ("assistant", "answer-2"),
    ]


@pytest.mark.asyncio
async def test_session_events_echo_and_clear_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    async def disconnect() -> dict[str, str]:
        return {"type": "websocket.disconnect"}

    monkeypatch.setattr(websocket, "receive", disconnect, raising=False)
    await session.run()
    assert websocket.events[-1]["type"] == "session.created"
    assert websocket.events[-1]["session"]["max_history_turns"] is None

    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": 5}}
    )
    assert session.session_object.max_history_turns == 5
    assert websocket.events[-1]["type"] == "session.updated"
    assert websocket.events[-1]["session"]["max_history_turns"] == 5

    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": None}}
    )
    assert session.session_object.max_history_turns is None
    assert websocket.events[-1]["type"] == "session.updated"
    assert websocket.events[-1]["session"]["max_history_turns"] is None


@pytest.mark.asyncio
async def test_omitted_bound_preserves_current_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])
    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": 5}}
    )

    await session.dispatch({"type": "session.update", "session": {"temperature": 0.5}})

    assert session.session_object.max_history_turns == 5
    assert websocket.events[-1]["type"] == "session.updated"
    assert websocket.events[-1]["session"]["max_history_turns"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, -1])
async def test_invalid_bound_rejected(
    value: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": value}}
    )

    assert session.session_object.max_history_turns is None
    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"]["type"] == "invalid_request_error"
    assert websocket.events[-1]["error"]["code"] == "invalid_max_history_turns"


@pytest.mark.asyncio
async def test_non_integer_bound_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": "1"}}
    )

    assert session.session_object.max_history_turns is None
    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"]["type"] == "invalid_request_error"
    assert websocket.events[-1]["error"]["code"] == "invalid_event"


@pytest.mark.asyncio
async def test_truncate_evicted_assistant_item_returns_item_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, turn_streams(2))
    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": 1}}
    )

    await session.run_turn("user-item-0", "audio")
    evicted_item_id = make_assistant_item_id(websocket.events)

    await session.run_turn("user-item-1", "audio")
    history_before = list(session.conversation)

    await session.dispatch(
        {
            "type": "conversation.item.truncate",
            "item_id": evicted_item_id,
            "content_index": 0,
            "audio_end_ms": 240,
        }
    )

    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"]["type"] == "invalid_request_error"
    assert websocket.events[-1]["error"]["code"] == "item_not_found"
    assert session.conversation == history_before
