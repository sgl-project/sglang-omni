# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import pytest

from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.serve.realtime.session import RealtimeSession
from tests.unit_test.serve.test_realtime_barge_in import _chunk, _session

StreamFactory = Callable[[], AsyncIterator[CompletionStreamChunk]]


def _response_stream(text: str) -> StreamFactory:
    async def response() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text=text)
        yield _chunk(finish_reason="stop")
        yield _chunk(modality="audio")
        yield _chunk(modality="audio", finish_reason="stop")

    return response


def _transcription_stream(text: str) -> StreamFactory:
    async def transcription() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text=text)
        yield _chunk(finish_reason="stop")

    return transcription


def _turn_streams(turns: int) -> list[StreamFactory]:
    streams: list[StreamFactory] = []
    for index in range(turns):
        streams.append(_response_stream(f"answer-{index}"))
        streams.append(_transcription_stream(f"question-{index}"))
    return streams


async def _run_turns(session: RealtimeSession, turns: int) -> None:
    for index in range(turns):
        await session.run_turn(f"user-item-{index}", "audio")


def _history(session: RealtimeSession) -> list[tuple[str, str]]:
    return [(item.role, item.text) for item in session.conversation]


@pytest.mark.asyncio
async def test_default_history_is_unbounded(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _, _ = _session(monkeypatch, _turn_streams(3))

    await _run_turns(session, 3)

    assert _history(session) == [
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
    session, _, _ = _session(monkeypatch, _turn_streams(3))
    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": 2}}
    )

    await _run_turns(session, 3)

    assert _history(session) == [
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
        _response_stream("answer-0"),
        _transcription_stream("question-0"),
        _response_stream(response_text),
        _transcription_stream(transcript_text),
    ]
    session, _, _ = _session(monkeypatch, streams)
    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": 1}}
    )

    await _run_turns(session, 2)

    assert _history(session) == expected_history
    assert {item.turn_id for item in session.conversation} == {"user-item-1"}


@pytest.mark.asyncio
async def test_new_bound_applies_to_existing_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _, _ = _session(monkeypatch, _turn_streams(3))
    await _run_turns(session, 3)

    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": 1}}
    )

    assert _history(session) == [
        ("user", "question-2"),
        ("assistant", "answer-2"),
    ]


@pytest.mark.asyncio
async def test_session_events_echo_and_clear_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = _session(monkeypatch, [])

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
@pytest.mark.parametrize("value", [0, -1])
async def test_invalid_bound_rejected(
    value: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, websocket, _ = _session(monkeypatch, [])

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
    session, websocket, _ = _session(monkeypatch, [])

    await session.dispatch(
        {"type": "session.update", "session": {"max_history_turns": "1"}}
    )

    assert session.session_object.max_history_turns is None
    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"]["type"] == "invalid_request_error"
    assert websocket.events[-1]["error"]["code"] == "invalid_event"
