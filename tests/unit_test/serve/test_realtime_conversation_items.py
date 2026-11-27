# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.serve.realtime.session import RealtimeSession
from tests.unit_test.serve.test_realtime_barge_in import make_session


def create_event(
    text: str,
    *,
    role: str = "user",
    previous_item_id: str | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        },
    }
    if previous_item_id is not None:
        event["previous_item_id"] = previous_item_id
    else:
        pass
    return event


def last_item_id(session: RealtimeSession) -> str:
    item_id = session.conversation[-1].item_id
    assert item_id is not None
    return item_id


@pytest.mark.asyncio
async def test_create_appends_text_to_history_and_next_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch(create_event("hello from the client"))

    assert [(item.role, item.text) for item in session.conversation] == [
        ("user", "hello from the client")
    ]
    created = websocket.events[-1]
    assert created["type"] == "conversation.item.created"
    assert created["item"]["role"] == "user"
    assert created["item"]["content"] == [
        {"type": "input_text", "text": "hello from the client"}
    ]

    request = session.build_response_request("audio")
    assert any(
        message.role == "user" and message.content == "hello from the client"
        for message in request.messages
    )


@pytest.mark.asyncio
async def test_create_after_previous_item_inserts_at_that_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _, _ = make_session(monkeypatch, [])
    await session.dispatch(create_event("first"))
    first_id = last_item_id(session)
    await session.dispatch(create_event("second"))

    await session.dispatch(create_event("between", previous_item_id=first_id))

    assert [item.text for item in session.conversation] == [
        "first",
        "between",
        "second",
    ]


@pytest.mark.asyncio
async def test_retrieve_returns_the_created_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])
    await session.dispatch(create_event("retrieve me"))
    item_id = last_item_id(session)

    await session.dispatch({"type": "conversation.item.retrieve", "item_id": item_id})

    retrieved = websocket.events[-1]
    assert retrieved["type"] == "conversation.item.retrieved"
    assert retrieved["item"]["id"] == item_id
    assert retrieved["item"]["content"] == [
        {"type": "input_text", "text": "retrieve me"}
    ]


@pytest.mark.asyncio
async def test_delete_removes_the_item_from_history_and_next_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])
    await session.dispatch(create_event("delete me"))
    item_id = last_item_id(session)

    await session.dispatch({"type": "conversation.item.delete", "item_id": item_id})

    assert session.conversation == []
    deleted = websocket.events[-1]
    assert deleted["type"] == "conversation.item.deleted"
    assert deleted["item_id"] == item_id
    request = session.build_response_request("audio")
    assert all(message.content != "delete me" for message in request.messages)


@pytest.mark.asyncio
async def test_unsupported_content_is_rejected_without_mutating_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_audio", "text": "not text"}],
            },
        }
    )

    assert session.conversation == []
    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"]["code"] == "invalid_item_content"


@pytest.mark.asyncio
async def test_non_message_item_is_rejected_without_mutating_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call",
                "role": "user",
                "content": [{"type": "input_text", "text": "not a message"}],
            },
        }
    )

    assert session.conversation == []
    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"]["code"] == "invalid_item_type"


@pytest.mark.asyncio
async def test_unknown_previous_item_is_rejected_without_mutating_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch(create_event("orphan", previous_item_id="item_missing"))

    assert session.conversation == []
    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"]["code"] == "item_not_found"


@pytest.mark.asyncio
async def test_delete_missing_item_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch(
        {"type": "conversation.item.delete", "item_id": "item_missing"}
    )

    assert session.conversation == []
    assert websocket.events[-1]["type"] == "error"
    assert websocket.events[-1]["error"]["code"] == "item_not_found"


@pytest.mark.asyncio
async def test_assistant_item_uses_text_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch(create_event("assistant text", role="assistant"))

    assert websocket.events[-1]["item"]["content"] == [
        {"type": "text", "text": "assistant text"}
    ]
