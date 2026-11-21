# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.serve.realtime.session import RealtimeSession
from tests.unit_test.serve.test_realtime_barge_in import make_session


def make_message_item(
    text: str, *, role: str = "user", item_id: str | None = None
) -> dict[str, object]:
    item: dict[str, object] = {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }
    if item_id is None:
        return item
    else:
        pass
    return {**item, "id": item_id}


def last_event(events: list[dict], event_type: str) -> dict:
    return next(event for event in reversed(events) if event["type"] == event_type)


def rendered_messages(session: RealtimeSession) -> list[tuple[str, str]]:
    request = session.build_response_request("audio")
    return [(message.role, message.content) for message in request.messages]


@pytest.mark.asyncio
async def test_create_appends_a_text_turn_to_the_next_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch(
        {
            "type": "conversation.item.create",
            "item": make_message_item("hello there"),
        }
    )

    created = last_event(websocket.events, "conversation.item.created")
    item_id = created["item"]["id"]
    assert created["item"] == {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "user",
        "content": [{"type": "input_text", "text": "hello there"}],
    }
    assert "previous_item_id" not in created
    assert ("user", "hello there") in rendered_messages(session)


@pytest.mark.asyncio
async def test_create_places_the_item_after_previous_item_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])
    await session.dispatch(
        {"type": "conversation.item.create", "item": make_message_item("first")}
    )
    await session.dispatch(
        {"type": "conversation.item.create", "item": make_message_item("third")}
    )
    first_id = session.conversation[0].item_id

    await session.dispatch(
        {
            "type": "conversation.item.create",
            "previous_item_id": first_id,
            "item": make_message_item("second"),
        }
    )

    assert [item.text for item in session.conversation] == ["first", "second", "third"]
    created = last_event(websocket.events, "conversation.item.created")
    assert created["previous_item_id"] == first_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("item", "expected_code"),
    [
        ({"type": "function_call", "name": "lookup"}, "unsupported_item_type"),
        ({"type": "message", "role": "system", "content": []}, "invalid_role"),
        (
            {"type": "message", "role": "user", "content": []},
            "unsupported_content_type",
        ),
        (
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_audio", "audio": "AQI="}],
            },
            "unsupported_content_type",
        ),
        (
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "https://x/y.png"}],
            },
            "unsupported_content_type",
        ),
    ],
)
async def test_create_rejects_unsupported_items_without_touching_history(
    item: dict[str, object],
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch({"type": "conversation.item.create", "item": item})

    assert session.conversation == []
    error = last_event(websocket.events, "error")
    assert error["error"]["code"] == expected_code


@pytest.mark.asyncio
async def test_create_rejects_a_duplicate_item_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch(
        {
            "type": "conversation.item.create",
            "item": make_message_item("first", item_id="item_fixed"),
        }
    )
    await session.dispatch(
        {
            "type": "conversation.item.create",
            "item": make_message_item("second", item_id="item_fixed"),
        }
    )

    assert [item.text for item in session.conversation] == ["first"]
    error = last_event(websocket.events, "error")
    assert error["error"]["code"] == "item_already_exists"


@pytest.mark.asyncio
async def test_retrieve_returns_the_created_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])
    await session.dispatch(
        {
            "type": "conversation.item.create",
            "item": make_message_item("remember me", role="assistant"),
        }
    )
    item_id = session.conversation[0].item_id

    await session.dispatch({"type": "conversation.item.retrieve", "item_id": item_id})

    assert last_event(websocket.events, "conversation.item.retrieved")["item"] == {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "input_text", "text": "remember me"}],
    }


@pytest.mark.asyncio
async def test_delete_removes_the_item_from_the_next_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])
    await session.dispatch(
        {"type": "conversation.item.create", "item": make_message_item("keep")}
    )
    await session.dispatch(
        {"type": "conversation.item.create", "item": make_message_item("drop")}
    )
    dropped_id = session.conversation[1].item_id

    await session.dispatch({"type": "conversation.item.delete", "item_id": dropped_id})

    assert last_event(websocket.events, "conversation.item.deleted")["item_id"] == (
        dropped_id
    )
    assert ("user", "drop") not in rendered_messages(session)
    assert ("user", "keep") in rendered_messages(session)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "expected_code"),
    [
        ("conversation.item.retrieve", "item_not_found"),
        ("conversation.item.delete", "item_not_found"),
    ],
)
async def test_unknown_item_reports_item_not_found(
    event_type: str, expected_code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch({"type": event_type, "item_id": "missing"})

    error = last_event(websocket.events, "error")
    assert error["error"]["code"] == expected_code


@pytest.mark.asyncio
async def test_create_after_previous_item_id_that_is_missing_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = make_session(monkeypatch, [])

    await session.dispatch(
        {
            "type": "conversation.item.create",
            "previous_item_id": "missing",
            "item": make_message_item("hello"),
        }
    )

    assert session.conversation == []
    error = last_event(websocket.events, "error")
    assert error["error"]["code"] == "item_not_found"
