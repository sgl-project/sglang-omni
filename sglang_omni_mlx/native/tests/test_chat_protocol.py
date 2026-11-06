# SPDX-License-Identifier: Apache-2.0
"""The Ming3 audio chat HTTP protocol around a stand-in generator."""

from __future__ import annotations

import base64
import http.client
import json
import struct
from contextlib import closing

import pytest
from native_server import SupervisedServer, silent_wav

SAMPLE_COUNT = 400
MAX_REQUEST_BODY_BYTES = 32 * 1024 * 1024
PROMPT_TOKEN_COUNT = 17
COMPLETION_TOKEN_COUNT = 3
CHAT_PATH = "/v1/chat/completions"


@pytest.fixture
def audio_part() -> dict[str, object]:
    audio = silent_wav(SAMPLE_COUNT)[:-4] + struct.pack("<hh", 16384, -16384)
    return {
        "type": "input_audio",
        "input_audio": {"format": "wav", "data": base64.b64encode(audio).decode()},
    }


@pytest.fixture
def chat_body(
    chat_server: SupervisedServer, audio_part: dict[str, object]
) -> dict[str, object]:
    return {
        "model": chat_server.first_event["model_name"],
        "messages": [
            {"role": "system", "content": "Answer briefly."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "text", "text": " audio"},
                    audio_part,
                    {"type": "text", "text": "after"},
                    {"type": "text", "text": " audio"},
                ],
            },
        ],
        "temperature": 0,
        "modalities": ["text"],
        "stream": False,
        "max_tokens": 7,
    }


@pytest.fixture
def blocking_chat_body(chat_body: dict[str, object]) -> dict[str, object]:
    messages = chat_body["messages"]
    assert isinstance(messages, list)
    return {
        **chat_body,
        "messages": [{"role": "system", "content": "block"}, *messages[1:]],
    }


def test_a_request_decodes_audio_and_returns_a_chat_completion(
    chat_server: SupervisedServer, chat_body: dict[str, object]
) -> None:
    response = chat_server.post_chat(chat_body)
    assert response.status == 200
    assert response.headers["Content-Type"] == "application/json"
    assert int(response.headers["Content-Length"]) == len(response.body)
    reply = response.json()
    assert reply["id"].startswith("chatcmpl-")
    assert reply["object"] == "chat.completion"
    assert reply["model"] == chat_server.first_event["model_name"]
    assert reply["created"] > 0
    choice = reply["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] == "stop"
    assert json.loads(choice["message"]["content"]) == {
        "system_prompt": "Answer briefly.",
        "before_audio": "before audio",
        "after_audio": "after audio",
        "samples": [0.0] * (SAMPLE_COUNT - 2) + [0.5, -0.5],
        "max_new_tokens": 7,
    }
    assert reply["usage"] == {
        "prompt_tokens": PROMPT_TOKEN_COUNT,
        "completion_tokens": COMPLETION_TOKEN_COUNT,
        "total_tokens": PROMPT_TOKEN_COUNT + COMPLETION_TOKEN_COUNT,
    }
    repeated = chat_server.post_chat(chat_body).json()
    assert repeated["id"] != reply["id"]
    assert repeated["choices"] == reply["choices"]


@pytest.mark.parametrize(
    ("limits", "requested_tokens", "completion_tokens", "finish_reason"),
    [
        ({"max_tokens": 1}, 1, 1, "length"),
        ({"max_tokens": 5, "max_completion_tokens": 2}, 2, 2, "length"),
        ({"max_tokens": 1025}, 1025, 3, "stop"),
    ],
)
def test_token_limits_reach_the_generator(
    chat_server: SupervisedServer,
    chat_body: dict[str, object],
    limits: dict[str, int],
    requested_tokens: int,
    completion_tokens: int,
    finish_reason: str,
) -> None:
    response = chat_server.post_chat({**chat_body, **limits})
    assert response.status == 200
    reply = response.json()
    choice = reply["choices"][0]
    assert (
        json.loads(choice["message"]["content"])["max_new_tokens"] == requested_tokens
    )
    assert reply["usage"]["completion_tokens"] == completion_tokens
    assert choice["finish_reason"] == finish_reason


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("stream", True),
        ("temperature", 0.7),
        ("modalities", ["text", "audio"]),
        ("stop", ["end"]),
        ("model", "another-model"),
        ("max_completion_tokens", 0),
        ("max_completion_tokens", 2**32 + 1),
        ("max_completion_tokens", 1.5),
        ("messages", []),
    ],
)
def test_unsupported_requests_get_400(
    chat_server: SupervisedServer,
    chat_body: dict[str, object],
    field_name: str,
    field_value: object,
) -> None:
    response = chat_server.post_chat({**chat_body, field_name: field_value})
    assert response.status == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("body", [b"{", b"[]", b"null", b"{}"])
def test_invalid_json_gets_a_chat_error(
    chat_server: SupervisedServer, body: bytes
) -> None:
    response = chat_server.request(
        "POST", CHAT_PATH, body, {"Content-Type": "application/json"}
    )
    assert response.status == 400
    assert response.json()["error"] == {
        "message": "invalid JSON, missing field or incorrect field type",
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }


@pytest.mark.parametrize("damage", ["base64", "wav", "format", "missing", "multiple"])
def test_invalid_audio_gets_400(
    chat_server: SupervisedServer,
    chat_body: dict[str, object],
    audio_part: dict[str, object],
    damage: str,
) -> None:
    if damage == "base64":
        audio_parts = [
            {"type": "input_audio", "input_audio": {"format": "wav", "data": "%%%"}}
        ]
    elif damage == "wav":
        audio_parts = [
            {
                "type": "input_audio",
                "input_audio": {
                    "format": "wav",
                    "data": base64.b64encode(b"not a wav").decode(),
                },
            }
        ]
    elif damage == "format":
        audio_parts = [
            {"type": "input_audio", "input_audio": {"format": "mp3", "data": ""}}
        ]
    elif damage == "missing":
        audio_parts = []
    else:
        audio_parts = [audio_part, audio_part]
    response = chat_server.post_chat(
        {**chat_body, "messages": [{"role": "user", "content": audio_parts}]}
    )
    assert response.status == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_a_five_minute_audio_request_reaches_generation(
    chat_server: SupervisedServer,
    chat_body: dict[str, object],
    audio_part: dict[str, object],
) -> None:
    """The former 4 MiB body cap rejected long audio before inference."""
    sample_count = 310 * 16000
    audio_part["input_audio"] = {
        "format": "wav",
        "data": base64.b64encode(silent_wav(sample_count)).decode(),
    }
    response = chat_server.post_chat(chat_body)
    assert response.status == 200
    reply = response.json()
    samples = json.loads(reply["choices"][0]["message"]["content"])["samples"]
    assert len(samples) == sample_count
    assert all(sample == 0.0 for sample in samples)
    assert reply["usage"]["completion_tokens"] == COMPLETION_TOKEN_COUNT


def test_methods_routes_and_body_limit(chat_server: SupervisedServer) -> None:
    response = chat_server.request("GET", CHAT_PATH)
    assert response.status == 405
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert chat_server.request("POST", "/v1/audio/transcriptions").status == 404
    oversized = chat_server.request(
        "POST", CHAT_PATH, b" " * (MAX_REQUEST_BODY_BYTES + 1)
    )
    assert oversized.status == 413


@pytest.mark.parametrize(
    ("outcome", "status", "message"),
    [("fail", 500, "generation failed"), ("cancel", 503, "generation cancelled")],
)
def test_generation_errors_release_the_slot(
    chat_server: SupervisedServer,
    chat_body: dict[str, object],
    outcome: str,
    status: int,
    message: str,
) -> None:
    messages = chat_body["messages"]
    assert isinstance(messages, list)
    response = chat_server.post_chat(
        {
            **chat_body,
            "messages": [{"role": "system", "content": outcome}, *messages[1:]],
        }
    )
    assert response.status == status
    assert response.json()["error"] == {
        "message": message,
        "type": "server_error",
        "param": None,
        "code": None,
    }
    chat_server.wait_for_request_states({})
    assert chat_server.post_chat(chat_body).status == 200


def test_busy_rejection_does_not_interrupt_the_active_request(
    chat_server: SupervisedServer,
    chat_body: dict[str, object],
    blocking_chat_body: dict[str, object],
) -> None:
    with closing(
        chat_server.open_request(
            CHAT_PATH, "application/json", json.dumps(blocking_chat_body).encode()
        )
    ) as client:
        assert chat_server.request("GET", "/stub/chat/started").status == 200
        assert chat_server.request_states() == {"running": 1}
        assert chat_server.post_chat({**chat_body, "stream": True}).status == 400
        messages = chat_body["messages"]
        assert isinstance(messages, list)
        invalid = chat_server.post_chat(
            {
                **chat_body,
                "messages": [{"role": "system", "content": "invalid"}, *messages[1:]],
            }
        )
        assert invalid.status == 400
        assert invalid.json()["error"]["message"] == "stub chat preparation failed"
        refused = chat_server.post_chat(chat_body)
        assert refused.status == 429
        assert refused.json()["error"]["type"] == "rate_limit_error"
        assert chat_server.request("POST", "/stub/chat/release").status == 200
        with http.client.HTTPResponse(client) as completed:
            completed.begin()
            assert completed.status == 200
            reply = json.loads(completed.read())
            assert reply["choices"][0]["finish_reason"] == "stop"
    chat_server.wait_for_request_states({})


def test_disconnect_cancels_generation_and_allows_the_next_request(
    chat_server: SupervisedServer,
    chat_body: dict[str, object],
    blocking_chat_body: dict[str, object],
) -> None:
    with closing(
        chat_server.open_request(
            CHAT_PATH, "application/json", json.dumps(blocking_chat_body).encode()
        )
    ):
        assert chat_server.request("GET", "/stub/chat/started").status == 200
        assert chat_server.request_states() == {"running": 1}
    chat_server.wait_for_request_states({})
    assert chat_server.post_chat(chat_body).status == 200
