# SPDX-License-Identifier: Apache-2.0
"""Tests for the minimal OpenAI-compatible adapter."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from sglang_omni.gateway import GenerateChunk
from sglang_omni.serve import create_app


class DummyGateway:
    def __init__(self, chunks: list[GenerateChunk]):
        self._chunks = chunks

    async def generate(self, request, request_id=None):
        for chunk in self._chunks:
            yield chunk


def test_chat_completions_non_stream() -> None:
    gateway = DummyGateway(
        [GenerateChunk(request_id="req-1", text="hello", finish_reason="stop")]
    )
    client = TestClient(create_app(gateway))

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hello"


def test_chat_completions_stream() -> None:
    gateway = DummyGateway(
        [
            GenerateChunk(request_id="req-1", text="hi"),
            GenerateChunk(request_id="req-1", finish_reason="stop"),
        ]
    )
    client = TestClient(create_app(gateway))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        timeout=5.0,
    ) as resp:
        assert resp.status_code == 200
        events = []
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            events.append(json.loads(payload))

    deltas = [event["choices"][0]["delta"] for event in events]
    assert any("content" in delta for delta in deltas)
