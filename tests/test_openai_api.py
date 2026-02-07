# SPDX-License-Identifier: Apache-2.0
"""Tests for the minimal OpenAI-compatible adapter."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from sglang_omni.client import (
    CompletionResult,
    CompletionStreamChunk,
    GenerateChunk,
)
from sglang_omni.serve import create_app


class DummyClient:
    """Minimal stand-in for ``Client`` that replays pre-built chunks."""

    def __init__(self, chunks: list[GenerateChunk]):
        self._chunks = chunks

    async def generate(self, request: Any, request_id: str | None = None):
        for chunk in self._chunks:
            yield chunk

    async def completion(
        self, request: Any, *, request_id: str, audio_format: str = "wav"
    ) -> CompletionResult:
        text_parts: list[str] = []
        finish_reason: str | None = None
        async for chunk in self.generate(request, request_id=request_id):
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
        return CompletionResult(
            request_id=request_id,
            text="".join(text_parts),
            finish_reason=finish_reason or "stop",
        )

    async def completion_stream(
        self, request: Any, *, request_id: str, audio_format: str = "wav"
    ):
        async for chunk in self.generate(request, request_id=request_id):
            yield CompletionStreamChunk(
                request_id=request_id,
                text=chunk.text,
                modality=chunk.modality,
                finish_reason=chunk.finish_reason,
            )

    def health(self) -> dict[str, Any]:
        return {"running": True}


def test_chat_completions_non_stream() -> None:
    dummy = DummyClient(
        [GenerateChunk(request_id="req-1", text="hello", finish_reason="stop")]
    )
    client = TestClient(create_app(dummy))

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
    dummy = DummyClient(
        [
            GenerateChunk(request_id="req-1", text="hi"),
            GenerateChunk(request_id="req-1", finish_reason="stop"),
        ]
    )
    client = TestClient(create_app(dummy))

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
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            events.append(json.loads(payload))

    deltas = [event["choices"][0]["delta"] for event in events]
    assert any("content" in delta for delta in deltas)
