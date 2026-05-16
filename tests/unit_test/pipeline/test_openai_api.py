# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from sglang_omni.client import ClientError, GenerateChunk
from sglang_omni.serve import create_app


class FailingSpeechClient:
    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def generate(self, request: Any, request_id: str | None = None):
        del request, request_id
        yield GenerateChunk(
            request_id="speech-1",
            modality="audio",
            audio_data=[0.0, 0.1, -0.1, 0.0],
            sample_rate=24000,
        )
        raise ClientError("stream failed")


def test_speech_stream_returns_error_event_after_chunk_failure() -> None:
    """Preserves deterministic SSE termination after a mid-stream client error."""
    client = TestClient(create_app(FailingSpeechClient(), model_name="s2-pro"))

    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "model": "s2-pro",
            "input": "hello",
            "stream": True,
            "response_format": "wav",
        },
        timeout=5.0,
    ) as resp:
        assert resp.status_code == 200
        events = []
        done = False
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                done = True
                break
            events.append(json.loads(payload))

    assert done
    assert len(events) == 2
    assert events[0]["audio"] is not None
    assert events[0]["finish_reason"] is None
    assert events[1]["audio"] is None
    assert events[1]["finish_reason"] == "error"
    assert events[1]["error"] == {
        "type": "ClientError",
        "message": "stream failed",
    }
