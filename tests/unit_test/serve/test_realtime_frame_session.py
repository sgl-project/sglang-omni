# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.config import RealtimeAudioConfig
from sglang_omni.serve.realtime.frame_session import FrameRealtimeSession


class RecordingWebSocket:
    application_state = WebSocketState.CONNECTED
    client_state = WebSocketState.CONNECTED

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send_text(self, payload: str) -> None:
        self.events.append(json.loads(payload))

    async def close(self) -> None:
        self.application_state = WebSocketState.DISCONNECTED
        self.client_state = WebSocketState.DISCONNECTED


class RecordingClient:
    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.aborted: list[str] = []

    async def completion_stream(
        self, request: Any, *, request_id: str, audio_format: str
    ) -> AsyncIterator[CompletionStreamChunk]:
        self.requests.append(request)
        assert audio_format == "pcm"
        yield CompletionStreamChunk(
            request_id=request_id,
            modality="audio",
            audio_b64="AQI=",
            finish_reason="stop",
        )

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


def _session(
    frame_samples: int = 4,
) -> tuple[FrameRealtimeSession, RecordingWebSocket, RecordingClient]:
    websocket = RecordingWebSocket()
    client = RecordingClient()
    session = FrameRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="voicechat",
        config=RealtimeAudioConfig(
            mode="frame",
            input_sample_rate=16_000,
            output_sample_rate=22_050,
            frame_samples=frame_samples,
        ),
    )
    return session, websocket, client


@pytest.mark.asyncio
async def test_append_splits_arbitrary_chunks_into_exact_pcm_frames() -> None:
    session, _, _ = _session(frame_samples=4)
    raw = bytes(range(12))

    await session.dispatch(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(raw).decode(),
        }
    )

    assert session._frames.qsize() == 1
    assert await session._frames.get() == raw[:8]
    session._frames.task_done()
    assert bytes(session._partial_pcm) == raw[8:]


@pytest.mark.asyncio
async def test_commit_pads_partial_frame_and_waits_for_generation() -> None:
    session, websocket, client = _session(frame_samples=4)
    session._worker = asyncio.create_task(session._drain_frames())
    try:
        raw = bytes(range(6))
        await session.dispatch(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(raw).decode(),
            }
        )
        await session.dispatch({"type": "input_audio_buffer.commit"})

        assert len(client.requests) == 1
        assert base64.b64decode(client.requests[0].prompt["pcm16"]) == raw + bytes(2)
        assert websocket.events[-1]["type"] == "input_audio_buffer.committed"
    finally:
        await session._frames.put(None)
        await session._worker
        session._worker = None


@pytest.mark.asyncio
async def test_frame_request_carries_stable_session_and_emits_audio_delta() -> None:
    session, websocket, client = _session()
    await session.dispatch(
        {"type": "session.update", "session": {"instructions": "Be concise."}}
    )

    await session._run_frame(bytes(8), frame_index=7)

    assert client.requests[0].prompt == {
        "event": "audio_frame",
        "session_id": session.session_id,
        "frame_index": 7,
        "pcm16": base64.b64encode(bytes(8)).decode("ascii"),
        "instructions": "Be concise.",
    }
    audio = [
        event for event in websocket.events if event["type"] == "response.audio.delta"
    ]
    assert audio == [
        {
            "type": "response.audio.delta",
            "delta": "AQI=",
            "sample_rate": 22_050,
            "frame_index": 7,
            "event_id": audio[0]["event_id"],
        }
    ]
