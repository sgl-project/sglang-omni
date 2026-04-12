# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from sglang_omni.realtime.backend import BackendCapabilities, ResponseEvent
from sglang_omni.realtime.session import RealtimeSession, RealtimeSessionConfig
from sglang_omni.realtime.vad import VadConfig


class _FakeChannel:
    readyState = "open"

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send(self, raw: str) -> None:
        self.messages.append(json.loads(raw))


class _FakeOutputTrack:
    def __init__(self) -> None:
        self.pending_samples = 0
        self.clear_calls = 0
        self.enqueue_calls: list[tuple[np.ndarray, int]] = []

    async def clear(self) -> None:
        self.clear_calls += 1
        self.pending_samples = 0

    async def enqueue(self, audio: np.ndarray, sample_rate: int) -> None:
        self.enqueue_calls.append((np.asarray(audio), sample_rate))
        self.pending_samples = 0


class _ScriptedBackend:
    model_name = "fake-model"
    capabilities = BackendCapabilities(
        accepts_audio_input=True,
        accepts_video_input=True,
        returns_text=True,
        returns_audio=True,
        supports_cancel=True,
    )

    def __init__(self) -> None:
        self.turns = []
        self.cancelled: list[str] = []

    async def stream_response(self, turn):
        self.turns.append(turn)
        response_idx = len(self.turns)
        response_id = f"resp-{response_idx}"
        yield ResponseEvent(type="response_started", response_id=response_id)
        yield ResponseEvent(
            type="text_delta",
            response_id=response_id,
            text=f"answer-{response_idx}",
        )
        yield ResponseEvent(
            type="audio_chunk",
            response_id=response_id,
            audio=np.array([0.1, -0.1], dtype=np.float32),
            sample_rate=24000,
        )
        yield ResponseEvent(
            type="done",
            response_id=response_id,
            finish_reason="stop",
        )

    async def cancel(self, response_id: str) -> None:
        self.cancelled.append(response_id)


class _BlockingBackend:
    model_name = "fake-model"
    capabilities = BackendCapabilities(
        accepts_audio_input=True,
        returns_text=True,
        returns_audio=False,
        supports_cancel=True,
    )

    def __init__(self) -> None:
        self.turns = []
        self.cancelled: list[str] = []
        self.started = asyncio.Event()
        self.released = asyncio.Event()

    async def stream_response(self, turn):
        self.turns.append(turn)
        response_id = "resp-cancel"
        yield ResponseEvent(type="response_started", response_id=response_id)
        self.started.set()
        await self.released.wait()
        yield ResponseEvent(
            type="done",
            response_id=response_id,
            finish_reason="cancelled",
        )

    async def cancel(self, response_id: str) -> None:
        self.cancelled.append(response_id)
        self.released.set()


def _make_session(backend) -> tuple[RealtimeSession, _FakeOutputTrack, _FakeChannel]:
    output_track = _FakeOutputTrack()
    channel = _FakeChannel()
    session = RealtimeSession(
        session_id="session-1",
        backend=backend,
        output_track=output_track,
        config=RealtimeSessionConfig(
            vad=VadConfig(
                start_threshold=0.02,
                stop_threshold=0.01,
                min_speech_s=0.1,
                min_silence_s=0.1,
                preroll_s=0.0,
            )
        ),
    )
    session.attach_event_channel(channel)
    return session, output_track, channel


async def _drive_turn(
    session: RealtimeSession,
    *,
    user_text: str,
    start_ts: float,
) -> None:
    await session.handle_client_event(
        {
            "type": "conversation.item.create",
            "item": {"role": "user", "content": user_text},
        }
    )

    speech = np.full(1600, 0.2, dtype=np.float32)
    silence = np.zeros(1600, dtype=np.float32)

    await session.handle_audio_chunk(speech, 16000, timestamp=start_ts)
    await session.handle_audio_chunk(speech, 16000, timestamp=start_ts + 0.1)
    await session.handle_audio_chunk(silence, 16000, timestamp=start_ts + 0.2)

    task = session.active_task
    assert task is not None
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_realtime_session_runs_turns_with_fake_backend_and_history():
    backend = _ScriptedBackend()
    session, output_track, channel = _make_session(backend)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    session.handle_video_frame(frame, timestamp=1.0)

    await _drive_turn(session, user_text="describe this", start_ts=2.0)
    await _drive_turn(session, user_text="follow up", start_ts=4.0)

    assert len(backend.turns) == 2
    assert backend.turns[0].user_text == "describe this"
    assert backend.turns[0].recent_video is not None
    assert backend.turns[1].history == [
        {"role": "user", "content": "describe this"},
        {"role": "assistant", "content": "answer-1"},
    ]

    assert session.history == [
        {"role": "user", "content": "describe this"},
        {"role": "assistant", "content": "answer-1"},
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "answer-2"},
    ]

    assert len(output_track.enqueue_calls) == 2
    assert output_track.enqueue_calls[0][1] == 24000
    assert output_track.clear_calls >= 2

    event_types = [event["type"] for event in channel.messages]
    assert event_types.count("conversation.item.created") == 2
    assert event_types.count("input_audio_buffer.speech_started") == 2
    assert event_types.count("input_audio_buffer.speech_stopped") == 2
    assert event_types.count("response.created") == 2
    assert event_types.count("response.output_text.delta") == 2
    assert event_types.count("response.output_audio.delta") == 2
    assert event_types.count("response.done") == 2


@pytest.mark.asyncio
async def test_realtime_session_cancel_delegates_to_backend():
    backend = _BlockingBackend()
    session, output_track, channel = _make_session(backend)

    drive_task = asyncio.create_task(
        _drive_turn(session, user_text="cancel this", start_ts=2.0)
    )
    await asyncio.wait_for(backend.started.wait(), timeout=1.0)

    await session.handle_client_event({"type": "response.cancel"})
    await asyncio.wait_for(drive_task, timeout=1.0)

    assert backend.cancelled == ["resp-cancel"]
    assert output_track.clear_calls >= 2
    assert any(event["type"] == "response.cancelled" for event in channel.messages)
