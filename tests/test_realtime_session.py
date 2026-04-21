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


class _FakeVadEvent:
    def __init__(
        self, *, speech_started: bool = False, speech_stopped: bool = False
    ) -> None:
        self.speech_started = speech_started
        self.speech_stopped = speech_stopped


class _FakeVad:
    def __init__(self) -> None:
        self.speaking = False
        self.last_frame_count = 2
        self.last_voiced_frame_count = 0
        self.last_speech_ratio = 0.0
        self._call_count = 0

    def measure_level(self, audio: np.ndarray) -> float:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio))))

    def process(self, _audio: np.ndarray) -> _FakeVadEvent:
        self._call_count += 1
        phase = ((self._call_count - 1) % 3) + 1
        if phase == 1:
            self.last_voiced_frame_count = 2
            self.last_speech_ratio = 1.0
            return _FakeVadEvent()
        if phase == 2:
            self.speaking = True
            self.last_voiced_frame_count = 2
            self.last_speech_ratio = 1.0
            return _FakeVadEvent(speech_started=True)

        self.speaking = False
        self.last_voiced_frame_count = 0
        self.last_speech_ratio = 0.0
        return _FakeVadEvent(speech_stopped=True)

    def reset(self) -> None:
        self.speaking = False
        self.last_frame_count = 0
        self.last_voiced_frame_count = 0
        self.last_speech_ratio = 0.0
        self._call_count = 0


def _make_session(
    backend,
) -> tuple[RealtimeSession, _FakeOutputTrack, _FakeChannel]:
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
            ),
        ),
    )
    session.vad = _FakeVad()
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


async def _drive_text_turn(
    session: RealtimeSession,
    *,
    user_text: str,
) -> None:
    await session.handle_client_event(
        {
            "type": "conversation.item.create",
            "item": {"role": "user", "content": user_text},
        }
    )
    await session.handle_client_event({"type": "response.create"})

    task = session.active_task
    assert task is not None
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_realtime_session_runs_turns_with_fake_backend_and_history():
    backend = _ScriptedBackend()
    session, output_track, channel = _make_session(backend)

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    await session.handle_video_frame(frame, timestamp=1.0)
    await session.handle_video_frame(frame, timestamp=1.6)

    await _drive_turn(session, user_text="describe this", start_ts=2.0)
    await _drive_turn(session, user_text="follow up", start_ts=4.0)

    assert len(backend.turns) == 2
    assert backend.turns[0].user_text == "describe this"
    assert backend.turns[0].recent_video is not None
    assert backend.turns[0].recent_video.shape[0] == 2
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
    assert event_types.count("input_audio_buffer.chunk_received") == 2
    assert event_types.count("input_audio_buffer.speech_started") == 2
    assert event_types.count("input_audio_buffer.speech_stopped") == 2
    assert event_types.count("input_video_buffer.frame_received") == 1
    assert event_types.count("turn.prepared") == 2
    assert event_types.count("response.created") == 2
    assert event_types.count("response.output_text.delta") == 2
    assert event_types.count("response.output_audio.delta") == 2
    assert event_types.count("response.done") == 2

    frame_event = next(
        event
        for event in channel.messages
        if event["type"] == "input_video_buffer.frame_received"
    )
    assert frame_event["frame_count"] == 1
    assert frame_event["buffered_frames"] == 1

    audio_chunk_events = [
        event
        for event in channel.messages
        if event["type"] == "input_audio_buffer.chunk_received"
    ]
    assert all(event["sample_count"] > 0 for event in audio_chunk_events)
    assert all(event["sample_rate"] == 16000 for event in audio_chunk_events)
    assert all(event["rms"] >= 0.0 for event in audio_chunk_events)
    assert all("dc_offset" in event for event in audio_chunk_events)
    assert all("frame_count" in event for event in audio_chunk_events)
    assert all("voiced_frame_count" in event for event in audio_chunk_events)
    assert all("speech_ratio" in event for event in audio_chunk_events)
    assert all("speaking_before" in event for event in audio_chunk_events)
    assert all("speaking_after" in event for event in audio_chunk_events)
    assert [event["chunk_count"] for event in audio_chunk_events] == [1, 4]

    turn_events = [
        event for event in channel.messages if event["type"] == "turn.prepared"
    ]
    assert all(event["audio_sample_count"] > 0 for event in turn_events)
    assert turn_events[0]["video_frame_count"] == 2
    assert turn_events[0]["video_fps"] is not None


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


@pytest.mark.asyncio
async def test_realtime_session_auto_vad_barge_in_cancels_and_queues_next_turn():
    backend = _BlockingBackend()
    session, output_track, channel = _make_session(backend)

    speech = np.full(1600, 0.2, dtype=np.float32)
    silence = np.zeros(1600, dtype=np.float32)

    await session.handle_client_event(
        {
            "type": "conversation.item.create",
            "item": {"role": "user", "content": "first request"},
        }
    )
    await session.handle_audio_chunk(speech, 16000, timestamp=1.0)
    await session.handle_audio_chunk(speech, 16000, timestamp=1.1)
    await session.handle_audio_chunk(silence, 16000, timestamp=1.2)

    await asyncio.wait_for(backend.started.wait(), timeout=1.0)
    assert len(backend.turns) == 1

    await session.handle_client_event(
        {
            "type": "conversation.item.create",
            "item": {"role": "user", "content": "second request"},
        }
    )
    await session.handle_audio_chunk(speech, 16000, timestamp=2.0)
    await session.handle_audio_chunk(speech, 16000, timestamp=2.1)
    await session.handle_audio_chunk(silence, 16000, timestamp=2.2)

    deadline = asyncio.get_running_loop().time() + 1.0
    while (
        session.active_task is not None and asyncio.get_running_loop().time() < deadline
    ):
        await asyncio.sleep(0.01)

    assert backend.cancelled == ["resp-cancel"]
    assert len(backend.turns) == 2
    assert backend.turns[1].user_text == "second request"
    assert output_track.clear_calls >= 2
    assert any(
        event["type"] == "response.cancelled" and event["reason"] == "barge_in"
        for event in channel.messages
    )


@pytest.mark.asyncio
async def test_realtime_session_supports_manual_push_to_talk_commit():
    backend = _ScriptedBackend()
    session, output_track, channel = _make_session(backend)

    await session.handle_client_event({"type": "input_audio_buffer.start"})
    await session.handle_audio_chunk(
        np.full(1600, 0.2, dtype=np.float32), 16000, timestamp=1.0
    )
    await session.handle_audio_chunk(
        np.full(1600, 0.1, dtype=np.float32), 16000, timestamp=1.1
    )
    await session.handle_client_event({"type": "input_audio_buffer.commit"})

    task = session.active_task
    assert task is not None
    await asyncio.wait_for(task, timeout=1.0)

    assert session.turn_mode == "manual"
    assert session.manual_recording is False
    assert len(backend.turns) == 1
    np.testing.assert_allclose(
        backend.turns[0].user_audio,
        np.concatenate(
            [
                np.full(1600, 0.2, dtype=np.float32),
                np.full(1600, 0.1, dtype=np.float32),
            ]
        ),
    )
    assert any(
        event["type"] == "input_audio_buffer.manual_started"
        for event in channel.messages
    )
    assert any(
        event["type"] == "input_audio_buffer.manual_committed"
        and event["empty"] is False
        for event in channel.messages
    )
    assert any(event["type"] == "turn.prepared" for event in channel.messages)
    assert any(event["type"] == "response.done" for event in channel.messages)
    assert len(output_track.enqueue_calls) == 1


@pytest.mark.asyncio
async def test_realtime_session_supports_text_only_turns_with_history():
    backend = _ScriptedBackend()
    session, output_track, channel = _make_session(backend)

    await _drive_text_turn(session, user_text="hello there")
    await _drive_text_turn(session, user_text="follow up")

    assert len(backend.turns) == 2
    assert backend.turns[0].user_text == "hello there"
    assert backend.turns[0].user_audio is None
    assert backend.turns[1].history == [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "answer-1"},
    ]
    assert session.history == [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "answer-1"},
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "answer-2"},
    ]
    assert len(output_track.enqueue_calls) == 2

    turn_events = [
        event for event in channel.messages if event["type"] == "turn.prepared"
    ]
    assert len(turn_events) == 2
    assert all(event["audio_sample_count"] == 0 for event in turn_events)
    assert all(event["audio_sample_rate"] is None for event in turn_events)


@pytest.mark.asyncio
async def test_realtime_session_can_switch_between_vad_and_manual_modes():
    backend = _ScriptedBackend()
    session, _output_track, channel = _make_session(backend)

    assert session.turn_mode == "vad"

    await session.handle_client_event(
        {
            "type": "session.update",
            "session": {"audio": {"input_mode": "manual"}},
        }
    )

    assert session.turn_mode == "manual"
    assert session.manual_recording is False
    assert channel.messages[-1]["type"] == "session.updated"
    assert channel.messages[-1]["session"]["audio"]["input_mode"] == "manual"

    await session.handle_audio_chunk(
        np.full(1600, 0.2, dtype=np.float32), 16000, timestamp=1.0
    )
    assert session.active_task is None

    await session.handle_client_event(
        {
            "type": "session.update",
            "session": {"audio": {"input_mode": "vad"}},
        }
    )

    assert session.turn_mode == "vad"
    assert session.manual_recording is False
    assert channel.messages[-1]["type"] == "session.updated"
    assert channel.messages[-1]["session"]["audio"]["input_mode"] == "vad"
