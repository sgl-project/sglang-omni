# SPDX-License-Identifier: Apache-2.0

import asyncio

import numpy as np
import pytest

from sglang_omni.realtime.backend import MockResponseBackend, TurnContext


@pytest.mark.asyncio
async def test_mock_response_backend_streams_text_and_audio():
    backend = MockResponseBackend(
        response_text="Mock backend replayed the captured utterance.",
        inter_chunk_delay_s=0.0,
        total_duration_s=0.2,
        chunk_duration_s=0.1,
    )
    user_audio = np.linspace(-0.25, 0.25, 32, dtype=np.float32)
    turn = TurnContext(
        session_id="session-1",
        history=[],
        instructions=None,
        user_text="hello",
        user_audio=user_audio,
        user_audio_sample_rate=16000,
        recent_video=None,
        recent_video_fps=None,
    )

    events = [event async for event in backend.stream_response(turn)]
    audio_events = [event for event in events if event.type == "audio_chunk"]

    assert events[0].type == "response_started"
    assert any(event.type == "text_delta" for event in events)
    assert audio_events
    assert [event.sample_rate for event in audio_events] == [16000]
    np.testing.assert_allclose(
        np.concatenate([event.audio for event in audio_events]),
        user_audio,
    )
    assert events[-1].type == "done"
    assert events[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_mock_response_backend_falls_back_to_tone_without_audio():
    backend = MockResponseBackend(
        inter_chunk_delay_s=0.0,
        total_duration_s=0.05,
        chunk_duration_s=0.05,
        sample_rate=24000,
    )
    turn = TurnContext(
        session_id="session-1",
        history=[],
        instructions=None,
        user_text="hello",
        user_audio=None,
        user_audio_sample_rate=None,
        recent_video=None,
        recent_video_fps=None,
    )

    events = [event async for event in backend.stream_response(turn)]
    audio_events = [event for event in events if event.type == "audio_chunk"]

    assert audio_events
    assert audio_events[0].sample_rate == 24000
    assert np.any(audio_events[0].audio != 0.0)


@pytest.mark.asyncio
async def test_mock_response_backend_cancel_stops_stream():
    backend = MockResponseBackend(
        inter_chunk_delay_s=0.05,
        total_duration_s=0.4,
        chunk_duration_s=0.1,
    )
    turn = TurnContext(
        session_id="session-1",
        history=[],
        instructions=None,
        user_text="hello",
        user_audio=np.zeros(32, dtype=np.float32),
        user_audio_sample_rate=16000,
        recent_video=None,
        recent_video_fps=None,
    )

    events = []

    async def _collect():
        async for event in backend.stream_response(turn):
            events.append(event)
            if event.type == "response_started":
                await backend.cancel(event.response_id)

    await asyncio.wait_for(_collect(), timeout=1.0)

    assert events[0].type == "response_started"
    assert events[-1].type == "done"
    assert events[-1].finish_reason == "cancelled"
