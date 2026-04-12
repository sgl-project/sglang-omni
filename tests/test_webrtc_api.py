# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from sglang_omni.serve.webrtc_api import (
    RealtimeSessionManager,
    SessionHandle,
    _consume_audio_track,
)


class _FakeSession:
    def __init__(self) -> None:
        self.session_id = "session-1"
        self.calls: list[tuple[np.ndarray, int]] = []

    async def handle_audio_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        timestamp: float | None = None,
    ) -> None:
        self.calls.append((np.asarray(audio), sample_rate))

    async def close(self) -> None:
        return


class _FakeAudioFrame:
    def __init__(self, audio: np.ndarray, sample_rate: int = 48000) -> None:
        self._audio = audio
        self.sample_rate = sample_rate

    def to_ndarray(self) -> np.ndarray:
        return self._audio


class _FakeAudioTrack:
    def __init__(self, frames: list[_FakeAudioFrame]) -> None:
        self._frames = list(frames)

    async def recv(self) -> _FakeAudioFrame:
        if self._frames:
            return self._frames.pop(0)
        raise RuntimeError("done")


class _FakePeerConnection:
    def __init__(self, on_close: asyncio.Event | None = None) -> None:
        self._on_close = on_close

    async def close(self) -> None:
        if self._on_close is not None:
            self._on_close.set()


class _FakeBackend:
    model_name = "fake-model"
    capabilities = type(
        "Caps",
        (),
        {
            "accepts_audio_input": True,
            "accepts_video_input": False,
            "returns_text": False,
            "returns_audio": False,
            "supports_cancel": False,
        },
    )()


@pytest.mark.asyncio
async def test_consume_audio_track_accepts_pyav_frames_without_format_kwarg():
    session = _FakeSession()
    audio = np.array([[1000, -1000, 500, -500]], dtype=np.int16)
    track = _FakeAudioTrack([_FakeAudioFrame(audio, sample_rate=24000)])

    await _consume_audio_track(track, session)

    assert len(session.calls) == 1
    np.testing.assert_array_equal(session.calls[0][0], audio)
    assert session.calls[0][1] == 24000


@pytest.mark.asyncio
async def test_realtime_session_manager_close_swallows_consumer_failures():
    manager = RealtimeSessionManager(
        backend_factory=lambda *_args: _FakeBackend(),
        default_model="fake-model",
    )
    handle = await manager.create(
        model=None,
        instructions=None,
        max_new_tokens=32,
        output_text=False,
        vad=None,
    )

    async def _boom() -> None:
        raise RuntimeError("consumer failed")

    failed_task = asyncio.create_task(_boom())
    await asyncio.sleep(0)
    handle.consumer_tasks.append(failed_task)

    await manager.close(handle.session.session_id)


@pytest.mark.asyncio
async def test_realtime_session_manager_close_closes_peer_before_waiting_on_consumers():
    manager = RealtimeSessionManager(
        backend_factory=lambda *_args: _FakeBackend(),
        default_model="fake-model",
    )

    released = asyncio.Event()

    async def _wait_for_peer_close() -> None:
        await released.wait()

    session = _FakeSession()
    handle = SessionHandle(
        session=session,
        peer_connection=_FakePeerConnection(on_close=released),
        consumer_tasks=[asyncio.create_task(_wait_for_peer_close())],
    )
    manager._sessions[session.session_id] = handle

    await asyncio.wait_for(manager.close(session.session_id), timeout=1.0)
