# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path

import av
import numpy as np
import pytest
import soundfile as sf
from aiortc.mediastreams import MediaStreamError

from sglang_omni.realtime.backend import MockResponseBackend
from sglang_omni.realtime.media import mono_float32, resample_linear
from sglang_omni.realtime.session import RealtimeSession, RealtimeSessionConfig
from sglang_omni.serve.webrtc_api import _consume_audio_track

TEST_AUDIO_PATH = Path(__file__).resolve().parent / "data" / "query_to_cars.wav"


class _CollectingOutputTrack:
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


class _FakeAudioTrack:
    def __init__(self, frames: list[av.AudioFrame]) -> None:
        self._frames = list(frames)

    async def recv(self) -> av.AudioFrame:
        if self._frames:
            return self._frames.pop(0)
        raise MediaStreamError


def _load_test_audio() -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(TEST_AUDIO_PATH, dtype="int16")
    return np.asarray(audio), int(sample_rate)


def _build_audio_frames(audio_i16: np.ndarray, sample_rate: int) -> list[av.AudioFrame]:
    frame_samples = sample_rate // 50  # 20 ms @ 48 kHz -> 960 samples
    frames: list[av.AudioFrame] = []
    for start in range(0, audio_i16.shape[0], frame_samples):
        chunk = audio_i16[start : start + frame_samples]
        if chunk.size == 0:
            continue
        frame = av.AudioFrame.from_ndarray(
            chunk.reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = sample_rate
        frames.append(frame)
    return frames


def _expected_session_audio(audio_i16: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_samples = sample_rate // 50
    chunks: list[np.ndarray] = []
    for start in range(0, audio_i16.shape[0], frame_samples):
        chunk = audio_i16[start : start + frame_samples]
        if chunk.size == 0:
            continue
        mono = mono_float32(chunk)
        chunks.append(resample_linear(mono, sample_rate, 16000))
    return np.concatenate(chunks)


@pytest.mark.asyncio
async def test_real_wav_audio_pipeline_round_trips_through_manual_mock_echo():
    audio_i16, sample_rate = _load_test_audio()
    backend = MockResponseBackend(
        audio_mode="echo",
        output_modalities=("audio",),
        inter_chunk_delay_s=0.0,
        chunk_duration_s=0.1,
    )
    output_track = _CollectingOutputTrack()
    session = RealtimeSession(
        session_id="session-audio-pipeline",
        backend=backend,
        output_track=output_track,
        config=RealtimeSessionConfig(),
    )

    await session.handle_client_event({"type": "input_audio_buffer.start"})
    await _consume_audio_track(
        _FakeAudioTrack(_build_audio_frames(audio_i16, sample_rate)), session
    )
    await session.handle_client_event({"type": "input_audio_buffer.commit"})

    assert session.active_task is not None
    await asyncio.wait_for(session.active_task, timeout=2.0)

    assert output_track.enqueue_calls
    assert {call_sample_rate for _, call_sample_rate in output_track.enqueue_calls} == {
        16000
    }

    echoed_audio = np.concatenate([audio for audio, _ in output_track.enqueue_calls])
    expected_user_audio = _expected_session_audio(audio_i16, sample_rate)
    expected_echo = backend._condition_echo_waveform(expected_user_audio)

    np.testing.assert_allclose(echoed_audio, expected_echo, atol=1e-5)
    assert np.max(np.abs(echoed_audio)) <= 0.35 + 1e-5
    assert np.sqrt(np.mean(np.square(echoed_audio))) > 0.01
