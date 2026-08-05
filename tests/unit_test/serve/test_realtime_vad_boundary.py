# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import io
import wave
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from sglang_omni.serve.realtime.audio_buffer import RealtimeAudioBuffer
from sglang_omni.serve.realtime.events import (
    InputAudioBufferAppend,
    InputAudioBufferClear,
)
from sglang_omni.serve.realtime.session import RealtimeSession
from sglang_omni.serve.realtime.vad import (
    VAD_FRAME_SAMPLES,
    VAD_SAMPLE_RATE,
    Emit,
    VADEvent,
    offsets_to_ms,
)

BOUNDARY = [
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_started",
]


def _pcm(frames: int, amp: int = 0) -> bytes:
    return amp.to_bytes(2, "little", signed=True) * (VAD_FRAME_SAMPLES * frames)


def _b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode()


def _wav_pcm(uri: str) -> bytes:
    with wave.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])), "rb") as wf:
        return wf.readframes(wf.getnframes())


def _wav_samples(uri: str) -> int:
    return len(_wav_pcm(uri)) // 2


def _speechish(frames: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(frames * VAD_FRAME_SAMPLES) / VAD_SAMPLE_RATE
    f0 = 120.0 + 10.0 * np.sin(2 * np.pi * 3 * t)
    x = (
        0.4 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
        + 0.08 * np.sin(2 * np.pi * 800 * t)
        + 0.05 * rng.normal(0.0, 1.0, t.size)
    )
    x = np.clip(x * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)), -1.0, 1.0)
    return (x * 30000).astype(np.int16).tobytes()


class FakeVAD:
    def __init__(self, emits: list[Emit]) -> None:
        self._emits = list(emits)
        self.reset_calls = 0

    def process(self, pcm_bytes: bytes) -> list[Emit]:
        out, self._emits = self._emits, []
        return out

    def reset(self) -> None:
        self.reset_calls += 1


async def _empty(*_a: Any, **_k: Any):
    if False:  # pragma: no cover
        yield None


async def _stop(session: RealtimeSession) -> None:
    session.closed = True
    if session.queue_drainer is not None and not session.queue_drainer.done():
        session.queue_drainer.cancel()
        await asyncio.gather(session.queue_drainer, return_exceptions=True)


def _session(
    vad: FakeVAD | None = None,
) -> tuple[RealtimeSession, list[dict[str, Any]], list[tuple[str, str]]]:
    sent: list[dict[str, Any]] = []
    commits: list[tuple[str, str]] = []
    client = MagicMock()
    client.abort = AsyncMock()
    client.completion_stream = MagicMock(side_effect=_empty)

    if vad is None:
        session = RealtimeSession(
            MagicMock(), client=client, model_name="m", session_id="s"
        )
    else:
        with patch("sglang_omni.serve.realtime.session.StreamingVAD", return_value=vad):
            session = RealtimeSession(
                MagicMock(), client=client, model_name="m", session_id="s"
            )
        session.vad = vad

    orig_put = session.response_queue.put

    async def _put(item: tuple[str, str]) -> None:
        commits.append(item)
        await orig_put(item)

    session.response_queue.put = _put  # type: ignore[method-assign]
    session.send = AsyncMock(side_effect=lambda e: sent.append(e))
    return session, sent, commits


async def _append(session: RealtimeSession, pcm: bytes) -> None:
    await session.handle_audio_append(
        InputAudioBufferAppend(type="input_audio_buffer.append", audio=_b64(pcm))
    )


def _by_type(sent: list[dict[str, Any]], typ: str) -> list[dict[str, Any]]:
    return [e for e in sent if e["type"] == typ]


@pytest.mark.asyncio
async def test_adjacent_turn_retains_suffix_and_second_commit():
    stop1 = VAD_FRAME_SAMPLES
    start2 = 4 * VAD_FRAME_SAMPLES
    stop2 = 5 * VAD_FRAME_SAMPLES
    vad = FakeVAD(
        [
            Emit(VADEvent.SPEECH_STARTED, 0),
            Emit(VADEvent.SPEECH_STOPPED, stop1),
            Emit(VADEvent.SPEECH_STARTED, start2),
        ]
    )
    session, sent, commits = _session(vad)
    second = _pcm(1, 9000)
    await _append(session, _pcm(1, 12000) + _pcm(3) + second + _pcm(1))

    assert [e["type"] for e in sent] == BOUNDARY
    assert vad.reset_calls == 0 and len(commits) == 1
    assert _wav_samples(commits[0][1]) == stop1
    assert session.buffer_origin_samples == stop1 and session.vad_origin_samples == 0
    assert second in bytes(session.audio_buffer.buf)

    started, stopped = _by_type(sent, BOUNDARY[0]), _by_type(sent, BOUNDARY[1])[0]
    assert started[0]["audio_start_ms"] == 0
    assert stopped["audio_end_ms"] == offsets_to_ms(stop1)
    assert started[1]["audio_start_ms"] == offsets_to_ms(start2)
    assert (
        started[0]["audio_start_ms"]
        < stopped["audio_end_ms"]
        < started[1]["audio_start_ms"]
    )

    vad._emits = [Emit(VADEvent.SPEECH_STOPPED, stop2)]
    await _append(session, _pcm(1))
    committed = _by_type(sent, BOUNDARY[2])
    assert len(committed) == len(commits) == 2
    assert committed[0]["item_id"] != committed[1]["item_id"]
    assert _wav_samples(commits[1][1]) == stop2 - start2
    assert session.buffer_origin_samples == stop2 and vad.reset_calls == 0
    await _stop(session)


@pytest.mark.asyncio
async def test_silero_adjacent_turns_preserve_second_speech():
    second = _speechish(5, seed=1)
    session, sent, commits = _session()
    await _append(session, _speechish(5, seed=0) + _pcm(16) + second)

    assert [e["type"] for e in sent] == BOUNDARY and len(commits) == 1
    started, stopped = _by_type(sent, BOUNDARY[0]), _by_type(sent, BOUNDARY[1])[0]
    assert (
        started[0]["audio_start_ms"]
        < stopped["audio_end_ms"]
        < started[1]["audio_start_ms"]
    )
    assert session.vad_origin_samples == 0
    assert (
        session.buffer_origin_samples
        == stopped["audio_end_ms"] * VAD_SAMPLE_RATE // 1000
    )
    assert bytes(session.audio_buffer.buf).endswith(second)
    await _stop(session)


@pytest.mark.asyncio
async def test_silero_silence_threshold_across_appends():
    """Silence threshold can accumulate across client appends (network streaming)."""
    first = _speechish(5, seed=0)
    second = _speechish(5, seed=1)
    session, sent, commits = _session()
    reset_calls = 0
    orig_reset = session.vad.reset

    def _counting_reset() -> None:
        nonlocal reset_calls
        reset_calls += 1
        orig_reset()

    session.vad.reset = _counting_reset  # type: ignore[method-assign]

    await _append(session, first + _pcm(15))
    assert [e["type"] for e in sent] == [BOUNDARY[0]]
    assert commits == [] and reset_calls == 0

    await _append(session, _pcm(1) + second)
    assert [e["type"] for e in sent] == BOUNDARY
    assert len(commits) == 1 and reset_calls == 0
    assert session.vad_origin_samples == 0
    assert second in bytes(session.audio_buffer.buf)

    started, stopped = _by_type(sent, BOUNDARY[0]), _by_type(sent, BOUNDARY[1])[0]
    assert (
        started[0]["audio_start_ms"]
        < stopped["audio_end_ms"]
        < started[1]["audio_start_ms"]
    )

    await _append(session, _pcm(16))
    assert len(commits) == 2 and reset_calls == 0
    assert second in _wav_pcm(commits[1][1])
    await _stop(session)


def test_tail_zero_returns_empty_bytes():
    buf = RealtimeAudioBuffer()
    buf.buf.extend(_pcm(2, 1000))
    assert buf.tail(0) == b""
    assert buf.tail(2) == bytes(buf.buf[-2:])


@pytest.mark.asyncio
async def test_empty_append_does_not_advance_after_partial_commit():
    """Empty append must not re-feed retained suffix through VAD."""
    second = _speechish(5, seed=1)
    session, sent, commits = _session()
    await _append(session, _speechish(5, seed=0) + _pcm(16) + second)

    assert [e["type"] for e in sent] == BOUNDARY and len(commits) == 1
    assert not session.audio_buffer.is_empty()
    samples_before = session.vad.samples_consumed
    origin_before = session.buffer_origin_samples
    retained = bytes(session.audio_buffer.buf)

    await _append(session, b"")

    assert session.vad.samples_consumed == samples_before
    assert session.buffer_origin_samples == origin_before
    assert bytes(session.audio_buffer.buf) == retained
    assert len(commits) == 1
    await _stop(session)


@pytest.mark.asyncio
async def test_clear_resets_vad_timeline():
    vad = FakeVAD([])
    session, sent, _ = _session(vad)
    session.audio_buffer.append_b64(_b64(_pcm(2, 1000)))
    await session.handle_audio_clear(
        InputAudioBufferClear(type="input_audio_buffer.clear")
    )
    assert session.audio_buffer.is_empty()
    assert (
        session.buffer_origin_samples
        == session.vad_origin_samples
        == 2 * VAD_FRAME_SAMPLES
    )
    assert vad.reset_calls == 1 and sent[-1]["type"] == "input_audio_buffer.cleared"
