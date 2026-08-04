# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import io
import wave
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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


def _pcm16_chunk(num_frames: int, *, amplitude: int = 0) -> bytes:
    sample = amplitude.to_bytes(2, "little", signed=True)
    return sample * (VAD_FRAME_SAMPLES * num_frames)


def _b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def _wav_num_samples(data_uri: str) -> int:
    _, b64 = data_uri.split(",", 1)
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as wf:
        return wf.getnframes()


class FakeVAD:
    def __init__(self, emits: list[Emit]) -> None:
        self._emits = emits
        self.reset_calls = 0
        self.samples_consumed = 0

    def process(self, pcm_bytes: bytes) -> list[Emit]:
        self.samples_consumed += len(pcm_bytes) // 2
        emits = self._emits
        self._emits = []
        return emits

    def reset(self) -> None:
        self.reset_calls += 1
        self.samples_consumed = 0


async def _empty_async_iter(*_args: Any, **_kwargs: Any):
    if False:  # pragma: no cover
        yield None
    return


async def _stop_session(session: RealtimeSession) -> None:
    session.closed = True
    if session.queue_drainer is not None and not session.queue_drainer.done():
        session.queue_drainer.cancel()
        await asyncio.gather(session.queue_drainer, return_exceptions=True)


def _make_session(fake_vad: FakeVAD) -> tuple[RealtimeSession, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    websocket = MagicMock()
    client = MagicMock()
    client.abort = AsyncMock()
    client.completion_stream = MagicMock(side_effect=_empty_async_iter)

    with patch("sglang_omni.serve.realtime.session.StreamingVAD", return_value=fake_vad):
        session = RealtimeSession(
            websocket,
            client=client,
            model_name="test-model",
            session_id="sess_test",
        )
    session.vad = fake_vad
    session.send = AsyncMock(side_effect=lambda event: sent.append(event))
    return session, sent


@pytest.mark.asyncio
async def test_adjacent_turn_in_single_chunk_keeps_suffix_audio():
    first_start = 0
    first_stop = VAD_FRAME_SAMPLES
    prefix_pad = 300 * VAD_SAMPLE_RATE // 1000
    second_start = max(0, 18 * VAD_FRAME_SAMPLES - VAD_FRAME_SAMPLES - prefix_pad)

    fake_vad = FakeVAD(
        [
            Emit(VADEvent.SPEECH_STARTED, first_start),
            Emit(VADEvent.SPEECH_STOPPED, first_stop),
            Emit(VADEvent.SPEECH_STARTED, second_start),
        ]
    )
    session, sent = _make_session(fake_vad)
    committed_payloads: list[tuple[str, str]] = []
    original_put = session.response_queue.put

    async def _capture_put(item: tuple[str, str]) -> None:
        committed_payloads.append(item)
        await original_put(item)

    session.response_queue.put = _capture_put  # type: ignore[method-assign]

    pcm = _pcm16_chunk(1, amplitude=12000) + _pcm16_chunk(16) + _pcm16_chunk(
        1, amplitude=12000
    )
    assert len(pcm) == 18 * VAD_FRAME_SAMPLES * 2

    await session.handle_audio_append(
        InputAudioBufferAppend(type="input_audio_buffer.append", audio=_b64(pcm))
    )

    types = [e["type"] for e in sent]
    assert types == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "input_audio_buffer.speech_started",
    ]

    assert fake_vad.reset_calls == 0
    committed = next(e for e in sent if e["type"] == "input_audio_buffer.committed")
    assert len(committed_payloads) == 1
    _, payload = committed_payloads[0]
    assert _wav_num_samples(payload) == first_stop - first_start

    expected_suffix_samples = 18 * VAD_FRAME_SAMPLES - first_stop
    assert session.audio_buffer.num_samples == expected_suffix_samples
    assert session.buffer_origin_samples == first_stop
    assert session.vad_origin_samples == 0

    started_events = [
        e for e in sent if e["type"] == "input_audio_buffer.speech_started"
    ]
    assert started_events[0]["audio_start_ms"] == offsets_to_ms(first_start)
    assert started_events[1]["audio_start_ms"] == offsets_to_ms(second_start)
    assert started_events[1]["audio_start_ms"] > started_events[0]["audio_start_ms"]

    assert session.utterance_start_byte is not None
    assert 0 <= session.utterance_start_byte < session.audio_buffer.num_bytes
    assert session.utterance_item_id != committed["item_id"]

    await _stop_session(session)


@pytest.mark.asyncio
async def test_second_turn_commit_uses_retained_suffix():
    first_stop = VAD_FRAME_SAMPLES
    second_start = 4 * VAD_FRAME_SAMPLES
    second_stop = 5 * VAD_FRAME_SAMPLES

    fake_vad = FakeVAD(
        [
            Emit(VADEvent.SPEECH_STARTED, 0),
            Emit(VADEvent.SPEECH_STOPPED, first_stop),
            Emit(VADEvent.SPEECH_STARTED, second_start),
        ]
    )
    session, sent = _make_session(fake_vad)
    committed_payloads: list[tuple[str, str]] = []
    original_put = session.response_queue.put

    async def _capture_put(item: tuple[str, str]) -> None:
        committed_payloads.append(item)
        await original_put(item)

    session.response_queue.put = _capture_put  # type: ignore[method-assign]

    pcm = _pcm16_chunk(6, amplitude=8000)
    await session.handle_audio_append(
        InputAudioBufferAppend(type="input_audio_buffer.append", audio=_b64(pcm))
    )

    fake_vad._emits = [Emit(VADEvent.SPEECH_STOPPED, second_stop)]
    await session.handle_audio_append(
        InputAudioBufferAppend(
            type="input_audio_buffer.append", audio=_b64(_pcm16_chunk(1))
        )
    )

    committed = [e for e in sent if e["type"] == "input_audio_buffer.committed"]
    assert len(committed) == 2
    assert len(committed_payloads) == 2

    first_item, first_payload = committed_payloads[0]
    second_item, second_payload = committed_payloads[1]
    assert first_item != second_item
    assert _wav_num_samples(first_payload) == first_stop
    assert _wav_num_samples(second_payload) == second_stop - second_start
    assert session.buffer_origin_samples == second_stop
    assert fake_vad.reset_calls == 0

    await _stop_session(session)


@pytest.mark.asyncio
async def test_audio_clear_still_resets_vad_timeline():
    fake_vad = FakeVAD([])
    session, sent = _make_session(fake_vad)
    session.audio_buffer.append_b64(_b64(_pcm16_chunk(2, amplitude=1000)))
    session.buffer_origin_samples = 0
    session.vad_origin_samples = 0
    fake_vad.samples_consumed = 2 * VAD_FRAME_SAMPLES

    await session.handle_audio_clear(
        InputAudioBufferClear(type="input_audio_buffer.clear")
    )

    assert session.audio_buffer.is_empty()
    assert session.buffer_origin_samples == 2 * VAD_FRAME_SAMPLES
    assert session.vad_origin_samples == session.buffer_origin_samples
    assert fake_vad.reset_calls == 1
    assert sent[-1]["type"] == "input_audio_buffer.cleared"


def test_audio_buffer_drop_prefix_keeps_suffix():
    buf = RealtimeAudioBuffer()
    buf.buf.extend(b"abcdefgh")
    buf.drop_prefix(3)
    assert bytes(buf.buf) == b"defgh"
    buf.drop_prefix(100)
    assert buf.is_empty()
