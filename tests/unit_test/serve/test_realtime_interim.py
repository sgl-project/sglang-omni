# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.serve.realtime.events import (
    INTERIM_PARTIAL_STYLE,
    INTERIM_TRANSCRIPTION_EVENT,
)
from sglang_omni.serve.realtime.session import _TRANSCRIPTION_PROMPT, RealtimeSession
from sglang_omni.serve.realtime.vad import Emit, VADEvent
from tests.unit_test.serve.test_realtime_barge_in import (
    FakeVAD,
    RecordingWebSocket,
    ScriptedClient,
    _chunk,
    _session,
)

StreamFactory = Callable[[], AsyncIterator[CompletionStreamChunk]]

# 16 kHz PCM16: 32000 bytes per second. Two chunks larger than
# _INTERIM_MIN_NEW_BYTES (6400) so consecutive refreshes both qualify.
_PCM_CHUNK = b"\x00\x01" * 4000  # 8000 bytes ≈ 0.25s


def _interim_stream(text: str) -> StreamFactory:
    async def stream() -> AsyncIterator[CompletionStreamChunk]:
        yield _chunk(text=text)
        yield _chunk(finish_reason="stop")

    return stream


def _failing_stream() -> StreamFactory:
    async def stream() -> AsyncIterator[CompletionStreamChunk]:
        raise RuntimeError("interim decode exploded")
        yield  # pragma: no cover - makes this an async generator

    return stream


class CapturingClient(ScriptedClient):
    def __init__(self, streams: list[StreamFactory]) -> None:
        super().__init__(streams)
        self.seen_requests: list[Any] = []

    async def completion_stream(
        self, request: Any, *, request_id: str, audio_format: str = "wav"
    ) -> AsyncIterator[CompletionStreamChunk]:
        self.seen_requests.append(request)
        async for chunk in super().completion_stream(
            request, request_id=request_id, audio_format=audio_format
        ):
            yield chunk


def _capturing_session(
    monkeypatch: pytest.MonkeyPatch, streams: list[StreamFactory]
) -> tuple[RealtimeSession, RecordingWebSocket, CapturingClient]:
    monkeypatch.setattr(
        "sglang_omni.serve.realtime.session.StreamingVAD", FakeVAD
    )
    websocket = RecordingWebSocket()
    client = CapturingClient(streams)
    session = RealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="qwen3-omni",
        supports_audio_output=True,
    )
    session.session_object.modalities = ["text", "audio"]
    return session, websocket, client


async def _append_pcm(session: RealtimeSession, chunk: bytes = _PCM_CHUNK) -> None:
    decoded_len = session.audio_buffer.append_b64(base64.b64encode(chunk).decode())
    assert decoded_len == len(chunk)


async def _wait_for(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def _interim_events(websocket: RecordingWebSocket) -> list[dict[str, Any]]:
    return [e for e in websocket.events if e["type"] == INTERIM_TRANSCRIPTION_EVENT]


async def _drive_speech_started(session: RealtimeSession) -> str:
    await _append_pcm(session)
    await session.handle_vad_emit(Emit(VADEvent.SPEECH_STARTED, 0))
    assert session.utterance_item_id is not None
    return session.utterance_item_id


@pytest.mark.asyncio
async def test_interim_emitted_during_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = _session(monkeypatch, [_interim_stream("hello")])
    session.session_object.interim_transcription = True
    session.interim_interval_s = 0.01

    item_id = await _drive_speech_started(session)
    await _wait_for(lambda: bool(_interim_events(websocket)))
    await session._stop_interim_loop()

    events = _interim_events(websocket)
    assert len(events) == 1
    assert events[0]["item_id"] == item_id
    assert events[0]["text"] == "hello"
    assert events[0]["content_index"] == 0
    assert events[0]["partial_style"] == INTERIM_PARTIAL_STYLE


@pytest.mark.asyncio
async def test_interim_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = _session(monkeypatch, [_interim_stream("hello")])
    session.interim_interval_s = 0.01

    await _drive_speech_started(session)
    await asyncio.sleep(0.1)

    assert _interim_events(websocket) == []
    assert session._interim_task is None


@pytest.mark.asyncio
async def test_interim_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = _session(
        monkeypatch, [_interim_stream("same"), _interim_stream("same")]
    )
    session.session_object.interim_transcription = True
    session.interim_interval_s = 0.01

    await _drive_speech_started(session)
    # Enough new audio for a second qualifying refresh.
    await _append_pcm(session)
    await asyncio.sleep(0.1)
    await session._stop_interim_loop()

    events = _interim_events(websocket)
    assert len(events) == 1
    assert events[0]["text"] == "same"


@pytest.mark.asyncio
async def test_interim_stopped_on_speech_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, client = _session(
        monkeypatch, [_interim_stream("hello"), _interim_stream("hello")]
    )
    session.session_object.interim_transcription = True
    session.interim_interval_s = 0.01

    await _drive_speech_started(session)
    await _wait_for(lambda: bool(_interim_events(websocket)))
    # Second refresh is in flight potential; stop must cancel + abort.
    await _append_pcm(session)
    await asyncio.sleep(0.01)
    in_flight_request_id = session._interim_request_id

    await session._stop_interim_loop()

    assert session._interim_task is None
    assert session._interim_request_id is None
    # Whatever request was live at stop time (first or second refresh) is
    # the one aborted; aborting an already-finished request is harmless.
    assert in_flight_request_id in client.aborted

    before = len(websocket.events)
    await asyncio.sleep(0.1)
    # No further interim events after the stop; the cancelled refresh
    # never reaches the wire.
    assert len(_interim_events(websocket)) == 1
    assert len(websocket.events) == before


@pytest.mark.asyncio
async def test_interim_decode_failure_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, websocket, _ = _session(
        monkeypatch, [_failing_stream(), _interim_stream("recovered")]
    )
    session.session_object.interim_transcription = True
    session.interim_interval_s = 0.01

    await _drive_speech_started(session)
    # The first refresh consumes all buffered audio and fails. A failed
    # refresh still advances last_decoded_end, so the retry only fires once
    # genuinely new audio arrives.
    await asyncio.sleep(0.05)
    await _append_pcm(session)
    await _wait_for(lambda: bool(_interim_events(websocket)))
    await session._stop_interim_loop()

    # The failed refresh never surfaces as an error event; the loop carried
    # on and delivered the next successful hypothesis.
    assert not [e for e in websocket.events if e["type"] == "error"]
    events = _interim_events(websocket)
    assert [e["text"] for e in events] == ["recovered"]


@pytest.mark.asyncio
async def test_interim_uses_verbatim_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _, client = _capturing_session(monkeypatch, [_interim_stream("hello")])
    session.session_object.interim_transcription = True
    session.interim_interval_s = 0.01

    await _drive_speech_started(session)
    await _wait_for(lambda: bool(client.seen_requests))
    await session._stop_interim_loop()

    assert client.seen_requests, "interim decode must issue an engine request"
    request = client.seen_requests[0]
    assert request.messages[0].role == "system"
    assert request.messages[0].content == _TRANSCRIPTION_PROMPT
    assert request.messages[-1].role == "user"
    assert request.output_modalities == ["text"]
