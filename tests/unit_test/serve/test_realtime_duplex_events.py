# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from sglang_omni.serve.realtime.audio_buffer import RealtimeAudioBuffer
from sglang_omni.serve.realtime.duplex_session import DuplexRealtimeSession
from sglang_omni.serve.realtime.events import (
    InputAudioBufferAppend,
    PlaybackAck,
    ResponseCancel,
    SessionClose,
    SessionConfig,
    SessionUpdate,
    parse_client_event,
)
from sglang_omni.serve.realtime.manager import RealtimeSessionManager


class _FakeWebSocket:
    def __init__(self) -> None:
        self.application_state = WebSocketState.CONNECTED
        self.client_state = WebSocketState.CONNECTED
        self.sent: list[dict[str, Any]] = []

    async def send_text(self, value: str) -> None:
        self.sent.append(json.loads(value))

    async def close(self) -> None:
        self.application_state = WebSocketState.DISCONNECTED
        self.client_state = WebSocketState.DISCONNECTED


class _BlockingWebSocket(_FakeWebSocket):
    def __init__(self) -> None:
        super().__init__()
        self.receive_started = asyncio.Event()
        self._receive_gate = asyncio.Event()

    async def receive(self) -> dict[str, Any]:
        self.receive_started.set()
        await self._receive_gate.wait()
        return {"type": "websocket.disconnect"}


class _FakeDuplex:
    def __init__(self) -> None:
        self.generation = 1
        self.response_epoch = 0
        self.is_closed = False
        self.appended: list[dict[str, Any]] = []
        self.appended_event = asyncio.Event()
        self.interrupts = 0
        self.playback: list[float] = []
        self.close_reasons: list[str] = []
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def append(self, data: dict[str, Any], *, wait_processed: bool) -> int:
        assert wait_processed is True
        self.appended.append(data)
        self.appended_event.set()
        return len(self.appended)

    async def interrupt(self) -> int:
        self.interrupts += 1
        self.response_epoch += 1
        return self.response_epoch

    async def playback_ack(self, audio_end_ms: float) -> int:
        self.playback.append(audio_end_ms)
        return len(self.playback)

    async def events(self):
        yield {
            "type": "session.created",
            "session_id": "sess-test",
            "generation": 1,
            "input_seq": 0,
            "response_epoch": 0,
            "output_seq": 1,
        }
        while True:
            event = await self._events.get()
            yield event
            if event["type"] == "session.closed":
                return

    async def close(self, *, reason: str, timeout_s: float) -> None:
        assert reason
        assert timeout_s == 10.0
        self.close_reasons.append(reason)
        self.is_closed = True
        await self._events.put(
            {
                "type": "session.closed",
                "session_id": "sess-test",
                "generation": 1,
                "input_seq": len(self.appended),
                "response_epoch": self.response_epoch,
                "output_seq": 2,
                "reason": reason,
            }
        )


class _FakeClient:
    def __init__(self, duplex: _FakeDuplex) -> None:
        self.duplex = duplex
        self.requests: list[tuple[Any, dict[str, Any]]] = []

    async def open_duplex_session(self, request: Any, **kwargs: Any) -> _FakeDuplex:
        self.requests.append((request, kwargs))
        return self.duplex


def test_realtime_audio_buffer_pops_complete_duplex_units() -> None:
    buffer = RealtimeAudioBuffer(max_bytes=16)
    buffer.append_b64(base64.b64encode(b"abcdefgh").decode())

    assert buffer.pop_left(4) == b"abcd"
    assert buffer.pop_left(4) == b"efgh"
    assert buffer.is_empty()


def test_realtime_audio_buffer_rejects_invalid_pop() -> None:
    buffer = RealtimeAudioBuffer(max_bytes=16)
    buffer.append_b64(base64.b64encode(b"ab").decode())

    with pytest.raises(ValueError, match="cannot pop"):
        buffer.pop_left(3)
    with pytest.raises(ValueError, match="non-negative"):
        buffer.pop_left(-1)


def test_duplex_realtime_events_parse_extensions() -> None:
    audio_event = parse_client_event(
        {
            "type": "input_audio_buffer.append",
            "audio": "AA==",
            "video_frames": ["frame"],
            "force_listen": True,
            "max_slice_nums": 2,
            "timestamp_ms": 1000,
        }
    )
    playback_event = parse_client_event(
        {"type": "response.audio.playback_ack", "audio_end_ms": 900}
    )
    close_event = parse_client_event({"type": "session.close", "reason": "done"})

    assert isinstance(audio_event, InputAudioBufferAppend)
    assert audio_event.video_frames == ["frame"]
    assert audio_event.force_listen is True
    assert isinstance(playback_event, PlaybackAck)
    assert playback_event.audio_end_ms == 900
    assert isinstance(close_event, SessionClose)
    assert close_event.reason == "done"


def test_duplex_realtime_event_bounds_are_validated() -> None:
    with pytest.raises(ValidationError):
        parse_client_event(
            {
                "type": "input_audio_buffer.append",
                "audio": "AA==",
                "max_slice_nums": 0,
            }
        )
    with pytest.raises(ValidationError):
        parse_client_event({"type": "response.audio.playback_ack", "audio_end_ms": -1})
    with pytest.raises(ValidationError):
        parse_client_event({"type": "session.close", "reason": "   "})
    with pytest.raises(ValidationError):
        parse_client_event({"type": "session.close", "reason": "x" * 129})


def test_realtime_manager_selects_native_duplex_adapter() -> None:
    duplex = _FakeDuplex()
    manager = RealtimeSessionManager(
        client=_FakeClient(duplex),  # type: ignore[arg-type]
        model_name="minicpmo",
        native_duplex=True,
    )

    session = manager.open(_FakeWebSocket())  # type: ignore[arg-type]

    assert isinstance(session, DuplexRealtimeSession)
    assert manager.active_sessions() == [session.session_id]


@pytest.mark.asyncio
async def test_realtime_manager_rejects_second_native_session() -> None:
    manager = RealtimeSessionManager(
        client=_FakeClient(_FakeDuplex()),  # type: ignore[arg-type]
        model_name="minicpmo",
        native_duplex=True,
    )
    first = manager.open(_FakeWebSocket())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="at capacity"):
        manager.open(_FakeWebSocket())  # type: ignore[arg-type]

    await manager.close(first.session_id)
    assert manager.active_sessions() == []


@pytest.mark.asyncio
async def test_native_duplex_realtime_aggregates_one_second_units() -> None:
    websocket = _FakeWebSocket()
    duplex = _FakeDuplex()
    session = DuplexRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=_FakeClient(duplex),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )
    session._input_task = asyncio.create_task(session._input_loop())
    pcm = np.arange(16000, dtype="<i2").tobytes()
    frame_b64 = base64.b64encode(b"frame").decode()

    await session.handle_audio_append(
        InputAudioBufferAppend(
            type="input_audio_buffer.append",
            audio=base64.b64encode(pcm).decode(),
            video_frames=[frame_b64],
            force_listen=True,
            timestamp_ms=1000,
        )
    )
    await asyncio.wait_for(duplex.appended_event.wait(), timeout=1.0)

    assert len(duplex.appended) == 1
    unit = duplex.appended[0]
    assert unit["audio_pcm16"] == pcm
    assert unit["sample_rate"] == 16000
    assert unit["video_frames"] == [frame_b64]
    assert unit["force_listen"] is True
    assert unit["timestamp_ms"] == 1000
    await session.teardown()


@pytest.mark.asyncio
async def test_native_duplex_realtime_rejects_malformed_pcm16() -> None:
    session = DuplexRealtimeSession(
        _FakeWebSocket(),  # type: ignore[arg-type]
        client=_FakeClient(_FakeDuplex()),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )

    with pytest.raises(ValueError):
        await session.handle_audio_append(
            InputAudioBufferAppend(
                type="input_audio_buffer.append",
                audio="not-base64",
            )
        )
    with pytest.raises(ValueError, match="whole number"):
        await session.handle_audio_append(
            InputAudioBufferAppend(
                type="input_audio_buffer.append",
                audio=base64.b64encode(b"x").decode(),
            )
        )

    with pytest.raises(ValueError):
        await session.handle_audio_append(
            InputAudioBufferAppend(
                type="input_audio_buffer.append",
                audio=base64.b64encode(b"\x00\x00").decode(),
                video_frames=["not-base64"],
            )
        )
    assert session.audio_buffer.is_empty()


@pytest.mark.asyncio
async def test_native_duplex_audio_backpressure_is_atomic_and_nonblocking() -> None:
    session = DuplexRealtimeSession(
        _FakeWebSocket(),  # type: ignore[arg-type]
        client=_FakeClient(_FakeDuplex()),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
        max_pending_units=1,
    )
    two_units = np.zeros(32000, dtype="<i2").tobytes()

    with pytest.raises(ValueError, match="queue is full"):
        await session.handle_audio_append(
            InputAudioBufferAppend(
                type="input_audio_buffer.append",
                audio=base64.b64encode(two_units).decode(),
            )
        )

    assert session.audio_buffer.is_empty()
    assert session._unit_queue.empty()


@pytest.mark.asyncio
async def test_native_duplex_realtime_rejects_request_provided_voice_paths() -> None:
    session = DuplexRealtimeSession(
        _FakeWebSocket(),  # type: ignore[arg-type]
        client=_FakeClient(_FakeDuplex()),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )

    with pytest.raises(ValueError, match="base64 reference audio fields"):
        await session.handle_session_update(
            SessionUpdate(
                type="session.update",
                session=SessionConfig(voice={"ref_audio_path": "/tmp/ref.wav"}),
            )
        )

    ref_audio = base64.b64encode(np.zeros(16000, dtype="<f4").tobytes()).decode()
    await session.handle_session_update(
        SessionUpdate(
            type="session.update",
            session=SessionConfig(voice={"ref_audio_base64": ref_audio}),
        )
    )
    assert session.session_object.voice == {"ref_audio_base64": ref_audio}


@pytest.mark.asyncio
async def test_native_duplex_rejects_ignored_session_update_fields() -> None:
    session = DuplexRealtimeSession(
        _FakeWebSocket(),  # type: ignore[arg-type]
        client=_FakeClient(_FakeDuplex()),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )

    with pytest.raises(ValueError, match="temperature"):
        await session.handle_session_update(
            SessionUpdate(
                type="session.update",
                session=SessionConfig(temperature=0.2),
            )
        )
    with pytest.raises(ValueError, match="supports only type"):
        await session.handle_session_update(
            SessionUpdate.model_validate(
                {
                    "type": "session.update",
                    "session": {
                        "turn_detection": {
                            "type": "model_native",
                            "threshold": 0.5,
                        }
                    },
                }
            )
        )

    payload = session._session_payload()
    assert "temperature" not in payload
    assert "max_response_output_tokens" not in payload


@pytest.mark.asyncio
async def test_native_duplex_realtime_maps_text_audio_and_listen() -> None:
    websocket = _FakeWebSocket()
    session = DuplexRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=_FakeClient(_FakeDuplex()),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )
    waveform = np.array([-1.0, 0.0, 1.0], dtype="<f4")

    await session._handle_output_delta(
        {"kind": "text", "text": "hello", "response_epoch": 0}
    )
    await session._handle_output_delta(
        {
            "kind": "audio",
            "audio": base64.b64encode(waveform.tobytes()).decode(),
            "audio_format": "float32",
            "sample_rate": 24000,
            "response_epoch": 0,
        }
    )
    await session._handle_output_delta(
        {"kind": "listen", "response_epoch": 0, "input_seq": 1}
    )

    event_types = [event["type"] for event in websocket.sent]
    assert event_types == [
        "response.created",
        "response.text.delta",
        "response.audio.delta",
        "response.output.delta",
        "response.text.done",
        "response.audio.done",
        "response.done",
    ]
    audio_event = websocket.sent[2]
    pcm = np.frombuffer(base64.b64decode(audio_event["delta"]), dtype="<i2")
    assert pcm.tolist() == [-32767, 0, 32767]
    assert audio_event["audio_start_ms"] == 0
    assert audio_event["audio_end_ms"] == pytest.approx(0.125)
    assert websocket.sent[-1]["response"]["status"] == "completed"


@pytest.mark.asyncio
async def test_native_duplex_response_output_done_finishes_only_the_response() -> None:
    websocket = _FakeWebSocket()
    session = DuplexRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=_FakeClient(_FakeDuplex()),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )

    await session._handle_output_delta(
        {"kind": "text", "text": "hello", "response_epoch": 0}
    )
    await session._handle_output_done({"response_epoch": 0, "reason": "model_turn_end"})

    assert [event["type"] for event in websocket.sent] == [
        "response.created",
        "response.text.delta",
        "response.text.done",
        "response.audio.done",
        "response.done",
    ]
    assert websocket.sent[-1]["response"]["status"] == "completed"
    assert websocket.sent[-1]["response"]["status_details"] == {
        "reason": "model_turn_end"
    }
    assert session._active_response_id is None
    assert session.closed is False


@pytest.mark.asyncio
async def test_native_duplex_zero_delta_done_has_complete_response_lifecycle() -> None:
    websocket = _FakeWebSocket()
    session = DuplexRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=_FakeClient(_FakeDuplex()),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )

    await session._handle_output_done({"response_epoch": 0})

    assert [event["type"] for event in websocket.sent] == [
        "response.created",
        "response.text.done",
        "response.audio.done",
        "response.done",
    ]
    assert websocket.sent[-1]["response"]["status_details"] == {
        "reason": "model_turn_end"
    }
    assert session.closed is False


@pytest.mark.asyncio
async def test_native_duplex_stale_output_done_does_not_finish_current_epoch() -> None:
    websocket = _FakeWebSocket()
    session = DuplexRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=_FakeClient(_FakeDuplex()),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )
    await session._ensure_response(1)
    response_id = session._active_response_id

    await session._handle_output_done({"response_epoch": 0, "reason": "stale_turn_end"})

    assert session._active_response_id == response_id
    assert [event["type"] for event in websocket.sent] == ["response.created"]


@pytest.mark.asyncio
async def test_native_duplex_cancel_fences_epoch_but_keeps_session() -> None:
    websocket = _FakeWebSocket()
    duplex = _FakeDuplex()
    session = DuplexRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=_FakeClient(duplex),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )
    session._duplex = duplex  # type: ignore[assignment]
    await session._ensure_response(0)

    await session.handle_response_cancel(ResponseCancel(type="response.cancel"))

    assert duplex.interrupts == 1
    assert duplex.is_closed is False
    assert session._active_response_epoch == 1
    assert websocket.sent[-1]["response"]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_native_duplex_cancel_fences_queued_old_epoch_output() -> None:
    class _SlowInterruptDuplex(_FakeDuplex):
        def __init__(self) -> None:
            super().__init__()
            self.interrupt_started = asyncio.Event()
            self.interrupt_release = asyncio.Event()

        async def interrupt(self) -> int:
            self.interrupt_started.set()
            await self.interrupt_release.wait()
            return await super().interrupt()

    websocket = _FakeWebSocket()
    duplex = _SlowInterruptDuplex()
    session = DuplexRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=_FakeClient(duplex),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )
    session._duplex = duplex  # type: ignore[assignment]
    await session._ensure_response(0)

    cancel_task = asyncio.create_task(
        session.handle_response_cancel(ResponseCancel(type="response.cancel"))
    )
    await duplex.interrupt_started.wait()
    stale_output_task = asyncio.create_task(
        session._handle_output_delta(
            {"kind": "text", "text": "stale", "response_epoch": 0}
        )
    )
    duplex.interrupt_release.set()
    await asyncio.gather(cancel_task, stale_output_task)

    assert not any(
        event.get("type") == "response.text.delta" and event.get("delta") == "stale"
        for event in websocket.sent
    )
    assert session._active_response_epoch == 1


@pytest.mark.asyncio
async def test_native_duplex_cancel_before_start_does_not_load_model() -> None:
    duplex = _FakeDuplex()
    client = _FakeClient(duplex)
    session = DuplexRealtimeSession(
        _FakeWebSocket(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )

    await session.handle_response_cancel(ResponseCancel(type="response.cancel"))

    assert client.requests == []
    assert session._pending_force_listen is True


@pytest.mark.asyncio
async def test_native_duplex_playback_ack_is_bounded_by_emitted_audio() -> None:
    duplex = _FakeDuplex()
    session = DuplexRealtimeSession(
        _FakeWebSocket(),  # type: ignore[arg-type]
        client=_FakeClient(duplex),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )
    session._duplex = duplex  # type: ignore[assignment]
    waveform = np.zeros(240, dtype="<f4")
    await session._handle_output_delta(
        {
            "kind": "audio",
            "audio": base64.b64encode(waveform.tobytes()).decode(),
            "audio_format": "float32",
            "sample_rate": 24000,
            "response_epoch": 0,
        }
    )

    await session.handle_playback_ack(
        PlaybackAck(type="response.audio.playback_ack", audio_end_ms=10.0)
    )
    assert duplex.playback == [10.0]
    with pytest.raises(ValueError, match="monotonic"):
        await session.handle_playback_ack(
            PlaybackAck(type="response.audio.playback_ack", audio_end_ms=9.0)
        )
    with pytest.raises(ValueError, match="exceeds emitted"):
        await session.handle_playback_ack(
            PlaybackAck(type="response.audio.playback_ack", audio_end_ms=12.0)
        )
    with pytest.raises(ValueError, match="exceeds emitted"):
        await session.handle_playback_ack(
            PlaybackAck(type="response.audio.playback_ack", audio_end_ms=10.0005)
        )


@pytest.mark.asyncio
async def test_native_duplex_playback_cursor_advances_after_backend_ack() -> None:
    duplex = _FakeDuplex()

    async def fail_playback_ack(audio_end_ms: float) -> int:
        del audio_end_ms
        raise RuntimeError("backend rejected playback cursor")

    duplex.playback_ack = fail_playback_ack  # type: ignore[method-assign]
    session = DuplexRealtimeSession(
        _FakeWebSocket(),  # type: ignore[arg-type]
        client=_FakeClient(duplex),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )
    session._duplex = duplex  # type: ignore[assignment]
    session._emitted_audio_end_ms = 10.0

    with pytest.raises(RuntimeError, match="backend rejected"):
        await session.handle_playback_ack(
            PlaybackAck(type="response.audio.playback_ack", audio_end_ms=5.0)
        )

    assert session._playback_cursor_ms == 0


@pytest.mark.asyncio
async def test_native_duplex_close_forwards_terminal_before_socket_close() -> None:
    websocket = _FakeWebSocket()
    duplex = _FakeDuplex()
    session = DuplexRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=_FakeClient(duplex),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )
    await session._ensure_started()

    await session.handle_session_close(
        SessionClose(type="session.close", reason="user_finished")
    )

    assert [event["type"] for event in websocket.sent] == [
        "session.ready",
        "session.closed",
    ]
    assert websocket.sent[-1]["reason"] == "user_finished"
    assert websocket.application_state == WebSocketState.DISCONNECTED


@pytest.mark.asyncio
async def test_native_duplex_close_waits_for_inflight_model_start() -> None:
    class _BlockingOpenClient(_FakeClient):
        def __init__(self, duplex: _FakeDuplex) -> None:
            super().__init__(duplex)
            self.open_started = asyncio.Event()
            self.open_release = asyncio.Event()

        async def open_duplex_session(self, request: Any, **kwargs: Any) -> _FakeDuplex:
            self.requests.append((request, kwargs))
            self.open_started.set()
            await self.open_release.wait()
            return self.duplex

    websocket = _FakeWebSocket()
    duplex = _FakeDuplex()
    client = _BlockingOpenClient(duplex)
    session = DuplexRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )

    start_task = asyncio.create_task(session._ensure_started())
    await client.open_started.wait()
    close_task = asyncio.create_task(
        session.handle_session_close(
            SessionClose(type="session.close", reason="user_finished")
        )
    )
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(close_task), timeout=0.01)

    client.open_release.set()
    assert await start_task is duplex
    await close_task

    assert len(client.requests) == 1
    assert duplex.close_reasons == ["user_finished"]
    assert [event["type"] for event in websocket.sent] == [
        "session.ready",
        "session.closed",
    ]


@pytest.mark.asyncio
async def test_native_duplex_closed_session_cannot_start_model() -> None:
    client = _FakeClient(_FakeDuplex())
    session = DuplexRealtimeSession(
        _FakeWebSocket(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )

    await session.handle_session_close(
        SessionClose(type="session.close", reason="closed_before_audio")
    )
    with pytest.raises(RuntimeError, match="is closed"):
        await session._ensure_started()
    assert client.requests == []


@pytest.mark.asyncio
async def test_native_duplex_terminal_is_sent_before_close_wakes_teardown() -> None:
    class _BlockingTerminalWebSocket(_FakeWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.terminal_send_started = asyncio.Event()
            self.terminal_send_release = asyncio.Event()

        async def send_text(self, value: str) -> None:
            event = json.loads(value)
            if event.get("type") == "session.closed":
                self.terminal_send_started.set()
                await self.terminal_send_release.wait()
            self.sent.append(event)

    websocket = _BlockingTerminalWebSocket()
    session = DuplexRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=_FakeClient(_FakeDuplex()),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )

    close_task = asyncio.create_task(
        session._close_before_model_start(reason="max_session_duration")
    )
    await websocket.terminal_send_started.wait()
    assert session.closed is False
    assert session._closed_signal.is_set() is False

    websocket.terminal_send_release.set()
    assert await close_task is True
    assert session.closed is True
    assert websocket.sent[-1]["type"] == "session.closed"


@pytest.mark.asyncio
async def test_native_duplex_internal_close_wakes_blocked_receive() -> None:
    websocket = _BlockingWebSocket()
    session = DuplexRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=_FakeClient(_FakeDuplex()),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )
    run_task = asyncio.create_task(session.run())
    await websocket.receive_started.wait()

    session._mark_closed()

    await asyncio.wait_for(run_task, timeout=1.0)
    await session.teardown()


@pytest.mark.asyncio
async def test_native_duplex_teardown_closes_model_after_socket_disconnect() -> None:
    websocket = _FakeWebSocket()
    await websocket.close()
    duplex = _FakeDuplex()
    session = DuplexRealtimeSession(
        websocket,  # type: ignore[arg-type]
        client=_FakeClient(duplex),  # type: ignore[arg-type]
        model_name="minicpmo",
        session_id="sess-test",
    )
    session._duplex = duplex  # type: ignore[assignment]
    session.closed = True

    await session.teardown()

    assert duplex.is_closed is True
    assert duplex.close_reasons == ["client_disconnected"]
