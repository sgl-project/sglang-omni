# SPDX-License-Identifier: Apache-2.0
"""Wire contract tests for shared duplex realtime sessions on /v1/realtime."""

from __future__ import annotations

import base64
import logging
from typing import Literal

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse, WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from sglang_omni.admission import QueueFullError
from sglang_omni.client.client import Client
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
    TurnFailure,
)
from sglang_omni.serve.realtime.schema import JsonObject, JsonValue
from sglang_omni.serve.realtime.types import Capabilities, RuntimeLimits
from tests.unit_test.fixtures.realtime_fakes import (
    MODEL_NAME,
    HealthCoordinator,
    ScriptedAdapter,
)

SAMPLE_RATE = 16000
NATIVE_UNIT_MS = 20
UNIT_BYTES = SAMPLE_RATE * NATIVE_UNIT_MS // 1000 * 2
HALF_UNIT_BYTES = UNIT_BYTES // 2
HALF_UNIT_SAMPLES = HALF_UNIT_BYTES // 2
CLEANUP_TIMEOUT_SECONDS = 0.05
OPEN_TIMEOUT_SECONDS = 0.05
REALTIME_LOGGER_PREFIX = "sglang_omni.serve.realtime."
AUDIO_REPLY: list[OutputEvent] = [
    ResponseStarted("resp"),
    AudioDelta("resp", "item", b"\1\2"),
    TextDelta("resp", "item", "hi"),
    AudioFinished("resp", "item"),
    TextFinished("resp", "item", "hi"),
    ResponseFinished("resp", "item", "hi", True, "completed", "turn_complete"),
]


def text_reply(text: str, *, is_finished: bool) -> list[OutputEvent]:
    events: list[OutputEvent] = [
        ResponseStarted("resp"),
        TextDelta("resp", "item", text),
    ]
    if is_finished:
        events.append(
            ResponseFinished("resp", "item", text, False, "completed", "turn_complete")
        )
    else:
        pass
    return events


def build_test_client(
    adapter: ScriptedAdapter,
    *,
    capabilities: Capabilities | None = None,
    limits: RuntimeLimits | None = None,
    max_connections: int = 4,
    is_running: bool = True,
) -> TestClient:
    deployment = RealtimeDeployment(
        capabilities=capabilities or Capabilities(),
        adapter_factory=lambda: adapter,
        limits=limits or RuntimeLimits(),
        max_connections=max_connections,
    )
    app = create_app(
        Client(HealthCoordinator(is_running)),
        model_name=MODEL_NAME,
        realtime_deployment=deployment,
    )
    return TestClient(app)


def send_event(
    websocket: WebSocketTestSession, event_type: str, **fields: JsonValue
) -> None:
    websocket.send_json(
        {"event_id": f"client_{event_type}", "type": event_type, **fields}
    )


def append_audio(
    websocket: WebSocketTestSession,
    pcm: bytes,
    seq: int,
    t_start_ms: float | None = None,
) -> None:
    metadata: JsonObject = {"seq": seq}
    if t_start_ms is not None:
        metadata["t_start_ms"] = t_start_ms
    else:
        pass
    send_event(
        websocket,
        "input_audio_buffer.append",
        audio=base64.b64encode(pcm).decode("ascii"),
        sglang=metadata,
    )


def receive_until(
    websocket: WebSocketTestSession, *event_types: str
) -> list[JsonObject]:
    events: list[JsonObject] = []
    while not events or events[-1]["type"] not in event_types:
        events.append(websocket.receive_json())
    return events


def open_session(websocket: WebSocketTestSession) -> JsonObject:
    assert websocket.receive_json()["type"] == "session.created"
    send_event(websocket, "session.update", session={})
    return receive_until(websocket, "session.updated")[-1]


def receive_close_frame(websocket: WebSocketTestSession) -> tuple[int, str]:
    """Expect the server to close the WebSocket next; return its close code and reason."""
    with pytest.raises(WebSocketDisconnect) as close_exception:
        websocket.receive_json()
    return close_exception.value.code, close_exception.value.reason


def events_of_type(events: list[JsonObject], event_type: str) -> list[JsonObject]:
    return [event for event in events if event["type"] == event_type]


def realtime_failure_records(
    caplog: pytest.LogCaptureFixture,
) -> list[logging.LogRecord]:
    """Warning and error records of the realtime layer, so that the info records of a session opening or closing do not count whatever level the process logs at."""
    return [
        record
        for record in caplog.records
        if record.name.startswith(REALTIME_LOGGER_PREFIX)
        and record.levelno >= logging.WARNING
    ]


def drained_media(drained: JsonObject) -> tuple[float, float, float, float]:
    return (
        drained["accepted_end_ms"],
        drained["consumed_ms"],
        drained["discarded_ms"],
        drained["padding_ms"],
    )


def test_session_is_created_before_negotiation_and_granted_on_update() -> None:
    with build_test_client(ScriptedAdapter()).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        created = websocket.receive_json()
        send_event(websocket, "session.update", session={})
        updated = websocket.receive_json()

    assert created["type"] == "session.created"
    assert created["session"]["model"] == MODEL_NAME
    assert created["session"]["sglang"]["granted"] is None
    assert updated["type"] == "session.updated"
    assert updated["client_event_id"] == "client_session.update"
    assert updated["session"]["id"] == created["session"]["id"]
    assert updated["session"]["sglang"]["granted"]["native_full_duplex"] is True


def test_audio_before_update_is_rejected_without_closing_the_session() -> None:
    with build_test_client(ScriptedAdapter()).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        websocket.receive_json()
        append_audio(websocket, b"\0" * UNIT_BYTES, 0)
        error = websocket.receive_json()
        send_event(websocket, "session.update", session={})
        updated = websocket.receive_json()

    assert error["error"]["code"] == "invalid_state"
    assert error["sglang"]["fatal"] is False
    assert updated["type"] == "session.updated"


def test_append_requires_contiguous_seq_and_media_time() -> None:
    with build_test_client(ScriptedAdapter()).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        open_session(websocket)
        append_audio(websocket, b"\0" * HALF_UNIT_BYTES, 0, t_start_ms=0.0)
        first = websocket.receive_json()
        append_audio(websocket, b"\0" * HALF_UNIT_BYTES, 0)
        repeated_seq = websocket.receive_json()
        append_audio(websocket, b"\0" * HALF_UNIT_BYTES, 1, t_start_ms=0.0)
        overlapping_time = websocket.receive_json()
        append_audio(websocket, b"\0" * UNIT_BYTES, 1, t_start_ms=10.0)
        second = websocket.receive_json()

    assert (first["seq"], first["accepted_end_ms"]) == (0, 10.0)
    assert repeated_seq["error"]["code"] == "invalid_state"
    assert overlapping_time["error"]["code"] == "invalid_state"
    assert (second["seq"], second["accepted_end_ms"]) == (1, 30.0)


def test_input_over_budget_is_rejected_and_the_same_seq_can_retry() -> None:
    limits = RuntimeLimits(max_input_bytes=UNIT_BYTES * 3 // 4)
    with build_test_client(ScriptedAdapter(), limits=limits).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * HALF_UNIT_BYTES, 0)
        websocket.receive_json()
        append_audio(websocket, b"\1" * HALF_UNIT_BYTES, 1)
        overflow = websocket.receive_json()
        send_event(websocket, "input_audio_buffer.clear")
        websocket.receive_json()
        append_audio(websocket, b"\1" * HALF_UNIT_BYTES, 1)
        retried = websocket.receive_json()

    assert overflow["error"]["code"] == "buffer_overflow"
    assert overflow["sglang"]["fatal"] is False
    assert (retried["type"], retried["seq"]) == ("sglang.input_audio.accepted", 1)


def test_end_of_input_flushes_tail_and_reports_drained_media() -> None:
    adapter = ScriptedAdapter()
    with build_test_client(adapter).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * (UNIT_BYTES + HALF_UNIT_BYTES), 0)
        send_event(websocket, "sglang.input_audio.end")
        events = receive_until(websocket, "sglang.input_audio.drained")

    unit_times = [
        event["sglang"]["media_time"]
        for event in events_of_type(events, "sglang.unit.done")
    ]
    assert unit_times == [
        {"t_start_ms": 0.0, "duration_ms": 20.0},
        {"t_start_ms": 20.0, "duration_ms": 10.0},
    ]
    assert [unit.eos for unit in adapter.units] == [False, True]
    ended = events_of_type(events, "sglang.input_audio.ended")[0]
    assert ended["tail_policy"] == "flush"
    assert drained_media(events[-1]) == (30.0, 30.0, 0.0, 0.0)


def test_input_is_final_after_end() -> None:
    with build_test_client(ScriptedAdapter()).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * HALF_UNIT_BYTES, 0)
        send_event(websocket, "sglang.input_audio.end")
        receive_until(websocket, "sglang.input_audio.drained")
        append_audio(websocket, b"\1" * HALF_UNIT_BYTES, 1)
        late_append = websocket.receive_json()
        send_event(websocket, "sglang.input_audio.end")
        second_end = websocket.receive_json()

    assert late_append["error"]["code"] == "invalid_state"
    assert second_end["error"]["code"] == "invalid_state"


def test_pad_tail_policy_pads_last_unit_and_reports_padding() -> None:
    adapter = ScriptedAdapter()
    capabilities = Capabilities(tail_policy="pad")
    with build_test_client(adapter, capabilities=capabilities).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * HALF_UNIT_BYTES, 0)
        send_event(websocket, "sglang.input_audio.end")
        drained = receive_until(websocket, "sglang.input_audio.drained")[-1]

    tail = adapter.units[-1]
    assert tail.pcm == b"\1" * HALF_UNIT_BYTES + b"\0" * HALF_UNIT_BYTES
    assert tail.real_samples == HALF_UNIT_SAMPLES
    assert drained_media(drained) == (10.0, 10.0, 0.0, 10.0)


def test_reject_tail_policy_refuses_partial_unit_at_end() -> None:
    capabilities = Capabilities(tail_policy="reject")
    with build_test_client(
        ScriptedAdapter(), capabilities=capabilities
    ).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * HALF_UNIT_BYTES, 0)
        websocket.receive_json()
        send_event(websocket, "sglang.input_audio.end")
        error = websocket.receive_json()

    assert error["error"]["code"] == "invalid_state"
    assert error["sglang"]["fatal"] is False


def test_cleared_audio_is_reported_as_discarded_media() -> None:
    with build_test_client(ScriptedAdapter()).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * HALF_UNIT_BYTES, 0)
        websocket.receive_json()
        send_event(websocket, "input_audio_buffer.clear")
        cleared = websocket.receive_json()
        append_audio(websocket, b"\1" * HALF_UNIT_BYTES, 1)
        send_event(websocket, "sglang.input_audio.end")
        drained = receive_until(websocket, "sglang.input_audio.drained")[-1]

    assert cleared["sglang"]["discarded_ms"] == 10.0
    assert drained_media(drained) == (20.0, 10.0, 10.0, 0.0)


def test_clear_discards_audio_held_by_the_adapter() -> None:
    adapter = ScriptedAdapter(is_holding_input=True)
    with build_test_client(adapter).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * UNIT_BYTES, 0)
        receive_until(websocket, "sglang.unit.done")
        send_event(websocket, "input_audio_buffer.clear")
        cleared = websocket.receive_json()
        send_event(websocket, "sglang.input_audio.end")
        drained = receive_until(websocket, "sglang.input_audio.drained")[-1]

    assert cleared["sglang"]["discarded_ms"] == 20.0
    assert drained_media(drained) == (20.0, 0.0, 20.0, 0.0)


def test_partial_adapter_consumption_is_reported_as_discarded_media() -> None:
    adapter = ScriptedAdapter(unconsumed_samples=HALF_UNIT_SAMPLES)
    with build_test_client(adapter).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * UNIT_BYTES, 0)
        send_event(websocket, "sglang.input_audio.end")
        drained = receive_until(websocket, "sglang.input_audio.drained")[-1]

    assert drained_media(drained) == (20.0, 10.0, 10.0, 0.0)


def test_adapter_output_carries_producing_unit_metadata() -> None:
    adapter = ScriptedAdapter(reply=text_reply("hello", is_finished=True))
    with build_test_client(adapter).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * UNIT_BYTES, 0)
        events = receive_until(websocket, "sglang.unit.done")

    delta = events_of_type(events, "response.output_text.delta")[0]
    done = events_of_type(events, "response.done")[0]
    assert delta["delta"] == "hello"
    assert delta["sglang"]["unit_id"] == "unit_0"
    assert delta["sglang"]["media_time"] == {"t_start_ms": 0.0, "duration_ms": 20.0}
    assert done["response"]["status"] == "completed"
    assert done["response"]["output"][0]["content"] == [
        {"type": "output_text", "text": "hello"}
    ]


def test_audio_session_projects_audio_and_transcript_events() -> None:
    adapter = ScriptedAdapter(reply=AUDIO_REPLY)
    capabilities = Capabilities(output_modalities=("audio",))
    with build_test_client(adapter, capabilities=capabilities).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * UNIT_BYTES, 0)
        websocket.receive_json()
        events = receive_until(websocket, "sglang.unit.done")[:-1]

    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_audio.delta",
        "response.output_audio_transcript.delta",
        "response.output_audio.done",
        "response.output_audio_transcript.done",
        "response.done",
    ]
    assert events[1]["delta"] == base64.b64encode(b"\1\2").decode("ascii")
    assert events[4]["transcript"] == "hi"
    assert events[5]["response"]["output"][0]["content"] == [
        {"type": "output_audio", "transcript": "hi"}
    ]


def test_text_session_drops_ungranted_audio_output() -> None:
    adapter = ScriptedAdapter(reply=AUDIO_REPLY)
    capabilities = Capabilities(output_modalities=("text", "audio"))
    with build_test_client(adapter, capabilities=capabilities).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * UNIT_BYTES, 0)
        websocket.receive_json()
        events = receive_until(websocket, "sglang.unit.done")[:-1]

    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.done",
        "response.done",
    ]
    assert events[3]["response"]["output"][0]["content"] == [
        {"type": "output_text", "text": "hi"}
    ]


def test_client_close_cancels_visible_response_then_closes_socket() -> None:
    """Client-requested closure must include client_closed in the WebSocket close frame."""
    adapter = ScriptedAdapter(reply=text_reply("partial", is_finished=False))
    with build_test_client(adapter).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * UNIT_BYTES, 0)
        receive_until(websocket, "response.created")
        send_event(websocket, "session.close")
        events = receive_until(websocket, "session.closed")
        close_frame = receive_close_frame(websocket)

    assert close_frame == (1000, "client_closed")
    text_done = events_of_type(events, "response.output_text.done")[0]
    response_done = events_of_type(events, "response.done")[0]
    assert text_done["text"] == "partial"
    assert response_done["response"]["status"] == "cancelled"
    assert response_done["response"]["status_details"] == {"reason": "client_closed"}
    assert events[-1]["client_event_id"] == "client_session.close"
    assert adapter.is_closed is True


@pytest.mark.parametrize(
    ("frame", "code", "event_id"),
    [
        (b"\0", "invalid_request", None),
        ("{not json", "invalid_request", None),
        ('{"type": "session.close"}', "invalid_request", None),
        ('{"event_id": "e", "type": "conversation.item.create"}', "not_supported", "e"),
        (
            '{"event_id": "e", "type": "input_audio_buffer.commit"}',
            "not_applicable",
            "e",
        ),
        (
            '{"event_id": "e", "type": "input_audio_buffer.append",'
            ' "audio": "***", "sglang": {"seq": 0}}',
            "invalid_request",
            "e",
        ),
        (
            '{"event_id": "e", "type": "input_audio_buffer.append",'
            ' "audio": "AA==", "sglang": {"seq": 0}}',
            "invalid_request",
            "e",
        ),
    ],
)
def test_malformed_or_unsupported_frames_are_non_fatal(
    frame: str | bytes, code: str, event_id: str | None
) -> None:
    with build_test_client(ScriptedAdapter()).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        open_session(websocket)
        if isinstance(frame, bytes):
            websocket.send_bytes(frame)
        else:
            websocket.send_text(frame)
        error = websocket.receive_json()
        send_event(websocket, "session.update", session={})
        updated = websocket.receive_json()

    assert error["error"]["code"] == code
    assert error["error"]["event_id"] == event_id
    assert error["sglang"]["fatal"] is False
    assert updated["type"] == "session.updated"


def test_full_admission_queue_releases_adapter_and_allows_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A capacity rejection releases the adapter and leaves the socket retryable."""
    adapter = ScriptedAdapter(open_error=QueueFullError())
    with build_test_client(adapter).websocket_connect("/v1/realtime") as websocket:
        created = websocket.receive_json()
        send_event(websocket, "session.update", event_id="rejected_update", session={})
        error = websocket.receive_json()
        assert adapter.is_closed is True
        adapter.open_error = None
        send_event(websocket, "session.update", event_id="retry_update", session={})
        updated = websocket.receive_json()
        assert adapter.is_closed is False
        send_event(websocket, "session.close")
        closed = receive_until(websocket, "session.closed")[-1]

    assert error["error"]["code"] == "admission_rejected"
    assert error["error"]["message"] == QueueFullError.MESSAGE
    assert error["error"]["event_id"] == "rejected_update"
    assert error["sglang"]["fatal"] is False
    assert updated["type"] == "session.updated"
    assert updated["client_event_id"] == "retry_update"
    assert updated["session"]["id"] == created["session"]["id"]
    assert closed["reason"] == "client_closed"
    assert adapter.is_closed is True
    assert not realtime_failure_records(caplog)


def test_backend_open_failure_is_fatal_and_keeps_its_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Backend initialization failures must not look like retryable admission."""
    adapter = ScriptedAdapter(open_error=RuntimeError("backend initialization failed"))
    with build_test_client(adapter).websocket_connect("/v1/realtime") as websocket:
        websocket.receive_json()
        send_event(websocket, "session.update", event_id="failed_update", session={})
        error = websocket.receive_json()
        assert error["error"]["code"] == "internal"
        assert error["sglang"]["fatal"] is True
        events = [error, *receive_until(websocket, "session.closed")]
        assert receive_close_frame(websocket) == (1011, "internal")

    errors = events_of_type(events, "error")
    assert len(errors) == 1
    assert errors[0]["error"]["message"] == "backend initialization failed"
    assert errors[0]["error"]["event_id"] == "failed_update"
    assert events[-1]["reason"] == "internal"
    assert adapter.is_closed is True
    failure_records = realtime_failure_records(caplog)
    assert len(failure_records) == 1
    assert failure_records[0].levelno == logging.ERROR
    assert failure_records[0].exc_info


@pytest.mark.parametrize(
    ("open_error", "is_holding_open", "expected_message"),
    [
        (None, True, "session open timed out"),
        (TimeoutError("backend open timed out"), False, "backend open timed out"),
        (TimeoutError(), False, "session open timed out"),
    ],
)
def test_open_timeout_is_fatal_and_has_a_message(
    caplog: pytest.LogCaptureFixture,
    open_error: TimeoutError | None,
    is_holding_open: bool,
    expected_message: str,
) -> None:
    """An opening timeout must not become a retryable rejection with no message."""
    adapter = ScriptedAdapter(open_error=open_error, is_holding_open=is_holding_open)
    with build_test_client(
        adapter, limits=RuntimeLimits(cleanup_timeout_s=OPEN_TIMEOUT_SECONDS)
    ).websocket_connect("/v1/realtime") as websocket:
        websocket.receive_json()
        send_event(websocket, "session.update", event_id="timed_out_update", session={})
        error = websocket.receive_json()
        assert error["error"]["code"] == "session_open_timeout"
        assert error["sglang"]["fatal"] is True
        events = [error, *receive_until(websocket, "session.closed")]
        assert receive_close_frame(websocket) == (1011, "session_open_timeout")

    errors = events_of_type(events, "error")
    assert len(errors) == 1
    assert errors[0]["error"]["message"] == expected_message
    assert errors[0]["error"]["event_id"] == "timed_out_update"
    assert events[-1]["reason"] == "session_open_timeout"
    assert adapter.is_closed is True
    failure_records = realtime_failure_records(caplog)
    assert len(failure_records) == 1
    assert failure_records[0].levelno == logging.ERROR
    assert not failure_records[0].exc_info


@pytest.mark.parametrize(
    ("adapter", "code", "close_code", "close_reason"),
    [
        (ScriptedAdapter(unconsumed_samples=-1), "internal", 1011, "internal"),
        *[
            (
                ScriptedAdapter(reply=[TurnFailure("server_error", code, "limit")]),
                code,
                1008,
                code,
            )
            for code in (
                "session_idle_timeout",
                "output_budget_exhausted",
                "context_limit",
            )
        ],
        (
            ScriptedAdapter(reply=[TurnFailure("server_error", "model_crash", "boom")]),
            "model_crash",
            1011,
            "model_crash",
        ),
        (
            ScriptedAdapter(
                reply=[TurnFailure("server_error", "context_exhausted", "context full")]
            ),
            "context_exhausted",
            1008,
            "context_exhausted",
        ),
        (
            ScriptedAdapter(reply=[TurnFailure("server_error", "é" * 100, "boom")]),
            "é" * 100,
            1011,
            "é" * 61,
        ),
    ],
)
def test_adapter_failure_closes_the_session_with_fatal_error(
    adapter: ScriptedAdapter, code: str, close_code: int, close_reason: str
) -> None:
    """Adapter failures must not close the socket with 1000 normal closure."""
    with build_test_client(adapter).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * UNIT_BYTES, 0)
        events = receive_until(websocket, "session.closed", "sglang.unit.done")
        close_frame = receive_close_frame(websocket)

    error = events_of_type(events, "error")[0]
    assert error["sglang"]["fatal"] is True
    assert error["error"]["code"] == code
    assert events[-1]["reason"] == code
    assert close_frame == (close_code, close_reason)


def test_a_response_outgrowing_max_history_chars_closes_the_session_with_context_limit() -> (
    None
):
    adapter = ScriptedAdapter(reply=text_reply("x" * 9, is_finished=True))
    with build_test_client(
        adapter, limits=RuntimeLimits(max_history_chars=8)
    ).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * UNIT_BYTES, 0)
        events = receive_until(websocket, "session.closed")
        close_frame = receive_close_frame(websocket)

    error = events_of_type(events, "error")[0]
    assert error["sglang"]["fatal"] is True
    assert error["error"]["code"] == "context_limit"
    assert events[-1]["reason"] == "context_limit"
    assert close_frame == (1008, "context_limit")


@pytest.mark.parametrize(
    ("close_reason", "should_stall_close"),
    [("client_closed", False), ("context_exhausted", True)],
    ids=["client-exception", "context-timeout"],
)
def test_adapter_cleanup_failure_replaces_closed_with_fatal_error(
    close_reason: Literal["client_closed", "context_exhausted"],
    should_stall_close: bool,
) -> None:
    """Cleanup failures must close the socket with 1011 and cleanup_timeout."""
    adapter = ScriptedAdapter(
        close_error="release failed", should_stall_close=should_stall_close
    )
    if close_reason == "context_exhausted":
        adapter.reply = [
            TurnFailure("server_error", "context_exhausted", "context full")
        ]
    else:
        pass
    limits = RuntimeLimits(cleanup_timeout_s=CLEANUP_TIMEOUT_SECONDS)
    events: list[JsonObject] = []
    with build_test_client(adapter, limits=limits).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        open_session(websocket)
        if close_reason == "context_exhausted":
            append_audio(websocket, b"\1" * UNIT_BYTES, 0)
        else:
            send_event(websocket, "session.close")
        with pytest.raises(WebSocketDisconnect) as close_exception:
            while True:
                events.append(websocket.receive_json())

    error = events_of_type(events, "error")[-1]
    assert error["error"]["code"] == "cleanup_timeout"
    assert error["sglang"]["fatal"] is True
    assert not events_of_type(events, "session.closed")
    assert adapter.is_closed
    assert (close_exception.value.code, close_exception.value.reason) == (
        1011,
        "cleanup_timeout",
    )


def test_capabilities_endpoint_reports_deployment_grant() -> None:
    limits = RuntimeLimits(max_output_bytes=8192)
    response = build_test_client(ScriptedAdapter(), limits=limits).get(
        "/v1/realtime/capabilities"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == MODEL_NAME
    assert body["native_full_duplex"] is True
    assert body["limits"]["max_output_bytes"] == 8192
    assert "max_output_events" not in body["limits"]


def test_capabilities_endpoint_is_unavailable_until_running() -> None:
    response = build_test_client(ScriptedAdapter(), is_running=False).get(
        "/v1/realtime/capabilities"
    )

    assert response.status_code == 503


@pytest.mark.parametrize("query", ["?model=other-model", "?session_id=sess_1"])
def test_foreign_model_or_resume_request_is_refused(query: str) -> None:
    client = build_test_client(ScriptedAdapter())

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/v1/realtime{query}") as websocket:
            websocket.receive_json()

    assert exc_info.value.code == 1008


def test_connections_over_capacity_are_denied_before_upgrade() -> None:
    client = build_test_client(ScriptedAdapter(), max_connections=1)

    with client.websocket_connect("/v1/realtime") as websocket:
        websocket.receive_json()
        with pytest.raises(WebSocketDenialResponse) as exc_info:
            with client.websocket_connect("/v1/realtime"):
                pass

    assert exc_info.value.status_code == 503


@pytest.mark.parametrize(
    ("first_update", "codes"),
    [
        (None, ["session_update_timeout"]),
        (
            {"sglang": {"sampling": {"temperature": 3.5}}},
            ["invalid_request", "session_update_timeout"],
        ),
    ],
)
def test_connection_that_never_opens_is_closed_and_frees_its_slot(
    first_update: JsonObject | None, codes: list[str]
) -> None:
    """Sessions that never finish initial configuration must close with 1008 policy violation."""
    client = build_test_client(
        ScriptedAdapter(),
        limits=RuntimeLimits(session_update_timeout_s=0.1),
        max_connections=1,
    )

    with client.websocket_connect("/v1/realtime") as websocket:
        if first_update is not None:
            send_event(websocket, "session.update", session=first_update)
        else:
            pass
        events = receive_until(websocket, "session.closed")
        close_frame = receive_close_frame(websocket)
        with client.websocket_connect("/v1/realtime") as reopened_socket:
            reopened = reopened_socket.receive_json()

    errors = events_of_type(events, "error")
    assert [error["error"]["code"] for error in errors] == codes
    assert errors[-1]["sglang"]["fatal"] is True
    assert reopened["type"] == "session.created"
    assert close_frame == (1008, "session_update_timeout")
