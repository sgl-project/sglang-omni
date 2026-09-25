# SPDX-License-Identifier: Apache-2.0
"""Wire contract tests for shared duplex realtime sessions on /v1/realtime."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse, WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

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
from sglang_omni.serve.realtime.schema import (
    JsonObject,
    JsonValue,
    SessionConfiguration,
)
from sglang_omni.serve.realtime.types import (
    Capabilities,
    InteractionAdapter,
    OutputSink,
    RuntimeLimits,
    Unit,
)

MODEL_NAME = "duplex-test"
SAMPLE_RATE = 16000
NATIVE_UNIT_MS = 20
UNIT_BYTES = SAMPLE_RATE * NATIVE_UNIT_MS // 1000 * 2
HALF_UNIT_BYTES = UNIT_BYTES // 2
HALF_UNIT_SAMPLES = HALF_UNIT_BYTES // 2
AUDIO_REPLY: list[OutputEvent] = [
    ResponseStarted("resp"),
    AudioDelta("resp", "item", b"\1\2"),
    TextDelta("resp", "item", "hi"),
    AudioFinished("resp", "item"),
    TextFinished("resp", "item", "hi"),
    ResponseFinished("resp", "item", "hi", True, "completed", "turn_complete"),
]


class HealthCoordinator:
    def __init__(self, is_running: bool) -> None:
        self.is_running = is_running

    def health(self) -> dict[str, bool]:
        return {"running": self.is_running}


@dataclass(kw_only=True)
class ScriptedAdapter(InteractionAdapter):
    """Records native units and emits a fixed reply while processing each one."""

    reply: list[OutputEvent] = field(default_factory=list)
    unconsumed_samples: int = 0
    is_holding_input: bool = False
    open_error: str | None = None
    close_error: str | None = None
    units: list[Unit] = field(default_factory=list)
    held_samples: int = 0
    output_sink: OutputSink | None = None
    is_closed: bool = False

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        if self.open_error is not None:
            raise RuntimeError(self.open_error)
        else:
            self.output_sink = emit

    async def process(self, unit: Unit) -> int | tuple[int, int]:
        assert self.output_sink is not None
        self.units.append(unit)
        for event in self.reply:
            await self.output_sink(event)
        if self.is_holding_input:
            self.held_samples += unit.real_samples
            return 0, 0
        else:
            return max(unit.real_samples - self.unconsumed_samples, 0)

    async def clear(self) -> int:
        held_samples, self.held_samples = self.held_samples, 0
        return held_samples

    async def close(self) -> None:
        self.is_closed = True
        if self.close_error is not None:
            raise RuntimeError(self.close_error)
        else:
            pass


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


def events_of_type(events: list[JsonObject], event_type: str) -> list[JsonObject]:
    return [event for event in events if event["type"] == event_type]


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
    adapter = ScriptedAdapter(reply=text_reply("partial", is_finished=False))
    with build_test_client(adapter).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * UNIT_BYTES, 0)
        receive_until(websocket, "response.created")
        send_event(websocket, "session.close")
        events = receive_until(websocket, "session.closed")
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()

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


def test_failed_admission_is_reported_and_releases_adapter() -> None:
    adapter = ScriptedAdapter(open_error="no capacity")
    with build_test_client(adapter).websocket_connect("/v1/realtime") as websocket:
        websocket.receive_json()
        send_event(websocket, "session.update", session={})
        error = websocket.receive_json()

    assert error["error"]["code"] == "admission_rejected"
    assert error["error"]["message"] == "no capacity"
    assert error["sglang"]["fatal"] is False
    assert adapter.is_closed is True


@pytest.mark.parametrize(
    ("adapter", "code"),
    [
        (ScriptedAdapter(unconsumed_samples=-1), "internal"),
        (
            ScriptedAdapter(reply=[TurnFailure("server_error", "model_crash", "boom")]),
            "model_crash",
        ),
    ],
)
def test_adapter_failure_closes_the_session_with_fatal_error(
    adapter: ScriptedAdapter, code: str
) -> None:
    with build_test_client(adapter).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        append_audio(websocket, b"\1" * UNIT_BYTES, 0)
        events = receive_until(websocket, "session.closed", "sglang.unit.done")

    error = events_of_type(events, "error")[0]
    assert error["sglang"]["fatal"] is True
    assert error["error"]["code"] == code
    assert events[-1]["reason"] == code


def test_adapter_cleanup_failure_replaces_closed_with_fatal_error() -> None:
    adapter = ScriptedAdapter(close_error="release failed")
    with build_test_client(adapter).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        send_event(websocket, "session.close")
        error = websocket.receive_json()
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()

    assert error["error"]["code"] == "cleanup_timeout"
    assert error["sglang"]["fatal"] is True


def test_capabilities_endpoint_reports_deployment_grant() -> None:
    limits = RuntimeLimits(max_output_events=8)
    response = build_test_client(ScriptedAdapter(), limits=limits).get(
        "/v1/realtime/capabilities"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == MODEL_NAME
    assert body["native_full_duplex"] is True
    assert body["limits"]["max_output_events"] == 8


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
