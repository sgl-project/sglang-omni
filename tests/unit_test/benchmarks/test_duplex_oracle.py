import base64
import copy

import pytest

from benchmarks.duplex.oracle import evaluate_trace

GRANTED = {
    "interaction": "native",
    "native_full_duplex": True,
    "proactive_output": False,
    "turn_control": [None],
    "client_commit": False,
    "input_modalities": ["audio"],
    "output_modalities": ["audio"],
    "input_audio_format": {"type": "audio/pcm", "rate": 16000},
    "output_audio_format": {"type": "audio/pcm", "rate": 22050},
    "native_unit_ms": 80,
    "first_unit_ms": 80,
    "microturn_ms": None,
    "tail_policy": "pad",
    "supports_server_interrupt": False,
    "supports_truncate": False,
    "supports_resume": False,
    "partial_style": "append_only",
    "pressure_policy": "reject",
    "strict_order": True,
    "limits": {
        "max_input_bytes": 1920000,
        "max_input_chunks": 128,
        "max_output_bytes": 4194304,
        "max_output_events": 256,
        "max_responses": 128,
        "max_segments": 256,
        "max_history_chars": 65536,
        "session_timeout_s": 240,
        "cleanup_timeout_s": 30,
    },
    "rejections": [],
}


def trace_fixture() -> list[dict]:
    pcm = base64.b64encode(b"\x00\x00" * 1280).decode()
    output = base64.b64encode(b"\x01\x00" * 1764).decode()
    result = []

    def add(time_s: float, direction: str, typ: str, **event: object) -> None:
        result.append(
            {
                "direction": direction,
                "time_s": 100 + time_s,
                "event": {"type": typ, "event_id": f"event_{len(result)}", **event},
            }
        )

    def unit_done(time_s: float, index: int, duration_ms: float) -> None:
        add(
            time_s,
            "receive",
            "sglang.unit.done",
            unit_id=f"unit_{index}",
            sglang={
                "unit_id": f"unit_{index}",
                "chunk_seq": 0,
                "media_time": {
                    "t_start_ms": index * 80,
                    "duration_ms": duration_ms,
                },
            },
        )

    add(
        0.00,
        "receive",
        "session.created",
        session={"id": "session_A", "sglang": {"granted": None}},
    )
    add(
        0.01,
        "send",
        "session.update",
        event_id="configure",
        session={"output_modalities": ["audio"]},
    )
    add(
        0.02,
        "receive",
        "session.updated",
        client_event_id="configure",
        session={"id": "session_A", "sglang": {"granted": copy.deepcopy(GRANTED)}},
    )
    add(
        0.10,
        "send",
        "input_audio_buffer.append",
        event_id="input0",
        audio=pcm,
        sglang={"seq": 0, "t_start_ms": 0},
    )
    add(
        0.11,
        "receive",
        "sglang.input_audio.accepted",
        seq=0,
        accepted_end_ms=80,
        client_event_id="input0",
    )
    add(
        0.12,
        "receive",
        "response.created",
        response={"id": "response0", "status": "in_progress"},
    )
    add(
        0.13,
        "receive",
        "response.output_audio.delta",
        response_id="response0",
        item_id="item0",
        output_index=0,
        content_index=0,
        delta=output,
    )
    unit_done(0.135, 0, 80)
    add(
        0.18,
        "send",
        "input_audio_buffer.append",
        event_id="input1",
        audio=pcm,
        sglang={"seq": 1, "t_start_ms": 80},
    )
    add(
        0.19,
        "receive",
        "sglang.input_audio.accepted",
        seq=1,
        accepted_end_ms=160,
        client_event_id="input1",
    )
    add(
        0.20,
        "receive",
        "response.output_audio.delta",
        response_id="response0",
        item_id="item0",
        output_index=0,
        content_index=0,
        delta=output,
    )
    unit_done(0.21, 1, 80)
    add(0.23, "send", "sglang.input_audio.end", event_id="end")
    add(
        0.24,
        "receive",
        "sglang.input_audio.ended",
        accepted_end_ms=160,
        tail_policy="pad",
        client_event_id="end",
    )
    add(
        0.25,
        "receive",
        "response.output_audio.done",
        response_id="response0",
    )
    add(
        0.26,
        "receive",
        "response.done",
        response={
            "id": "response0",
            "status": "completed",
            "status_details": {"reason": "stop"},
        },
    )
    unit_done(0.264, 2, 0)
    add(
        0.27,
        "receive",
        "sglang.input_audio.drained",
        accepted_end_ms=160,
        consumed_ms=160,
        discarded_ms=0,
        padding_ms=0,
        client_event_id="end",
    )
    add(0.28, "send", "session.close", event_id="close")
    add(
        0.29,
        "receive",
        "session.closed",
        client_event_id="close",
        reason="client_closed",
    )
    return result


def event_of(records: list[dict], typ: str) -> dict:
    return next(record["event"] for record in records if record["event"]["type"] == typ)


def test_current_session_protocol_without_epoch_or_resource_receipt() -> None:
    trace = trace_fixture()
    assert all("epoch" not in r["event"].get("sglang", {}) for r in trace)
    assert "cancel_is_noop" not in GRANTED
    assert "held" not in event_of(trace, "session.closed")
    result = evaluate_trace(trace, scenario="continuous")
    assert result["violations"] == []
    assert result["status"] == "pass"


def test_known_good_trace_and_hand_calculated_metrics() -> None:
    result = evaluate_trace(trace_fixture(), scenario="continuous")
    assert result["status"] == "pass"
    assert result["violations"] == []
    assert all(result["coverage"].values())
    assert result["metrics"]["input_audio_s"] == pytest.approx(0.16)
    assert result["metrics"]["output_audio_s"] == pytest.approx(0.16)
    assert result["metrics"]["first_audio_packet_s"] == pytest.approx(0.03)
    assert result["metrics"]["audio_packet_gap_max_s"] == pytest.approx(0.07)
    assert result["metrics"]["drain_after_eos_s"] == pytest.approx(0.04)
    assert result["metrics"]["close_ack_s"] == pytest.approx(0.01)
    assert result["metrics"]["input_output_overlap"] is True
    assert set(result["coverage"]) == {"input_output_overlap"}


@pytest.mark.parametrize(
    ("typ", "key", "value", "violation"),
    [
        ("session.updated", "client_event_id", "unknown", "matching prior client"),
        (
            "input_audio_buffer.append",
            "sglang",
            {"seq": 1},
            "sequence is not contiguous",
        ),
        ("input_audio_buffer.append", "sglang", {"seq": True}, "valid integer"),
        (
            "input_audio_buffer.append",
            "sglang",
            {"seq": 0, "t_start_ms": 80},
            "media time is not sample-contiguous",
        ),
        ("input_audio_buffer.append", "audio", "?!", "base64"),
        ("input_audio_buffer.append", "audio", "AA==", "whole nonempty samples"),
        ("response.output_audio.delta", "delta", "", "whole nonempty samples"),
        ("response.output_audio.delta", "item_id", None, "audio missing item"),
        ("response.output_audio.delta", "response_id", "foreign", "unknown response"),
        ("response.output_audio.delta", "content_index", True, "valid integer"),
        ("sglang.input_audio.accepted", "seq", 9, "sequence mismatch"),
        ("sglang.input_audio.accepted", "accepted_end_ms", 81, "duration mismatch"),
        ("sglang.input_audio.ended", "accepted_end_ms", 159, "duration mismatch"),
        ("sglang.input_audio.drained", "consumed_ms", 80, "consumed duration mismatch"),
        ("sglang.input_audio.drained", "discarded_ms", 1, "was discarded"),
        ("sglang.input_audio.drained", "padding_ms", 80, "padding mismatch"),
        ("sglang.unit.done", "unit_id", "bogus", "invalid native unit ID"),
        ("session.closed", "reason", "disconnect", "unexpected session close reason"),
        ("session.closed", "reason", None, "unexpected session close reason"),
        ("session.closed", "client_event_id", "unknown", "matching prior client"),
    ],
)
def test_wire_mutations_fail(typ: str, key: str, value: object, violation: str) -> None:
    trace = trace_fixture()
    event_of(trace, typ)[key] = value
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert any(violation in item for item in result["violations"])


def test_control_receipt_does_not_require_sglang_envelope() -> None:
    trace = trace_fixture()
    event_of(trace, "sglang.input_audio.accepted").pop("sglang", None)
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "pass"
    assert result["violations"] == []


@pytest.mark.parametrize(
    ("unit_id", "violation"),
    [
        ("unit_9", "native unit receipts are not contiguous from zero"),
        ("unit_0", "native unit receipts are not increasing"),
    ],
)
def test_native_unit_receipt_identity(unit_id: str, violation: str) -> None:
    trace = trace_fixture()
    last = [
        record["event"]
        for record in trace
        if record["event"]["type"] == "sglang.unit.done"
    ][-1]
    last["unit_id"] = unit_id
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert violation in result["violations"]


@pytest.mark.parametrize(
    "typ",
    [
        "session.created",
        "session.updated",
        "sglang.input_audio.ended",
        "sglang.input_audio.drained",
        "session.close",
        "session.closed",
        "response.done",
        "response.output_audio.done",
        "sglang.input_audio.accepted",
    ],
)
def test_missing_required_steps_fail(typ: str) -> None:
    trace = [record for record in trace_fixture() if record["event"]["type"] != typ]
    assert evaluate_trace(trace, scenario="continuous")["status"] == "fail"


@pytest.mark.parametrize(
    "typ",
    [
        "response.done",
        "response.output_audio.done",
        "session.updated",
        "sglang.input_audio.accepted",
    ],
)
def test_duplicate_receipts_or_terminals_fail(typ: str) -> None:
    trace = trace_fixture()
    index = next(i for i, record in enumerate(trace) if record["event"]["type"] == typ)
    duplicate = copy.deepcopy(trace[index])
    duplicate["event"]["event_id"] = "duplicate_semantics"
    trace.insert(index, duplicate)
    assert evaluate_trace(trace, scenario="continuous")["status"] == "fail"


@pytest.mark.parametrize("value", [False, 1, "true"])
def test_native_capability_requires_actual_true(value: object) -> None:
    trace = trace_fixture()
    event_of(trace, "session.updated")["session"]["sglang"]["granted"][
        "native_full_duplex"
    ] = value
    assert evaluate_trace(trace, scenario="continuous")["status"] == "fail"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("supports_resume", True),
        ("supports_truncate", True),
        ("supports_server_interrupt", True),
        ("output_modalities", ["text"]),
        ("output_audio_format", {"type": "audio/pcm", "rate": 16000}),
    ],
)
def test_grant_must_match_the_measured_profile(key: str, value: object) -> None:
    trace = trace_fixture()
    event_of(trace, "session.updated")["session"]["sglang"]["granted"][key] = value
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert f"unsupported capability: {key}" in result["violations"]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_id",
        "missing_id",
        "bad_event_shape",
        "nonfinite_time",
        "regressed_time",
        "client_error",
        "server_error",
    ],
)
def test_invalid_trace_or_execution_errors_fail(mutation: str) -> None:
    trace = trace_fixture()
    if mutation == "duplicate_id":
        trace[2]["event"]["event_id"] = trace[0]["event"]["event_id"]
    elif mutation == "missing_id":
        del trace[0]["event"]["event_id"]
    elif mutation == "bad_event_shape":
        trace[0]["event"] = []
    elif mutation == "nonfinite_time":
        trace[0]["time_s"] = float("nan")
    elif mutation == "regressed_time":
        trace[2]["time_s"] = 99.0
    elif mutation == "client_error":
        trace.insert(
            0, {"direction": "error", "time_s": 99.0, "event": {"message": "timeout"}}
        )
    else:
        trace.insert(
            0,
            {
                "direction": "receive",
                "time_s": 99.0,
                "event": {
                    "type": "error",
                    "event_id": "failure",
                    "sglang": {"fatal": True},
                    "error": {"message": "rejected"},
                },
            },
        )
    assert evaluate_trace(trace, scenario="continuous")["status"] == "fail"


@pytest.mark.parametrize("typ", ["response.output_audio.delta", "response.done"])
def test_output_after_terminal_fails(typ: str) -> None:
    trace = trace_fixture()
    late = copy.deepcopy(
        next(record for record in trace if record["event"]["type"] == typ)
    )
    late["event"]["event_id"] = "late_output"
    late["time_s"] = 100.265
    index = next(
        i
        for i, record in enumerate(trace)
        if record["event"]["type"] == "sglang.input_audio.drained"
    )
    trace.insert(index, late)
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert any(
        "output after response terminal" in item for item in result["violations"]
    )


def test_audio_delta_after_drain_fails() -> None:
    trace = trace_fixture()
    late = copy.deepcopy(
        next(
            record
            for record in trace
            if record["event"]["type"] == "response.output_audio.delta"
        )
    )
    late["event"]["event_id"] = "late_audio"
    late["time_s"] = 100.275
    index = next(
        i
        for i, record in enumerate(trace)
        if record["event"]["type"] == "session.close"
    )
    trace.insert(index, late)
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert "response.output_audio.delta after drain" in result["violations"]


def test_completed_terminal_before_input_end_fails() -> None:
    trace = [
        record
        for record in trace_fixture()
        if not (
            record["event"]["type"] == "response.output_audio.delta"
            and record["time_s"] > 100.15
        )
    ]
    for record in trace:
        if record["event"]["type"] == "response.output_audio.done":
            record["time_s"] = 100.150
        elif record["event"]["type"] == "response.done":
            record["time_s"] = 100.155
    trace.sort(key=lambda record: record["time_s"])
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert "terminal reason stop without its client command" in result["violations"]


def test_close_time_cleanup_is_not_native_completion() -> None:
    trace = trace_fixture()
    for record in trace:
        if record["event"]["type"] == "response.output_audio.done":
            record["time_s"] = 100.285
        elif record["event"]["type"] == "response.done":
            record["time_s"] = 100.286
            record["event"]["response"].update(
                status="cancelled", status_details={"reason": "client_closed"}
            )
    trace.sort(key=lambda record: record["time_s"])
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert any(
        "ended cancelled/client_closed, expected completed" in item
        for item in result["violations"]
    )


def test_completed_terminal_after_drain_fails() -> None:
    trace = trace_fixture()
    next(
        record
        for record in trace
        if record["event"]["type"] == "response.done"
        and record["event"]["response"]["status"] == "completed"
    )["time_s"] = 100.285
    trace.sort(key=lambda record: record["time_s"])
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert "native completed terminal after the drain receipt" in result["violations"]


def test_terminal_after_drain_without_session_close_fails() -> None:
    trace = trace_fixture()
    for record in trace:
        if record["event"]["type"] == "response.done":
            record["time_s"] = 100.275
    trace.sort(key=lambda record: record["time_s"])
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert "response.done after drain" in result["violations"]


@pytest.mark.parametrize("dropped", [0, 1])
def test_continuous_output_must_be_conserved(dropped: int) -> None:
    trace = trace_fixture()
    deltas = [
        record
        for record in trace
        if record["event"]["type"] == "response.output_audio.delta"
    ]
    trace.remove(deltas[dropped])
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert any(
        "output audio is not conserved: 3528 of 7056 bytes for 2 units" in item
        for item in result["violations"]
    )


def test_short_output_frame_fails_conservation() -> None:
    trace = trace_fixture()
    for record in trace:
        if record["event"]["type"] == "response.output_audio.delta":
            record["event"]["delta"] = base64.b64encode(b"\x01\x00" * 320).decode()
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert any("output audio is not conserved" in item for item in result["violations"])


def test_unaligned_tail_conserves_output_and_padding() -> None:
    tail = base64.b64encode(b"\x00\x00" * 640).decode()
    output = base64.b64encode(b"\x01\x00" * 1764).decode()
    trace = trace_fixture()
    for record in trace:
        event = record["event"]
        if event["type"] == "sglang.input_audio.ended":
            event["accepted_end_ms"] = 200
        elif event["type"] == "sglang.input_audio.drained":
            event.update(accepted_end_ms=200, consumed_ms=200, padding_ms=40)
        elif event["type"] == "sglang.unit.done" and event["unit_id"] == "unit_2":
            event["sglang"]["media_time"]["duration_ms"] = 40
    trace += [
        {
            "direction": "send",
            "time_s": 100.22,
            "event": {
                "type": "input_audio_buffer.append",
                "event_id": "input2",
                "audio": tail,
                "sglang": {"seq": 2, "t_start_ms": 160},
            },
        },
        {
            "direction": "receive",
            "time_s": 100.221,
            "event": {
                "type": "sglang.input_audio.accepted",
                "event_id": "accept2",
                "seq": 2,
                "accepted_end_ms": 200,
                "client_event_id": "input2",
            },
        },
        {
            "direction": "receive",
            "time_s": 100.222,
            "event": {
                "type": "response.output_audio.delta",
                "event_id": "delta2",
                "response_id": "response0",
                "item_id": "item0",
                "output_index": 0,
                "content_index": 0,
                "delta": output,
            },
        },
    ]
    trace.sort(key=lambda record: record["time_s"])
    result = evaluate_trace(trace, scenario="continuous")
    assert result["violations"] == []
    assert result["status"] == "pass"
    assert result["metrics"]["input_audio_s"] == pytest.approx(0.2)
    assert result["metrics"]["output_audio_s"] == pytest.approx(0.24)


@pytest.mark.parametrize(
    ("mutation", "violation"),
    [
        ("drop_all", "expected 2 nonempty native units, observed 0"),
        ("drop_first", "not contiguous from zero"),
        ("empty_unaligned", "empty EOS unit requires exactly unit-aligned input"),
        ("short_duration", "do not account for the accepted input"),
        ("misaligned_start", "media time is not unit-aligned"),
        ("no_media_time", "unit receipt missing media time"),
    ],
)
def test_native_unit_accounting(mutation: str, violation: str) -> None:
    trace = trace_fixture()
    receipts = [
        record for record in trace if record["event"]["type"] == "sglang.unit.done"
    ]
    if mutation == "drop_all":
        for record in receipts:
            trace.remove(record)
    elif mutation == "drop_first":
        trace.remove(receipts[0])
    elif mutation == "empty_unaligned":
        receipts[1]["event"]["sglang"]["media_time"]["duration_ms"] = 0
    elif mutation == "short_duration":
        receipts[1]["event"]["sglang"]["media_time"]["duration_ms"] = 40
    elif mutation == "misaligned_start":
        receipts[1]["event"]["sglang"]["media_time"]["t_start_ms"] = 240
    else:
        del receipts[1]["event"]["sglang"]["media_time"]
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert any(violation in item for item in result["violations"])


def admission(attempt: int, http_status: int = 503) -> dict:
    return {
        "direction": "admission",
        "time_s": 99.0 + attempt / 100,
        "event": {
            "type": "connection_denied",
            "http_status": http_status,
            "attempt": attempt,
        },
    }


def test_admission_denials_before_the_session_are_not_failures() -> None:
    trace = [admission(1), admission(2), admission(3), *trace_fixture()]
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "pass"
    assert result["violations"] == []
    assert result["metrics"]["admission_denials"] == 3


@pytest.mark.parametrize(
    ("records", "violation"),
    [
        ([admission(1), admission(3)], "not contiguous from one"),
        ([admission(2)], "not contiguous from one"),
        ([admission(1), admission(2), admission(3), admission(4)], "less_than_equal"),
        ([admission(1, http_status=500)], "http_status"),
    ],
)
def test_malformed_admission_diagnostics_fail(
    records: list[dict], violation: str
) -> None:
    result = evaluate_trace([*records, *trace_fixture()], scenario="continuous")
    assert result["status"] == "fail"
    assert any(violation in item for item in result["violations"])


def test_admission_diagnostic_after_session_start_fails() -> None:
    trace = trace_fixture()
    late = admission(1)
    late["time_s"] = 100.05
    trace.insert(3, late)
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert "admission diagnostic after the session started" in result["violations"]


def test_extra_admission_fields_fail() -> None:
    record = admission(1)
    record["event"]["reason"] = "capacity"
    result = evaluate_trace([record, *trace_fixture()], scenario="continuous")
    assert result["status"] == "fail"
    assert any("Extra inputs" in item for item in result["violations"])


def test_exhausted_admission_without_a_session_fails() -> None:
    trace = [
        admission(1),
        admission(2),
        admission(3),
        {
            "direction": "error",
            "time_s": 99.5,
            "event": {"message": "admission denied after 3 attempts"},
        },
    ]
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert any("admission denied after 3 attempts" in i for i in result["violations"])
    assert "missing audio input" in result["violations"]


@pytest.mark.parametrize(
    ("status", "reason", "violation"),
    [
        ("completed", "client_closed", "contradicts reason"),
        ("cancelled", "stop", "contradicts reason"),
        ("completed", "disconnect", "unsupported response terminal reason"),
        ("cancelled", "client_cancelled", "unsupported response terminal reason"),
    ],
)
def test_response_terminal_reason_must_match_client_commands(
    status: str, reason: str, violation: str
) -> None:
    trace = trace_fixture()
    event_of(trace, "response.done")["response"].update(
        status=status, status_details={"reason": reason}
    )
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert any(violation in item for item in result["violations"])


def test_response_terminal_without_reason_fails() -> None:
    trace = trace_fixture()
    del event_of(trace, "response.done")["response"]["status_details"]
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert "unsupported response terminal reason: None" in result["violations"]


def test_healthy_trace_without_overlap_is_not_exercised() -> None:
    trace = trace_fixture()
    next(
        record
        for record in trace
        if record["event"]["type"] == "input_audio_buffer.append"
        and record["time_s"] > 100.15
    )["time_s"] = 100.105
    trace.sort(key=lambda record: record["time_s"])
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "not_exercised"
    assert result["violations"] == []
    assert result["coverage"]["input_output_overlap"] is False


def test_client_clock_translation_preserves_metrics() -> None:
    trace = trace_fixture()
    before = evaluate_trace(trace, scenario="continuous")
    for record in trace:
        record["time_s"] += 5000
    after = evaluate_trace(trace, scenario="continuous")
    assert after["status"] == before["status"]
    assert after["coverage"] == before["coverage"]
    assert after["metrics"] == pytest.approx(before["metrics"])


def test_empty_trace_fails() -> None:
    assert evaluate_trace([], scenario="continuous")["status"] == "fail"


def test_removed_cancel_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported scenario: cancel_resume"):
        evaluate_trace(trace_fixture(), scenario="cancel_resume")


@pytest.mark.parametrize(
    ("direction", "typ"),
    [("send", "response.cancel"), ("receive", "sglang.response.cancelled")],
)
def test_removed_cancel_events_cannot_qualify(direction: str, typ: str) -> None:
    trace = trace_fixture()
    trace.insert(
        3,
        {
            "direction": direction,
            "time_s": 100.05,
            "event": {"type": typ, "event_id": "obsolete_cancel"},
        },
    )
    result = evaluate_trace(trace, scenario="continuous")
    assert result["status"] == "fail"
    assert any(typ in violation for violation in result["violations"])
