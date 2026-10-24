# SPDX-License-Identifier: Apache-2.0
"""Focused tests for Full Duplex Unit and stage timing reports."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import sglang_omni.profiler.event_recorder as event_recorder_module
from sglang_omni.admission import QueueFullError
from sglang_omni.pipeline.sessions import CoordinatorSessions, Session
from sglang_omni.profiler.duplex_events import (
    capture_session_stage_start,
    capture_session_unit_ready,
    emit_session_output_event,
    emit_session_stage_bypassed,
    emit_session_stage_started,
    emit_session_unit_event,
)
from sglang_omni.profiler.duplex_metrics import build_duplex_session_metrics
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import (
    OutputChunk,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
)
from sglang_omni.utils.json import JsonValue


def session_event(
    event_name: str,
    timestamp_ns: int,
    *,
    request_id: str = "r1",
    session_id: str = "session-1",
    session_open_index: int = 1,
    input_seq: int = 0,
    stage: str = "coordinator",
    metadata: dict[str, JsonValue] | None = None,
    include_input_metadata: bool = True,
) -> dict[str, JsonValue]:
    event_metadata: dict[str, JsonValue] = {
        "session_id": session_id,
        "session_open_index": session_open_index,
        "input_seq": input_seq,
    }
    if include_input_metadata:
        event_metadata.update(
            {
                "input_modality": "audio",
                "input_t_start_ms": input_seq * 20,
                "input_duration_ms": 20,
            }
        )
    else:
        pass
    if metadata is not None:
        event_metadata.update(metadata)
    else:
        pass
    return {
        "request_id": request_id,
        "stage": stage,
        "event_name": event_name,
        "timestamp_ns": timestamp_ns,
        "metadata": event_metadata,
    }


def stage_event(
    event_name: str,
    timestamp_ns: int,
    *,
    request_id: str = "r1",
    stage: str = "source",
    metadata: dict[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    event_metadata = {} if metadata is None else dict(metadata)
    return {
        "request_id": request_id,
        "stage": stage,
        "event_name": event_name,
        "timestamp_ns": timestamp_ns,
        "metadata": event_metadata,
    }


def read_events(path: str) -> list[dict[str, JsonValue]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def build_admission_coordinator(
    *, max_pending_chunks: int = 8
) -> tuple[CoordinatorSessions, SessionIdentity]:
    coordinator = CoordinatorSessions()
    identity = SessionIdentity("session-lifecycle", 1)
    coordinator.sessions[identity.id] = Session(
        session_identity=identity,
        request=OmniRequest(None),
        stages=("source",),
        bindings={},
        limits=SessionLimits(max_pending_chunks=max_pending_chunks),
    )
    return coordinator, identity


@pytest.fixture(autouse=True)
def reset_recorder() -> None:
    recorder = get_recorder()
    if recorder.is_active():
        recorder.stop()
    else:
        pass
    yield
    if recorder.is_active():
        recorder.stop()
    else:
        pass


def test_ready_capture_and_emission_are_scoped_to_one_profiler_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(event_recorder_module.time, "time_ns", lambda: 123)
    assert capture_session_unit_ready() is None

    recorder = get_recorder()
    first_path = recorder.start("run-a", str(tmp_path / "run-a"), "coordinator")
    observation = capture_session_unit_ready()
    assert observation is not None
    coordinator, identity = build_admission_coordinator()
    asyncio.run(
        coordinator.append_session(
            identity,
            TimedChunk("audio", 0, 20, 0, b"x"),
            ready_timestamp_ns=observation.timestamp_ns,
            ready_run_id=observation.run_id,
        )
    )
    recorder.stop()

    second_path = recorder.start("run-b", str(tmp_path / "run-b"), "coordinator")
    asyncio.run(
        coordinator.append_session(
            identity,
            TimedChunk("audio", 20, 20, 1, b"y"),
            ready_timestamp_ns=observation.timestamp_ns,
            ready_run_id=observation.run_id,
        )
    )
    recorder.stop()

    assert [event["event_name"] for event in read_events(first_path)] == [
        "session_unit_ready",
        "session_unit_admitted",
    ]
    second_events = read_events(second_path)
    assert [event["event_name"] for event in second_events] == ["session_unit_admitted"]
    assert second_events[0]["metadata"]["input_seq"] == 1


def test_queue_full_retry_admits_exact_sequence_once(tmp_path: Path) -> None:
    async def run_test() -> None:
        recorder = get_recorder()
        path = recorder.start("queue-full-test", str(tmp_path), "coordinator")
        coordinator, identity = build_admission_coordinator(max_pending_chunks=1)
        ready_timestamp_ns = 100
        await coordinator.append_session(
            identity,
            TimedChunk("audio", 0, 20, 0, b"x"),
            ready_timestamp_ns=ready_timestamp_ns,
            ready_run_id="queue-full-test",
        )
        retry = TimedChunk("audio", 20, 20, 1, b"y")
        with pytest.raises(QueueFullError):
            await coordinator.append_session(
                identity,
                retry,
                ready_timestamp_ns=ready_timestamp_ns,
                ready_run_id="queue-full-test",
            )
        session = coordinator.sessions[identity.id]
        pending_chunk = session.pending.popleft()
        session.pending_count -= 1
        session.pending_bytes -= pending_chunk.encoded_bytes
        assert (
            await coordinator.append_session(
                identity,
                retry,
                ready_timestamp_ns=ready_timestamp_ns,
                ready_run_id="queue-full-test",
            )
            == 1
        )
        recorder.stop()

        events = read_events(path)
        ready_events = [
            event for event in events if event["event_name"] == "session_unit_ready"
        ]
        admitted_events = [
            event for event in events if event["event_name"] == "session_unit_admitted"
        ]
        assert [event["metadata"]["input_seq"] for event in ready_events] == [0, 1]
        assert [event["metadata"]["input_seq"] for event in admitted_events] == [0, 1]

    asyncio.run(run_test())


def test_payload_is_not_copied_into_event_metadata(tmp_path: Path) -> None:
    recorder = get_recorder()
    path = recorder.start("duplex-test", str(tmp_path), "coordinator")
    identity = SessionIdentity("session-1", 7)
    input_chunk = TimedChunk("audio", 240, 80, 3, b"pcm-bytes")
    output_chunk = OutputChunk(identity, 4, 3, "audio", 240, 40, b"audio-payload")
    emit_session_unit_event(
        request_id="request-3",
        event_name="session_unit_admitted",
        session_identity=identity,
        input_chunk=input_chunk,
        timestamp_ns=100,
    )
    emit_session_output_event(
        request_id="request-3", output_chunk=output_chunk, timestamp_ns=200
    )
    recorder.stop()

    events = read_events(path)
    assert events[0]["metadata"] == {
        "session_id": "session-1",
        "session_open_index": 7,
        "input_seq": 3,
        "input_modality": "audio",
        "input_t_start_ms": 240,
        "input_duration_ms": 80,
    }
    assert "payload" not in events[1]["metadata"]


def test_request_id_keeps_colliding_units_and_reopened_sessions_distinct() -> None:
    events = [
        session_event("session_unit_admitted", 100, request_id="r1"),
        session_event("session_unit_finished", 150, request_id="r1"),
        session_event("session_unit_admitted", 200, request_id="r2"),
        session_event("session_unit_finished", 250, request_id="r2"),
        session_event(
            "session_unit_admitted",
            300,
            request_id="r3",
            session_open_index=2,
        ),
    ]

    report = build_duplex_session_metrics(events)

    assert [unit["request_id"] for unit in report["units"]] == ["r1", "r2", "r3"]
    assert [
        (session["session_id"], session["session_open_index"], session["unit_count"])
        for session in report["sessions"]
    ] == [("session-1", 1, 2), ("session-1", 2, 1)]


def test_partial_trace_fills_input_metadata_from_later_session_event() -> None:
    report = build_duplex_session_metrics(
        [
            session_event(
                "session_output_emitted",
                100,
                request_id="partial",
                session_id="session-partial",
                session_open_index=4,
                input_seq=7,
                include_input_metadata=False,
                metadata={
                    "output_seq": 0,
                    "output_modality": "text",
                    "output_t_start_ms": 0,
                    "output_duration_ms": 20,
                    "kind": "data",
                },
            ),
            session_event(
                "session_unit_finished",
                200,
                request_id="partial",
                session_id="session-partial",
                session_open_index=4,
                input_seq=7,
                metadata={
                    "input_modality": "image",
                    "input_t_start_ms": 140,
                    "input_duration_ms": 60,
                },
            ),
        ]
    )

    unit = report["units"][0]
    assert (unit["session_id"], unit["session_open_index"], unit["input_seq"]) == (
        "session-partial",
        4,
        7,
    )
    assert unit["input_modality"] == "image"
    assert unit["media_start_ms"] == 140
    assert unit["media_duration_ms"] == 60


def test_golden_unit_latency_metrics() -> None:
    report = build_duplex_session_metrics(
        [
            session_event("session_unit_ready", 100_000_000),
            session_event("session_unit_admitted", 250_000_000),
            session_event(
                "session_output_emitted",
                300_000_000,
                metadata={
                    "output_seq": 0,
                    "output_modality": "text",
                    "output_t_start_ms": 0,
                    "output_duration_ms": 0,
                    "kind": "data",
                },
            ),
            session_event("session_unit_finished", 340_000_000),
        ]
    )
    unit = report["units"][0]

    assert unit["ready_to_admitted_ms"] == 150
    assert unit["ready_to_first_output_ms"] == 200
    assert unit["ready_to_finished_ms"] == 240
    assert unit["admitted_to_first_output_ms"] == 50
    assert unit["admitted_to_finished_ms"] == 90
    assert report["summary"]["ready_to_admitted_ms"]["p50"] == 150


def test_stage_queue_service_and_handoff_metrics() -> None:
    events = [
        session_event("session_unit_admitted", 0),
        stage_event("stage_dispatch", 100_000_000, stage="source"),
        session_event("session_stage_started", 120_000_000, stage="source"),
        stage_event("stage_complete", 170_000_000, stage="source"),
        stage_event("stage_dispatch", 180_000_000, stage="sink"),
        session_event("session_stage_started", 200_000_000, stage="sink"),
        stage_event("stage_complete", 250_000_000, stage="sink"),
        session_event("session_unit_finished", 300_000_000),
    ]

    stages = build_duplex_session_metrics(events)["units"][0]["stages"]

    assert stages == [
        {
            "stage": "source",
            "dispatch_timestamp_ns": 100_000_000,
            "queued_timestamp_ns": 100_000_000,
            "started_timestamp_ns": 120_000_000,
            "finished_timestamp_ns": 170_000_000,
            "bypassed_timestamp_ns": None,
            "bypass_reason": None,
            "timing_status": "observed",
            "queue_ms": 20,
            "service_ms": 50,
            "handoff_from_previous_ms": None,
        },
        {
            "stage": "sink",
            "dispatch_timestamp_ns": 180_000_000,
            "queued_timestamp_ns": 180_000_000,
            "started_timestamp_ns": 200_000_000,
            "finished_timestamp_ns": 250_000_000,
            "bypassed_timestamp_ns": None,
            "bypass_reason": None,
            "timing_status": "observed",
            "queue_ms": 20,
            "service_ms": 50,
            "handoff_from_previous_ms": 10,
        },
    ]


def test_ar_boundaries_bypass_and_missing_telemetry_are_distinct() -> None:
    events = [
        session_event("session_unit_admitted", 0, request_id="ar"),
        stage_event("stage_dispatch", 20_000_000, request_id="ar", stage="ar"),
        stage_event("scheduler_queue_enter", 40_000_000, request_id="ar", stage="ar"),
        stage_event("scheduler_prefill_start", 60_000_000, request_id="ar", stage="ar"),
        stage_event("stage_complete", 100_000_000, request_id="ar", stage="ar"),
        session_event("session_unit_finished", 110_000_000, request_id="ar"),
        session_event("session_unit_admitted", 0, request_id="bypass"),
        session_event(
            "session_stage_bypassed",
            40,
            request_id="bypass",
            stage="ar",
            metadata={"reason": "generation_bypassed"},
        ),
        session_event("session_unit_admitted", 0, request_id="incomplete"),
        stage_event("stage_dispatch", 2, request_id="incomplete", stage="source"),
    ]

    units = {
        unit["request_id"]: unit
        for unit in build_duplex_session_metrics(events)["units"]
    }
    ar_stage = units["ar"]["stages"][0]
    bypass_stage = units["bypass"]["stages"][0]
    incomplete_stage = units["incomplete"]["stages"][0]

    assert ar_stage["queued_timestamp_ns"] == 40_000_000
    assert ar_stage["started_timestamp_ns"] == 60_000_000
    assert ar_stage["queue_ms"] == 20
    assert ar_stage["service_ms"] == 40
    assert bypass_stage["timing_status"] == "bypassed"
    assert bypass_stage["bypass_reason"] == "generation_bypassed"
    assert bypass_stage["queue_ms"] is None
    assert bypass_stage["service_ms"] is None
    assert incomplete_stage["timing_status"] == "incomplete"
    assert incomplete_stage["queue_ms"] is None


def test_tp_duplicate_boundaries_use_earliest_start_and_latest_finish() -> None:
    report = build_duplex_session_metrics(
        [
            session_event("session_unit_admitted", 0),
            stage_event("stage_dispatch", 105, metadata={}, stage="source"),
            stage_event("stage_dispatch", 100, metadata={}, stage="source"),
            session_event("session_stage_started", 122, stage="source"),
            session_event("session_stage_started", 120, stage="source"),
            stage_event("stage_complete", 168, metadata={}, stage="source"),
            stage_event("stage_complete", 170, metadata={}, stage="source"),
        ]
    )

    stage = report["units"][0]["stages"][0]
    assert stage["dispatch_timestamp_ns"] == 100
    assert stage["started_timestamp_ns"] == 120
    assert stage["finished_timestamp_ns"] == 170


def test_bypassed_event_emitter_records_reason(tmp_path: Path) -> None:
    recorder = get_recorder()
    path = recorder.start("duplex-bypass-test", str(tmp_path), "ar-stage")
    identity = SessionIdentity("session-bypass", 1)
    input_chunk = TimedChunk("audio", 0, 20, 0, b"pcm")
    emit_session_unit_event(
        request_id="request-bypass",
        event_name="session_unit_admitted",
        session_identity=identity,
        input_chunk=input_chunk,
        timestamp_ns=0,
    )
    emit_session_stage_bypassed(
        request_id="request-bypass",
        session_identity=identity,
        input_chunk=input_chunk,
        reason="generation_bypassed",
        stage="ar-stage",
        timestamp_ns=40,
    )
    recorder.stop()

    stage = build_duplex_session_metrics(read_events(path))["units"][0]["stages"][0]
    assert stage["timing_status"] == "bypassed"
    assert stage["bypassed_timestamp_ns"] == 40
    assert stage["bypass_reason"] == "generation_bypassed"


def test_stage_start_observation_does_not_cross_profiler_runs(tmp_path: Path) -> None:
    recorder = get_recorder()
    recorder.start("run-a", str(tmp_path / "run-a"), "stage")
    observation = capture_session_stage_start()
    assert observation is not None
    run_id, timestamp_ns = observation

    run_b_path = recorder.start("run-b", str(tmp_path / "run-b"), "stage")
    emit_session_stage_started(
        request_id="request-a",
        session_identity=SessionIdentity("session-a", 1),
        input_chunk=TimedChunk("audio", 0, 20, 0, b"pcm"),
        timestamp_ns=timestamp_ns,
        expected_run_id=run_id,
    )
    recorder.stop()

    assert read_events(run_b_path) == []
