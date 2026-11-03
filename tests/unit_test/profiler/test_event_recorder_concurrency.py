# SPDX-License-Identifier: Apache-2.0
"""Producer buffering and stop/write concurrency tests."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import ItemsView
from io import TextIOWrapper
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sglang_omni.profiler import event_recorder as event_recorder_module
from sglang_omni.profiler import event_recorder_writer
from sglang_omni.profiler.event_recorder import (
    RequestEvent,
    RequestEventBuffer,
    RequestEventRecorder,
    RequestEventSnapshot,
)
from sglang_omni.profiler.event_recorder_types import (
    EVENT_FILE_BUFFER_BYTES,
    MAX_PRODUCER_BUFFERS,
    RUNTIME_WRITER_BACKLOG_YIELD_INTERVAL_SECONDS,
    RUNTIME_WRITER_MAX_EVENTS_PER_ROUND,
    RUNTIME_WRITER_POLL_INTERVAL_SECONDS,
    ProducerDropCounter,
    RecorderRunState,
)
from tests.unit_test.profiler.event_recorder_test_utils import (
    read_events,
    read_status,
    stall_writer,
)


@pytest.fixture(autouse=True)
def runtime_writer_has_idle_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    original_set_writer_scheduling = RequestEventRecorder.set_writer_scheduling

    def set_writer_scheduling(
        recorder: RequestEventRecorder, run_state: RecorderRunState
    ) -> None:
        if run_state.mode == "runtime":
            run_state.writer_scheduling = "SCHED_IDLE"
        else:
            original_set_writer_scheduling(recorder, run_state)

    monkeypatch.setattr(
        RequestEventRecorder, "set_writer_scheduling", set_writer_scheduling
    )


def test_disabled_emit_returns_before_timestamp_and_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RequestEventRecorder()

    def fail(*arguments: object, **keyword_arguments: object) -> None:
        del arguments, keyword_arguments
        raise AssertionError("disabled emit performed producer work")

    with monkeypatch.context() as capture_patch:
        capture_patch.setattr(event_recorder_writer.time, "time_ns", fail)
        capture_patch.setattr(event_recorder_writer, "snapshot_metadata", fail)
        recorder.emit(
            request_id="request",
            stage="stage",
            event_name="event",
            metadata={"value": 1},
        )

    assert recorder.is_active() is False


def test_active_capture_does_not_serialize_or_open_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder()
    path = recorder.start("capture", str(tmp_path), "stage")

    def fail(*arguments: object, **keyword_arguments: object) -> None:
        del arguments, keyword_arguments
        raise AssertionError("active capture performed writer work")

    with monkeypatch.context() as capture_patch:
        capture_patch.setattr(event_recorder_writer.json, "dumps", fail)
        capture_patch.setattr(recorder, "open_event_file", fail)
        recorder.emit(
            request_id="request",
            stage="stage",
            event_name="event",
            metadata={"value": 1},
        )

    assert recorder.writer_thread is None
    assert Path(path).exists() is False
    assert recorder.stop(wait=True) == path
    assert len(read_events(path)) == 1


def test_runtime_flushes_complete_low_volume_jsonl_while_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        event_recorder_writer, "RUNTIME_WRITER_FLUSH_INTERVAL_SECONDS", 0.05
    )
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(
        "runtime-live-flush",
        str(tmp_path),
        "stage",
        mode="runtime",
        sampling_unit="request",
        sample_rate=1.0,
    )
    for event_number in range(3):
        recorder.emit(
            request_id="request",
            stage="stage",
            event_name="event",
            metadata={"number": event_number},
        )

    visible_lines: list[bytes] = []
    visibility_deadline = time.monotonic() + 5
    while not visible_lines and time.monotonic() < visibility_deadline:
        if Path(path).exists():
            visible_lines = [
                line
                for line in Path(path).read_bytes().splitlines(keepends=True)
                if line.endswith(b"\n")
            ]
        else:
            pass
        if not visible_lines:
            threading.Event().wait(timeout=0.01)
        else:
            pass
    assert visible_lines
    assert Path(path).stat().st_size < EVENT_FILE_BUFFER_BYTES
    visible_events = [json.loads(line) for line in visible_lines]
    assert visible_events
    assert recorder.active_state is not None

    assert recorder.stop(wait=True) == path
    events = read_events(path)
    status = read_status(path)
    assert len(events) == 3
    assert status["accepted_events"] == 3
    assert status["written_events"] == 3
    assert status["unwritten_events"] == 0
    assert status["complete"] is True


def test_runtime_live_flush_failure_marks_run_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        event_recorder_writer, "RUNTIME_WRITER_FLUSH_INTERVAL_SECONDS", 0.02
    )
    recorder = RequestEventRecorder(queue_capacity=8)
    original_open_event_file = recorder.open_event_file

    def open_file_with_flush_failure(
        run_state: RecorderRunState,
    ) -> TextIOWrapper | None:
        file_handle = original_open_event_file(run_state)
        if file_handle is not None:
            failing_file_handle = MagicMock(spec=TextIOWrapper, wraps=file_handle)
            failing_file_handle.closed = False
            failing_file_handle.flush.side_effect = OSError("synthetic flush failure")
            return failing_file_handle
        else:
            return None

    monkeypatch.setattr(recorder, "open_event_file", open_file_with_flush_failure)
    path = recorder.start(
        "runtime-live-flush-failure",
        str(tmp_path),
        "stage",
        mode="runtime",
        sampling_unit="request",
        sample_rate=1.0,
    )
    recorder.emit(
        request_id="accepted",
        stage="stage",
        event_name="event",
    )
    run_state = recorder.active_state
    assert run_state is not None
    writer_failure_deadline = time.monotonic() + 5
    while not run_state.writer_failed and time.monotonic() < writer_failure_deadline:
        threading.Event().wait(timeout=0.001)
    assert run_state.writer_failed is True
    assert "event_file_flush_failure" in run_state.incomplete_reasons
    recorder.emit(
        request_id="after-flush-failure",
        stage="stage",
        event_name="event",
    )

    assert recorder.stop(wait=False) == path
    assert recorder.wait_for_finalizations(timeout_seconds=5) is True
    status = read_status(path)
    assert status["accepted_events"] == 1
    assert status["processed_events"] == 1
    assert status["written_events"] == 1
    assert status["unwritten_events"] == 0
    assert status["write_failures"] >= 1
    assert status["complete"] is False
    assert "event_file_flush_failure" in status["incomplete_reasons"]


def test_concurrent_producers_preserve_per_producer_order(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=128)
    path = recorder.start("concurrent", str(tmp_path), "stage")
    producer_count = 8
    events_per_producer = 40

    def produce_events(producer_number: int) -> None:
        for event_number in range(events_per_producer):
            recorder.emit(
                request_id=f"producer-{producer_number}",
                stage="stage",
                event_name="event",
                metadata={"number": event_number},
            )

    producers = [
        threading.Thread(target=produce_events, args=(producer_number,))
        for producer_number in range(producer_count)
    ]
    for producer in producers:
        producer.start()
    for producer in producers:
        producer.join(timeout=5)
        assert producer.is_alive() is False

    recorder.stop(wait=True)
    events = read_events(path)
    assert len(events) == producer_count * events_per_producer
    for producer_number in range(producer_count):
        producer_events = [
            event
            for event in events
            if event["request_id"] == f"producer-{producer_number}"
        ]
        assert [event["metadata"]["number"] for event in producer_events] == list(
            range(events_per_producer)
        )
    status = read_status(path)
    assert status["accepted_events"] == len(events)
    assert status["processed_events"] == len(events)
    assert status["written_events"] == len(events)
    assert status["complete"] is True


def test_buffer_overflow_reports_exact_loss_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=2)
    path = recorder.start("overflow", str(tmp_path), "stage")
    recorder.emit(request_id="request-0", stage="stage", event_name="event")

    with monkeypatch.context() as overflow_patch:
        overflow_patch.setattr(recorder, "control_lock", object())
        for event_number in range(1, 4):
            recorder.emit(
                request_id=f"request-{event_number}",
                stage="stage",
                event_name="event",
            )

    recorder.stop(wait=True)
    status = read_status(path)
    assert len(read_events(path)) == 2
    assert status["accepted_events"] == 2
    assert status["written_events"] == 2
    assert status["dropped_events"] == 2
    assert status["complete"] is False


def test_overflow_producers_share_bounded_drop_accounting(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=2)
    path = recorder.start("bounded-overflow", str(tmp_path), "stage")
    observed_drop_counters: list[ProducerDropCounter] = []

    def produce(producer_number: int) -> None:
        recorder.emit(
            request_id=f"request-{producer_number}",
            stage="stage",
            event_name="event",
        )
        if producer_number >= MAX_PRODUCER_BUFFERS:
            assert recorder.active_state is not None
            producer_drop_counter = recorder.active_state.producer_drop_counter
            assert producer_drop_counter is not None
            observed_drop_counters.append(producer_drop_counter)

    producer_count = MAX_PRODUCER_BUFFERS + 8
    producers = [
        threading.Thread(target=produce, args=(producer_number,))
        for producer_number in range(producer_count)
    ]
    for producer in producers:
        producer.start()
        producer.join(timeout=5)
        assert producer.is_alive() is False

    assert recorder.active_state is not None
    assert len(recorder.active_state.producer_buffers) == MAX_PRODUCER_BUFFERS
    assert recorder.active_state.producer_drop_counter is not None
    assert len(observed_drop_counters) == producer_count - MAX_PRODUCER_BUFFERS
    assert all(
        drop_counter is recorder.active_state.producer_drop_counter
        for drop_counter in observed_drop_counters
    )

    recorder.stop(wait=True)
    status = read_status(path)
    assert status["accepted_events"] == MAX_PRODUCER_BUFFERS
    assert status["dropped_events"] == producer_count - MAX_PRODUCER_BUFFERS
    assert status["complete"] is False
    assert status["producer_buffer_count"] == MAX_PRODUCER_BUFFERS
    assert sorted(status["producer_buffer_event_counts"]) == [1] * MAX_PRODUCER_BUFFERS
    assert "producer_buffer_registration_limit" in status["incomplete_reasons"]


def test_stop_waits_for_an_admitted_emit_to_finish(tmp_path: Path) -> None:
    recorder = RequestEventRecorder()
    path = recorder.start("emit-race", str(tmp_path), "stage")
    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()

    class BlockingMetadata(dict[str, object]):
        def items(self) -> ItemsView[str, object]:
            snapshot_entered.set()
            if not release_snapshot.wait(timeout=5):
                raise AssertionError("test did not release producer snapshot")
            else:
                pass
            return super().items()

    producer = threading.Thread(
        target=recorder.emit,
        kwargs={
            "request_id": "request",
            "stage": "stage",
            "event_name": "event",
            "metadata": BlockingMetadata(value=1),
        },
    )
    producer.start()
    assert snapshot_entered.wait(timeout=5)

    finalization = recorder.deactivate()
    assert finalization is not None
    assert finalization.completion_event.is_set() is False
    release_snapshot.set()
    producer.join(timeout=5)
    assert producer.is_alive() is False

    assert finalization.wait(timeout_seconds=5) is True
    assert [event["request_id"] for event in read_events(path)] == ["request"]
    status = read_status(path)
    assert status["accepted_events"] == 1
    assert status["written_events"] == 1
    assert status["complete"] is True


def test_stalled_writer_does_not_block_stop_or_inactive_emit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder()
    path = recorder.start("stalled", str(tmp_path), "stage")
    recorder.emit(request_id="request", stage="stage", event_name="event")
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)

    assert recorder.stop(wait=False) == path
    assert writer_entered.wait(timeout=5)
    recorder.emit(request_id="after-stop", stage="stage", event_name="event")
    with pytest.raises(RuntimeError, match="still draining"):
        recorder.start("next", str(tmp_path / "next"), "stage")

    release_writer.set()
    assert recorder.wait_for_finalizations(timeout_seconds=5) is True
    assert [event["request_id"] for event in read_events(path)] == ["request"]
    assert recorder.start("next", str(tmp_path / "next"), "stage")
    recorder.stop(wait=True)


@pytest.mark.parametrize("sampling_unit", ["request", "session"])
def test_runtime_sampling_is_deterministic_across_independent_recorders(
    sampling_unit: str,
    tmp_path: Path,
) -> None:
    sampled_identities: list[set[str]] = []
    for recorder_number in range(2):
        recorder = RequestEventRecorder(queue_capacity=256)
        path = recorder.start(
            "stable-sampling",
            str(tmp_path / str(recorder_number)),
            "coordinator",
            mode="runtime",
            sampling_unit=sampling_unit,
            sample_rate=0.35,
        )
        assert recorder.active_state is not None
        unadmitted_selected: set[str] = set()
        for request_number in range(12):
            request_id = f"unadmitted-{request_number}"
            session_identity = (f"session-unadmitted-{request_number}", 1)
            sampling_identity: str | tuple[str, int] = (
                request_id if sampling_unit == "request" else session_identity
            )
            decision = recorder.deterministic_sampling_decision(
                recorder.active_state, sampling_identity
            )
            assert (
                recorder.should_sample_request(
                    recorder.active_state,
                    request_id,
                    sampling_session_identity=(
                        session_identity if sampling_unit == "session" else None
                    ),
                )
                is decision
            )
            recorder.emit(
                request_id=request_id,
                stage="coordinator",
                event_name="session_unit_ready",
                sampling_session_identity=(
                    session_identity if sampling_unit == "session" else None
                ),
            )
            if decision:
                unadmitted_selected.add(request_id)

        assert recorder.active_state.request_sampling_decisions == {}
        assert recorder.active_state.request_sampling_stage_names == {}
        assert recorder.active_state.request_session_identities == {}

        admitted_decisions: dict[str, bool] = {}
        for identity_number in range(60):
            session_id = f"session-{identity_number}"
            operations: list[tuple[str, dict[str, object]]]
            if sampling_unit == "request":
                operations = [(f"request-{identity_number}", {})]
            else:
                operations = [
                    (
                        f"{operation}-{identity_number}",
                        {
                            "omni_session": {
                                "operation": operation,
                                "session_identity": {
                                    "id": session_id,
                                    "open_index": 2,
                                },
                            }
                        },
                    )
                    for operation in ("open", "append", "close")
                ]
            session_decision: bool | None = None
            for request_id, request_metadata in operations:
                recorder.record_request_admission(
                    request_id,
                    request_metadata,
                    stage_name="coordinator",
                )
                assert request_id in recorder.active_state.request_sampling_decisions
                decision = recorder.active_state.request_sampling_decisions[request_id]
                if sampling_unit == "session":
                    assert decision is recorder.deterministic_sampling_decision(
                        recorder.active_state, (session_id, 2)
                    )
                    if session_decision is None:
                        session_decision = decision
                    else:
                        assert decision is session_decision
                else:
                    assert decision is recorder.deterministic_sampling_decision(
                        recorder.active_state, request_id
                    )
                admitted_decisions[request_id] = decision
                recorder.emit(
                    request_id=request_id,
                    stage="coordinator",
                    event_name="request_admission",
                )
                recorder.clear_request_sampling(request_id, stage_name="coordinator")
                assert (
                    request_id not in recorder.active_state.request_sampling_decisions
                )
                assert (
                    request_id not in recorder.active_state.request_session_identities
                )

        assert set(admitted_decisions.values()) == {False, True}
        expected_selected = unadmitted_selected | {
            request_id
            for request_id, decision in admitted_decisions.items()
            if decision
        }
        assert recorder.active_state.request_sampling_decisions == {}
        assert recorder.active_state.request_session_identities == {}
        recorder.stop(wait=True)
        sampled = {event["request_id"] for event in read_events(path)}
        assert sampled == expected_selected
        sampled_identities.append(sampled)
        status = read_status(path)
        assert status["dropped_events"] == 0
        assert status["complete"] is True

    assert sampled_identities[0] == sampled_identities[1]
    assert sampled_identities[0]


def test_runtime_writer_drains_backlog_in_fair_bounded_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=64)
    writer_entered = threading.Event()
    release_setup = threading.Event()
    all_events_drained = threading.Event()
    accepted_count = 128
    active_round_sizes: list[int] = []
    active_round_buffer_counts: list[list[int]] = []
    backlog_wait_intervals: list[float | None] = []
    last_drain_had_backlog = False

    def pause_writer(run_state: RecorderRunState) -> None:
        writer_entered.set()
        if not release_setup.wait(timeout=5):
            raise AssertionError("test did not release the runtime writer")
        run_state.writer_scheduling = "SCHED_IDLE"

    original_drain = recorder.drain_runtime_buffers

    def track_drain(
        run_state: RecorderRunState, file_handle: TextIOWrapper | None
    ) -> int:
        nonlocal last_drain_had_backlog
        before_counts = [
            len(producer_buffer.events)
            for producer_buffer in run_state.producer_buffers
        ]
        count = original_drain(run_state, file_handle)
        after_counts = [
            len(producer_buffer.events)
            for producer_buffer in run_state.producer_buffers
        ]
        last_drain_had_backlog = any(after_counts)
        if count:
            active_round_sizes.append(count)
            active_round_buffer_counts.append(
                [before - after for before, after in zip(before_counts, after_counts)]
            )
            if run_state.processed_events == accepted_count:
                all_events_drained.set()
            else:
                pass
        else:
            pass
        return count

    monkeypatch.setattr(recorder, "set_writer_scheduling", pause_writer)
    monkeypatch.setattr(recorder, "drain_runtime_buffers", track_drain)
    path = recorder.start(
        "runtime-backlog",
        str(tmp_path),
        "stage",
        mode="runtime",
        sampling_unit="request",
        sample_rate=1.0,
    )
    assert writer_entered.wait(timeout=5)
    run_state = recorder.active_state
    assert run_state is not None
    original_wait = run_state.stop_requested.wait

    def track_stop_wait(timeout: float | None = None) -> bool:
        if last_drain_had_backlog:
            backlog_wait_intervals.append(timeout)
        else:
            pass
        return original_wait(timeout)

    monkeypatch.setattr(run_state.stop_requested, "wait", track_stop_wait)

    def produce_events(producer_number: int) -> None:
        for event_number in range(64):
            recorder.emit(
                request_id=f"producer-{producer_number}",
                stage="stage",
                event_name="event",
                metadata={"number": event_number},
            )

    for producer_number in range(2):
        producer = threading.Thread(target=produce_events, args=(producer_number,))
        producer.start()
        producer.join(timeout=5)
        assert producer.is_alive() is False

    release_setup.set()
    assert all_events_drained.wait(timeout=5)
    assert recorder.stop(wait=False) == path
    assert recorder.wait_for_finalizations(timeout_seconds=5) is True

    assert len(active_round_sizes) >= 2
    assert all(
        0 < round_size <= RUNTIME_WRITER_MAX_EVENTS_PER_ROUND
        for round_size in active_round_sizes
    )
    assert active_round_buffer_counts[0] == [16, 16]
    assert len(backlog_wait_intervals) >= 2
    assert all(
        interval is not None
        and 0 < interval <= RUNTIME_WRITER_BACKLOG_YIELD_INTERVAL_SECONDS
        and interval < RUNTIME_WRITER_POLL_INTERVAL_SECONDS
        for interval in backlog_wait_intervals
    )

    events = read_events(path)
    assert len(events) == accepted_count
    first_round = events[:RUNTIME_WRITER_MAX_EVENTS_PER_ROUND]
    assert sum(event["request_id"] == "producer-0" for event in first_round) == 16
    assert sum(event["request_id"] == "producer-1" for event in first_round) == 16
    for producer_number in range(2):
        producer_events = [
            event
            for event in events
            if event["request_id"] == f"producer-{producer_number}"
        ]
        assert [event["metadata"]["number"] for event in producer_events] == list(
            range(64)
        )
    status = read_status(path)
    assert status["accepted_events"] == accepted_count
    assert status["processed_events"] == accepted_count
    assert status["written_events"] == accepted_count
    assert status["dropped_events"] == 0
    assert status["complete"] is True


def test_runtime_final_drain_is_batched_and_asynchronous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=96)
    writer_entered = threading.Event()
    release_setup = threading.Event()
    final_batch_sizes: list[int] = []

    def pause_writer(run_state: RecorderRunState) -> None:
        writer_entered.set()
        if not release_setup.wait(timeout=5):
            raise AssertionError("test did not release the runtime writer")
        run_state.writer_scheduling = "SCHED_IDLE"

    original_drain = recorder.drain_runtime_buffers

    def track_drain(
        run_state: RecorderRunState, file_handle: TextIOWrapper | None
    ) -> int:
        stopping = run_state.stop_requested.is_set()
        count = original_drain(run_state, file_handle)
        if stopping and count:
            final_batch_sizes.append(count)
        return count

    monkeypatch.setattr(recorder, "set_writer_scheduling", pause_writer)
    monkeypatch.setattr(recorder, "drain_runtime_buffers", track_drain)
    path = recorder.start(
        "runtime-final-drain",
        str(tmp_path),
        "stage",
        mode="runtime",
        sampling_unit="request",
        sample_rate=1.0,
    )
    assert writer_entered.wait(timeout=5)
    accepted_count = RUNTIME_WRITER_MAX_EVENTS_PER_ROUND * 2 + 1
    for event_number in range(accepted_count):
        recorder.emit(
            request_id="request",
            stage="stage",
            event_name="event",
            metadata={"number": event_number},
        )

    assert recorder.stop(wait=False) == path
    assert recorder.draining_state is not None
    finalization = recorder.draining_state.finalization
    assert finalization.completion_event.is_set() is False
    release_setup.set()
    assert finalization.wait(timeout_seconds=5) is True

    assert final_batch_sizes
    assert all(
        0 < batch_size <= RUNTIME_WRITER_MAX_EVENTS_PER_ROUND
        for batch_size in final_batch_sizes
    )
    assert sum(final_batch_sizes) == accepted_count
    assert len(read_events(path)) == accepted_count
    status = read_status(path)
    assert status["accepted_events"] == accepted_count
    assert status["processed_events"] == accepted_count
    assert status["written_events"] == accepted_count
    assert status["unwritten_events"] == 0
    assert status["complete"] is True


def test_runtime_session_sampling_covers_units_and_ready_boundary(
    tmp_path: Path,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=64)
    path = recorder.start(
        "session-sampling",
        str(tmp_path),
        "coordinator",
        mode="runtime",
        sampling_unit="session",
        sample_rate=1.0,
    )
    session_identity = {"id": "session-A", "open_index": 3}
    pre_payload_request_id = "session-operation-pre-payload"
    recorder.emit(
        request_id=pre_payload_request_id,
        stage="stage",
        event_name="stage_stream_chunk_received",
        sampling_session_identity=("session-A", 3),
    )
    assert recorder.active_state is not None
    assert (
        pre_payload_request_id not in recorder.active_state.request_sampling_decisions
    )
    recorder.record_request_admission(
        pre_payload_request_id,
        {
            "omni_session": {
                "operation": "append",
                "session_identity": session_identity,
            }
        },
        stage_name="stage",
    )
    recorder.emit(
        request_id=pre_payload_request_id,
        stage="stage",
        event_name="stage_input_received",
    )
    request_ids = [
        f"session-operation-{operation}" for operation in ("open", "append", "close")
    ]

    for operation, request_id in zip(("open", "append", "close"), request_ids):
        recorder.emit(
            request_id=request_id,
            stage="coordinator",
            event_name="session_unit_ready",
            metadata={"session_id": "session-A", "session_open_index": 3},
        )
        recorder.record_request_admission(
            request_id,
            {
                "omni_session": {
                    "operation": operation,
                    "session_identity": session_identity,
                }
            },
            stage_name="coordinator",
        )
        recorder.emit(
            request_id=request_id,
            stage="stage",
            event_name="stage_input_received",
        )

    recorder.emit(
        request_id="ordinary-request",
        stage="coordinator",
        event_name="request_admission",
    )
    recorder.stop(wait=True)

    events = read_events(path)
    event_counts = {
        request_id: sum(event["request_id"] == request_id for event in events)
        for request_id in (*request_ids, pre_payload_request_id)
    }
    assert event_counts == dict.fromkeys((*request_ids, pre_payload_request_id), 2)
    assert "ordinary-request" not in {event["request_id"] for event in events}
    status = read_status(path)
    assert status["dropped_events"] == 0
    assert status["unwritten_events"] == 0
    assert status["complete"] is True
    assert status["completeness_scope"] == "sampled_event_storage"


def test_runtime_writer_concurrency_preserves_all_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=64)
    path = recorder.start(
        "runtime-concurrency",
        str(tmp_path),
        "stage",
        mode="runtime",
        sampling_unit="request",
        sample_rate=1.0,
    )
    producer_count = 8
    events_per_producer = 48
    first_write_entered = threading.Event()
    release_first_write = threading.Event()
    first_write_finished = threading.Event()
    producer_snapshot_entered = threading.Event()
    release_producer_snapshot = threading.Event()
    original_write_event = recorder.write_event

    def block_first_write(
        request_event: RequestEvent,
        file_handle: TextIOWrapper,
        run_state: RecorderRunState,
    ) -> None:
        if request_event.request_id == "request-seed":
            first_write_entered.set()
            if not release_first_write.wait(timeout=5):
                raise AssertionError("test did not release the runtime writer")
        original_write_event(request_event, file_handle, run_state)
        if request_event.request_id == "request-seed":
            first_write_finished.set()

    class BlockingMetadata(dict[str, object]):
        def items(self) -> ItemsView[str, object]:
            producer_snapshot_entered.set()
            if not release_producer_snapshot.wait(timeout=5):
                raise AssertionError("test did not release producer capture")
            return super().items()

    monkeypatch.setattr(recorder, "write_event", block_first_write)
    recorder.emit(
        request_id="request-seed",
        stage="stage",
        event_name="event",
        metadata={"number": -1},
    )
    assert first_write_entered.wait(timeout=5)

    def produce_events(producer_number: int) -> None:
        for event_number in range(events_per_producer):
            recorder.emit(
                request_id=f"request-{producer_number}",
                stage="stage",
                event_name="event",
                metadata=(
                    BlockingMetadata(number=event_number)
                    if producer_number == 0 and event_number == 0
                    else {"number": event_number}
                ),
            )

    producers = [
        threading.Thread(target=produce_events, args=(producer_number,))
        for producer_number in range(producer_count)
    ]
    for producer in producers:
        producer.start()
    assert producer_snapshot_entered.wait(timeout=5)
    release_first_write.set()
    assert first_write_finished.wait(timeout=5)
    release_producer_snapshot.set()
    for producer in producers:
        producer.join(timeout=5)
        assert producer.is_alive() is False

    recorder.stop(wait=True)
    events = read_events(path)
    event_keys = {
        (event["request_id"], event["metadata"]["number"]) for event in events
    }
    expected_keys = {
        (f"request-{producer_number}", event_number)
        for producer_number in range(producer_count)
        for event_number in range(events_per_producer)
    }
    expected_keys.add(("request-seed", -1))
    assert event_keys == expected_keys
    assert len(events) == len(expected_keys)
    status = read_status(path)
    assert status["accepted_events"] == len(expected_keys)
    assert status["written_events"] == len(expected_keys)
    assert status["dropped_events"] == 0
    assert status["complete"] is True


def test_runtime_buffer_overflow_is_counted_while_writer_is_paused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=2)
    writer_entered = threading.Event()
    release_writer = threading.Event()

    def pause_writer(run_state: RecorderRunState) -> None:
        writer_entered.set()
        if not release_writer.wait(timeout=5):
            raise AssertionError("test did not release the runtime writer")
        run_state.writer_scheduling = "SCHED_IDLE"

    monkeypatch.setattr(recorder, "set_writer_scheduling", pause_writer)
    path = recorder.start(
        "runtime-overflow",
        str(tmp_path),
        "stage",
        mode="runtime",
        sampling_unit="request",
        sample_rate=1.0,
    )
    assert writer_entered.wait(timeout=5)
    for event_number in range(4):
        recorder.emit(
            request_id=f"request-{event_number}",
            stage="stage",
            event_name="event",
        )
    release_writer.set()
    recorder.stop(wait=True)

    status = read_status(path)
    assert status["accepted_events"] == 2
    assert status["written_events"] == 2
    assert status["dropped_events"] == 2
    assert status["complete"] is False


def test_runtime_disk_cap_reports_unwritten_events_without_partial_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    limit_reached = threading.Event()
    original_write_event = recorder.write_event
    original_snapshot_metadata = event_recorder_writer.snapshot_metadata
    original_should_sample = recorder.should_sample_request
    original_time_ns = event_recorder_writer.time.time_ns
    snapshot_calls = 0
    sampling_calls = 0
    timestamp_calls = 0

    def write_event(
        request_event: RequestEvent,
        file_handle: TextIOWrapper,
        run_state: RecorderRunState,
    ) -> None:
        original_write_event(request_event, file_handle, run_state)
        if run_state.file_size_limit_reached:
            limit_reached.set()

    def snapshot_metadata(metadata: object) -> dict[str, object]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_snapshot_metadata(metadata)

    def should_sample_request(*arguments: object, **keyword_arguments: object) -> bool:
        nonlocal sampling_calls
        sampling_calls += 1
        return original_should_sample(*arguments, **keyword_arguments)

    def time_ns() -> int:
        nonlocal timestamp_calls
        timestamp_calls += 1
        return original_time_ns()

    monkeypatch.setattr(recorder, "write_event", write_event)
    monkeypatch.setattr(event_recorder_writer, "snapshot_metadata", snapshot_metadata)
    monkeypatch.setattr(recorder, "should_sample_request", should_sample_request)
    monkeypatch.setattr(event_recorder_writer.time, "time_ns", time_ns)
    path = recorder.start(
        "runtime-disk-cap",
        str(tmp_path),
        "stage",
        mode="runtime",
        sampling_unit="session",
        sample_rate=1.0,
        max_event_file_bytes=1,
    )
    recorder.record_request_admission(
        "request-1",
        {"session_id": "session-1", "session_open_index": 1},
        stage_name="stage",
    )
    recorder.emit(
        request_id="request-1",
        stage="stage",
        event_name="event",
        metadata={"value": 1},
    )
    assert limit_reached.wait(timeout=5)
    run_state = recorder.active_state
    assert run_state is not None
    assert run_state.file_size_limit_reached is True

    def fail_sampling_decision(*arguments: object, **keyword_arguments: object) -> bool:
        del arguments, keyword_arguments
        raise AssertionError("terminal Runtime sampling computed a new decision")

    monkeypatch.setattr(
        recorder, "deterministic_sampling_decision", fail_sampling_decision
    )
    recorder.record_request_admission(
        "request-2",
        {"session_id": "session-2", "session_open_index": 2},
        stage_name="stage",
    )
    recorder.register_session_sampling_context(
        "request-3", ("session-3", 3), stage_name="stage"
    )
    assert "request-2" not in run_state.request_sampling_decisions
    assert "request-3" not in run_state.request_sampling_decisions
    assert "request-3" not in run_state.request_session_identities
    assert recorder.session_sampling_identity_for_request("request-1") is None
    calls_after_limit = (sampling_calls, timestamp_calls, snapshot_calls)
    recorder.emit(
        request_id="request-2",
        stage="stage",
        event_name="event",
        metadata={"value": 2},
    )
    assert (sampling_calls, timestamp_calls, snapshot_calls) == calls_after_limit
    monkeypatch.setattr(event_recorder_module, "get_recorder", lambda: recorder)
    monkeypatch.setattr(event_recorder_module, "snapshot_metadata", snapshot_metadata)
    RequestEventBuffer().capture(
        "deferred_event",
        [RequestEventSnapshot(request_id="request-3", metadata={})],
        {},
    )
    assert (sampling_calls, timestamp_calls, snapshot_calls) == calls_after_limit
    recorder.clear_request_sampling("request-1", stage_name="stage")
    assert "request-1" not in run_state.request_sampling_decisions
    assert "request-1" not in run_state.request_session_identities
    recorder.stop(wait=True)

    status = read_status(path)
    assert Path(path).read_bytes() == b""
    assert status["accepted_events"] == 1
    assert status["written_events"] == 0
    assert status["unwritten_events"] == 1
    assert status["dropped_events"] == 0
    assert "file_size_limit_reached" in status["incomplete_reasons"]
    assert status["complete"] is False


def test_runtime_stop_waits_in_writer_for_in_flight_emit(
    tmp_path: Path,
) -> None:
    recorder = RequestEventRecorder()
    path = recorder.start(
        "runtime-emit-race",
        str(tmp_path),
        "stage",
        mode="runtime",
        sampling_unit="request",
        sample_rate=1.0,
    )
    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()

    class BlockingMetadata(dict[str, object]):
        def items(self) -> ItemsView[str, object]:
            snapshot_entered.set()
            if not release_snapshot.wait(timeout=5):
                raise AssertionError("test did not release producer snapshot")
            return super().items()

    producer = threading.Thread(
        target=recorder.emit,
        kwargs={
            "request_id": "request",
            "stage": "stage",
            "event_name": "event",
            "metadata": BlockingMetadata(value=1),
        },
    )
    producer.start()
    assert snapshot_entered.wait(timeout=5)
    finalization = recorder.deactivate()
    assert finalization is not None
    assert finalization.completion_event.is_set() is False
    release_snapshot.set()
    producer.join(timeout=5)
    assert producer.is_alive() is False
    assert finalization.wait(timeout_seconds=5) is True

    status = read_status(path)
    assert status["accepted_events"] == 1
    assert status["written_events"] == 1
    assert status["unwritten_events"] == 0
    assert status["complete"] is True


def test_runtime_priority_failure_disables_capture_and_finalizes_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    scheduling_started = threading.Event()
    release_scheduling = threading.Event()
    scheduling_outcome_set = threading.Event()
    open_calls = 0
    original_open_event_file = recorder.open_event_file

    def fail_priority(run_state: RecorderRunState) -> None:
        scheduling_started.set()
        if not release_scheduling.wait(timeout=5):
            raise AssertionError("test did not release scheduling setup")
        run_state.writer_scheduling = "default"
        scheduling_outcome_set.set()

    def track_open_event_file(
        run_state: RecorderRunState,
    ) -> TextIOWrapper | None:
        nonlocal open_calls
        open_calls += 1
        return original_open_event_file(run_state)

    monkeypatch.setattr(recorder, "set_writer_scheduling", fail_priority)
    monkeypatch.setattr(recorder, "open_event_file", track_open_event_file)
    path = recorder.start(
        "priority-unavailable",
        str(tmp_path),
        "stage",
        mode="runtime",
        sampling_unit="request",
        sample_rate=1.0,
    )
    assert scheduling_started.wait(timeout=5)
    recorder.emit(
        request_id="before-failure",
        stage="stage",
        event_name="event",
    )
    release_scheduling.set()
    assert scheduling_outcome_set.wait(timeout=5)
    run_state = recorder.active_state
    assert run_state is not None
    writer_failure_deadline = time.monotonic() + 5
    while not run_state.writer_failed and time.monotonic() < writer_failure_deadline:
        threading.Event().wait(timeout=0.001)
    assert run_state.writer_failed is True
    recorder.record_request_admission("after-failure", {}, stage_name="stage")
    recorder.emit(
        request_id="after-failure",
        stage="stage",
        event_name="event",
    )
    assert "after-failure" not in run_state.request_sampling_decisions
    assert recorder.stop(wait=False) == path
    assert recorder.wait_for_finalizations(timeout_seconds=5) is True

    status = read_status(path)
    assert open_calls == 0
    assert Path(path).exists() is False
    assert status["accepted_events"] == 1
    assert status["written_events"] == 0
    assert status["unwritten_events"] == 1
    assert status["complete"] is False
    assert "writer_priority_unavailable" in status["incomplete_reasons"]


def test_runtime_event_file_open_failure_disables_capture_and_finalizes_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    scheduling_started = threading.Event()
    release_scheduling = threading.Event()

    def pause_scheduling(run_state: RecorderRunState) -> None:
        scheduling_started.set()
        if not release_scheduling.wait(timeout=5):
            raise AssertionError("test did not release scheduling setup")
        run_state.writer_scheduling = "SCHED_IDLE"

    monkeypatch.setattr(recorder, "set_writer_scheduling", pause_scheduling)
    path = recorder.start(
        "event-file-open-failure",
        str(tmp_path),
        "stage",
        mode="runtime",
        sampling_unit="request",
        sample_rate=1.0,
    )
    assert scheduling_started.wait(timeout=5)
    run_state = recorder.active_state
    assert run_state is not None
    recorder.emit(
        request_id="before-failure",
        stage="stage",
        event_name="event",
    )
    run_state.path.touch()
    release_scheduling.set()
    writer_failure_deadline = time.monotonic() + 5
    while not run_state.writer_failed and time.monotonic() < writer_failure_deadline:
        threading.Event().wait(timeout=0.001)
    assert run_state.writer_failed is True
    assert run_state.event_file_open_failures == 1
    recorder.emit(
        request_id="after-failure",
        stage="stage",
        event_name="event",
    )
    assert recorder.stop(wait=False) == path
    assert recorder.wait_for_finalizations(timeout_seconds=5) is True

    status = read_status(path)
    assert Path(path).read_text(encoding="utf-8") == ""
    assert status["event_file_open_failures"] == 1
    assert status["accepted_events"] == 1
    assert status["processed_events"] == 1
    assert status["written_events"] == 0
    assert status["unwritten_events"] == 1
    assert status["complete"] is False
    assert "event_file_open_failure" in status["incomplete_reasons"]


def test_runtime_deferred_sampling_and_colocated_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(
        "runtime-session-cleanup",
        str(tmp_path),
        "stage-a",
        mode="runtime",
        sampling_unit="session",
        sample_rate=1.0,
    )
    session_metadata = {
        "omni_session": {
            "operation": "append",
            "session_identity": {"id": "shared-session", "open_index": 1},
        }
    }
    recorder.record_request_admission(
        "shared-request",
        session_metadata,
        stage_name="stage-a",
    )
    recorder.record_request_admission(
        "shared-request",
        session_metadata,
        stage_name="stage-b",
    )
    recorder.clear_request_sampling("shared-request", stage_name="stage-a")
    assert recorder.active_state is not None
    assert recorder.active_state.request_sampling_decisions == {"shared-request": True}
    monkeypatch.setattr(event_recorder_module, "get_recorder", lambda: recorder)

    deferred_buffer = RequestEventBuffer()
    deferred_buffer.capture(
        "deferred_event",
        (
            RequestEventSnapshot(request_id="shared-request", metadata={}),
            RequestEventSnapshot(request_id="unsampled", metadata={}),
        ),
        {},
    )
    deferred_buffer.flush(stage="stage-b")
    recorder.emit(
        request_id="ordinary-request",
        stage="stage-b",
        event_name="ordinary_event",
    )
    recorder.clear_request_sampling("shared-request", stage_name="stage-b")
    assert recorder.active_state is not None
    assert recorder.active_state.request_sampling_decisions == {}
    assert recorder.active_state.request_session_identities == {}
    recorder.stop(wait=True)

    events = read_events(path)
    assert [(event["request_id"], event["event_name"]) for event in events] == [
        ("shared-request", "deferred_event")
    ]
    status = read_status(path)
    assert status["accepted_events"] == 1
    assert status["dropped_events"] == 0
    assert status["complete"] is True


def test_status_is_published_after_event_file_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder()
    path = recorder.start("close-before-status", str(tmp_path), "stage")
    close_finished = False
    original_close = recorder.close_event_file
    original_publish = recorder.publish_status

    def close_event_file(
        run_state: RecorderRunState, file_handle: TextIOWrapper | None
    ) -> None:
        nonlocal close_finished
        original_close(run_state, file_handle)
        close_finished = True

    def publish_status(run_state: RecorderRunState) -> bool:
        assert close_finished is True
        return original_publish(run_state)

    monkeypatch.setattr(recorder, "close_event_file", close_event_file)
    monkeypatch.setattr(recorder, "publish_status", publish_status)
    recorder.emit(request_id="request", stage="stage", event_name="event")
    recorder.stop(wait=True)

    assert read_status(path)["complete"] is True
