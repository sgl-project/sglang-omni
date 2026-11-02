# SPDX-License-Identifier: Apache-2.0
"""Producer buffering and stop/write concurrency tests."""

from __future__ import annotations

import threading
from collections.abc import ItemsView
from io import TextIOWrapper
from pathlib import Path

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
    MAX_PRODUCER_BUFFERS,
    RUNTIME_WRITER_MAX_EVENTS_PER_ROUND,
    RUNTIME_WRITER_POLL_INTERVAL_SECONDS,
    RecorderRunState,
)
from tests.unit_test.profiler.event_recorder_test_utils import (
    read_events,
    read_status,
    stall_writer,
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


def test_producer_buffer_limit_is_counted(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=2)
    path = recorder.start("producer-limit", str(tmp_path), "stage")
    barrier = threading.Barrier(MAX_PRODUCER_BUFFERS + 2)
    rejected_ready = threading.Event()
    release_rejected = threading.Event()
    rejected_thread_id: int | None = None

    def produce(producer_number: int) -> None:
        nonlocal rejected_thread_id
        barrier.wait(timeout=5)
        recorder.emit(
            request_id=f"request-{producer_number}",
            stage="stage",
            event_name="event",
        )
        if getattr(recorder.producer_local, "buffer", None) is None:
            rejected_thread_id = threading.get_ident()
            rejected_ready.set()
            if not release_rejected.wait(timeout=5):
                raise AssertionError("test did not release the rejected producer")
            else:
                pass
            for event_number in range(3):
                recorder.emit(
                    request_id=f"rejected-{event_number}",
                    stage="stage",
                    event_name="event",
                )
        else:
            pass

    producers = [
        threading.Thread(target=produce, args=(producer_number,))
        for producer_number in range(MAX_PRODUCER_BUFFERS + 1)
    ]
    for producer in producers:
        producer.start()
    barrier.wait(timeout=5)
    assert rejected_ready.wait(timeout=5)
    accepted_producers = [
        producer for producer in producers if producer.ident != rejected_thread_id
    ]
    for producer in accepted_producers:
        producer.join(timeout=5)
        assert producer.is_alive() is False

    rejected_producer = next(
        producer for producer in producers if producer.ident == rejected_thread_id
    )
    with pytest.MonkeyPatch.context() as control_lock_patch:
        control_lock_patch.setattr(recorder, "control_lock", object())
        release_rejected.set()
        rejected_producer.join(timeout=5)
        assert rejected_producer.is_alive() is False

    recorder.stop(wait=True)
    status = read_status(path)
    assert status["accepted_events"] == MAX_PRODUCER_BUFFERS
    assert status["dropped_events"] == 4
    assert status["complete"] is False
    assert status["producer_buffer_count"] == MAX_PRODUCER_BUFFERS
    assert sorted(status["producer_buffer_event_counts"]) == [1] * MAX_PRODUCER_BUFFERS


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


def test_runtime_request_sampling_matches_independent_recorders(
    tmp_path: Path,
) -> None:
    sampled_request_ids: list[set[str]] = []
    for recorder_number in range(2):
        recorder = RequestEventRecorder(queue_capacity=256)
        path = recorder.start(
            "stable-request-sampling",
            str(tmp_path / str(recorder_number)),
            "coordinator",
            mode="runtime",
            sampling_unit="request",
            sample_rate=0.35,
        )
        assert recorder.active_state is not None
        for request_number in range(12):
            request_id = f"unadmitted-{request_number}"
            decision = recorder.deterministic_sampling_decision(
                recorder.active_state, request_id
            )
            assert (
                recorder.should_sample_request(recorder.active_state, request_id)
                is decision
            )
            recorder.emit(
                request_id=request_id,
                stage="coordinator",
                event_name="session_unit_ready",
            )
        assert not any(
            request_id.startswith("unadmitted-")
            for request_id in recorder.active_state.request_sampling_decisions
        )
        recorder.record_request_admission(
            "later-admitted",
            {},
            stage_name="coordinator",
        )
        assert "later-admitted" in recorder.active_state.request_sampling_decisions
        recorder.clear_request_sampling("later-admitted", stage_name="coordinator")
        assert "later-admitted" not in recorder.active_state.request_sampling_decisions
        for request_number in range(80):
            request_id = f"request-{request_number}"
            recorder.record_request_admission(
                request_id,
                {},
                stage_name="coordinator",
            )
            recorder.emit(
                request_id=request_id,
                stage="coordinator",
                event_name="request_admission",
            )
            recorder.emit(
                request_id=request_id,
                stage="stage",
                event_name="stage_complete",
            )
        recorder.stop(wait=True)
        sampled_request_ids.append({event["request_id"] for event in read_events(path)})
        status = read_status(path)
        assert status["dropped_events"] == 0
        assert status["complete"] is True

    assert sampled_request_ids[0] == sampled_request_ids[1]
    assert sampled_request_ids[0]


def test_runtime_final_drain_is_batched_paced_and_asynchronous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=96)
    writer_entered = threading.Event()
    release_setup = threading.Event()
    pacing_entered = threading.Event()
    release_pacing = threading.Event()
    final_batch_sizes: list[int] = []
    pacing_intervals: list[float] = []

    def pause_writer(run_state: RecorderRunState) -> None:
        writer_entered.set()
        if not release_setup.wait(timeout=5):
            raise AssertionError("test did not release the runtime writer")
        else:
            pass
        run_state.writer_scheduling = "default"

    original_drain = recorder.drain_runtime_buffers

    def track_drain(
        run_state: RecorderRunState, file_handle: TextIOWrapper | None
    ) -> int:
        stopping = run_state.stop_requested.is_set()
        count = original_drain(run_state, file_handle)
        if stopping and count:
            assert count <= RUNTIME_WRITER_MAX_EVENTS_PER_ROUND
            final_batch_sizes.append(count)
        else:
            pass
        return count

    def pause_between_batches(interval: float) -> None:
        pacing_intervals.append(interval)
        pacing_entered.set()
        if not release_pacing.wait(timeout=5):
            raise AssertionError("test did not release final-drain pacing")
        else:
            pass

    monkeypatch.setattr(recorder, "set_writer_scheduling", pause_writer)
    monkeypatch.setattr(recorder, "drain_runtime_buffers", track_drain)
    monkeypatch.setattr(event_recorder_writer.time, "sleep", pause_between_batches)
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
    assert pacing_entered.wait(timeout=5)
    assert finalization.completion_event.is_set() is False
    assert final_batch_sizes == [RUNTIME_WRITER_MAX_EVENTS_PER_ROUND]

    release_pacing.set()
    assert finalization.wait(timeout_seconds=5) is True

    assert final_batch_sizes == [
        RUNTIME_WRITER_MAX_EVENTS_PER_ROUND,
        RUNTIME_WRITER_MAX_EVENTS_PER_ROUND,
        1,
    ]
    assert pacing_intervals == [
        RUNTIME_WRITER_POLL_INTERVAL_SECONDS,
        RUNTIME_WRITER_POLL_INTERVAL_SECONDS,
    ]
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


def test_runtime_session_sampling_is_consistent_across_stage_processes(
    tmp_path: Path,
) -> None:
    selected_session_ids: list[set[str]] = []
    for process_number in range(3):
        recorder = RequestEventRecorder(queue_capacity=64)
        path = recorder.start(
            "stage-process-sampling",
            str(tmp_path / str(process_number)),
            "coordinator",
            mode="runtime",
            sampling_unit="session",
            sample_rate=0.3,
        )
        for session_number in range(60):
            session_id = f"session-{session_number}"
            request_id = f"operation-{session_number}"
            session_metadata = {
                "omni_session": {
                    "operation": "append",
                    "session_identity": {
                        "id": session_id,
                        "open_index": 2,
                    },
                }
            }
            recorder.record_request_admission(
                request_id,
                session_metadata,
                stage_name="stage",
            )
            recorder.emit(
                request_id=request_id,
                stage="stage",
                event_name="stage_input_received",
            )
        recorder.stop(wait=True)
        selected_session_ids.append(
            {
                event["request_id"].removeprefix("operation-")
                for event in read_events(path)
            }
        )

    assert selected_session_ids[0] == selected_session_ids[1] == selected_session_ids[2]


def test_runtime_writer_concurrency_preserves_all_events(
    tmp_path: Path,
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

    def produce_events(producer_number: int) -> None:
        for event_number in range(events_per_producer):
            recorder.emit(
                request_id=f"request-{producer_number}",
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
    event_keys = {
        (event["request_id"], event["metadata"]["number"]) for event in events
    }
    expected_keys = {
        (f"request-{producer_number}", event_number)
        for producer_number in range(producer_count)
        for event_number in range(events_per_producer)
    }
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
        else:
            pass
        run_state.writer_scheduling = "default"

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
        else:
            pass

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
        sampling_unit="request",
        sample_rate=1.0,
        max_event_file_bytes=1,
    )
    recorder.emit(
        request_id="request-1",
        stage="stage",
        event_name="event",
        metadata={"value": 1},
    )
    assert limit_reached.wait(timeout=5)
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
    recorder.stop(wait=True)

    status = read_status(path)
    assert Path(path).read_bytes() == b""
    assert Path(path).stat().st_size <= 1
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

    status = read_status(path)
    assert status["accepted_events"] == 1
    assert status["written_events"] == 1
    assert status["unwritten_events"] == 0
    assert status["complete"] is True


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
    closing_metadata = {
        "omni_session": {
            "operation": "close",
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
    recorder.record_request_admission(
        "closing-request",
        closing_metadata,
        stage_name="stage-a",
    )
    recorder.record_request_admission(
        "closing-request",
        closing_metadata,
        stage_name="stage-b",
    )
    recorder.clear_request_sampling("closing-request", stage_name="stage-a")
    recorder.clear_request_sampling("closing-request", stage_name="stage-b")
    assert recorder.active_state is not None
    assert recorder.active_state.request_sampling_decisions == {"shared-request": True}
    assert recorder.active_state.session_sampling_decisions == {
        ("shared-session", 1): True
    }
    assert recorder.active_state.session_sampling_clear_when_idle == {
        ("shared-session", 1)
    }
    recorder.clear_session_sampling("shared-session", 1)
    assert recorder.active_state.session_sampling_decisions == {
        ("shared-session", 1): True
    }
    recorder.clear_request_sampling("shared-request", stage_name="stage-a")
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
    recorder.clear_request_sampling(
        "shared-request",
        stage_name="stage-b",
        clear_session=True,
    )
    assert recorder.active_state is not None
    assert recorder.active_state.request_sampling_decisions == {}
    assert recorder.active_state.request_session_identities == {}
    assert recorder.active_state.request_session_operations == {}
    assert recorder.active_state.session_sampling_decisions == {}
    assert recorder.active_state.session_sampling_clear_when_idle == set()
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
