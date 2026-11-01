# SPDX-License-Identifier: Apache-2.0
"""Writer concurrency, drain, and failure tests for the request event recorder."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from io import TextIOWrapper
from pathlib import Path

import pytest

from sglang_omni.profiler.event_recorder import RequestEventRecorder
from sglang_omni.profiler.event_recorder_types import (
    ProducerBuffer,
    QueuedRequestEvent,
    RecorderRunState,
    RequestEventFinalization,
)
from tests.unit_test.profiler.event_recorder_test_utils import (
    read_events,
    read_status,
    read_status_file,
    stall_writer,
)


def test_slow_writer_does_not_block_producers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=4)
    path = recorder.start(run_id="slow", event_dir=str(tmp_path), stage="stage")
    recorder.emit(request_id="first", stage="stage", event_name="event")

    producer_finished = threading.Event()

    def produce_events() -> None:
        for event_number in range(4):
            recorder.emit(
                request_id=f"request-{event_number}",
                stage="stage",
                event_name="event",
            )
        producer_finished.set()

    producer_thread = threading.Thread(target=produce_events)
    producer_thread.start()
    assert producer_finished.wait(timeout=5)
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)
    finalization = recorder.deactivate()
    assert finalization is not None
    assert writer_entered.wait(timeout=5)
    release_writer.set()
    assert finalization.wait(timeout_seconds=5)
    producer_thread.join(timeout=5)
    assert producer_thread.is_alive() is False

    events = read_events(path)
    assert all(event["run_id"] == "slow" for event in events)


def test_concurrent_producers_preserve_counts_and_per_producer_order(
    tmp_path: Path,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=1024)
    path = recorder.start(run_id="concurrent", event_dir=str(tmp_path), stage="stage")
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


def test_retired_run_waits_for_events_accepted_during_idle_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopping cannot finalize a run while an accepted append is in flight."""
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(run_id="idle-race", event_dir=str(tmp_path), stage="stage")
    run_state = recorder.active_state
    assert run_state is not None
    recorder.emit(request_id="first", stage="stage", event_name="event")
    producer_buffer = recorder._producer_buffer_for(run_state)
    assert producer_buffer is not None

    append_entered = threading.Event()
    release_append = threading.Event()

    class BlockingEvents(list[QueuedRequestEvent]):
        def append(self, event: QueuedRequestEvent) -> None:
            append_entered.set()
            if not release_append.wait(timeout=5):
                raise AssertionError("test did not release the blocked append")
            else:
                pass
            super().append(event)

    producer_buffer.events = BlockingEvents(producer_buffer.events)

    def shared_producer_buffer(candidate: RecorderRunState) -> ProducerBuffer:
        del candidate
        return producer_buffer

    monkeypatch.setattr(recorder, "_producer_buffer_for", shared_producer_buffer)
    emit_thread = threading.Thread(
        target=recorder.emit,
        kwargs={
            "request_id": "second",
            "stage": "stage",
            "event_name": "event",
        },
    )
    emit_thread.start()
    assert append_entered.wait(timeout=5)

    stop_returned = threading.Event()
    stop_completion_states: list[bool] = []
    finalizations: list[RequestEventFinalization | None] = []

    def stop_run() -> None:
        stop_path = recorder.stop(wait=False)
        finalization = run_state.finalization if stop_path is not None else None
        finalizations.append(finalization)
        stop_completion_states.append(
            False if finalization is None else finalization.completion_event.is_set()
        )
        stop_returned.set()

    stop_thread = threading.Thread(target=stop_run)
    stop_thread.start()
    assert stop_returned.wait(timeout=5)
    assert stop_completion_states == [False]
    release_append.set()
    emit_thread.join(timeout=5)
    stop_thread.join(timeout=5)
    assert emit_thread.is_alive() is False
    assert stop_thread.is_alive() is False
    finalization = finalizations[0]
    assert isinstance(finalization, RequestEventFinalization)
    assert finalization.wait(timeout_seconds=5)
    status = read_status(path)
    assert [event["request_id"] for event in read_events(path)] == [
        "first",
        "second",
    ]
    assert status["accepted_events"] == 2
    assert status["processed_events"] == 2
    assert status["written_events"] == 2
    assert status["complete"] is True


def test_idle_writer_does_not_open_new_run_ahead_of_queued_old_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path_a = recorder.start(
        run_id="run-a", event_dir=str(tmp_path / "run-a"), stage="stage"
    )
    recorder.emit(request_id="request-a", stage="stage", event_name="event")
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)
    finalization_a = recorder.deactivate(run_id="run-a")
    assert finalization_a is not None
    assert writer_entered.wait(timeout=5)
    with pytest.raises(RuntimeError, match="still draining"):
        recorder.start(run_id="run-b", event_dir=str(tmp_path / "run-b"), stage="stage")
    release_writer.set()
    assert finalization_a.wait(timeout_seconds=5)

    path_b = recorder.start(
        run_id="run-b", event_dir=str(tmp_path / "run-b"), stage="stage"
    )
    recorder.emit(request_id="request-b", stage="stage", event_name="event")
    finalization_b = recorder.deactivate(run_id="run-b")
    assert finalization_b is not None
    assert finalization_b.wait(timeout_seconds=5)
    assert [event["request_id"] for event in read_events(path_a)] == ["request-a"]
    assert [event["request_id"] for event in read_events(path_b)] == ["request-b"]
    assert read_status_file(finalization_a.status_path)["complete"] is True
    assert read_status_file(finalization_b.status_path)["complete"] is True


def test_full_queue_drops_events_and_marks_status_incomplete(
    tmp_path: Path,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=1)
    path = recorder.start(run_id="full", event_dir=str(tmp_path), stage="stage")
    recorder.emit(request_id="first", stage="stage", event_name="event")
    for event_number in range(8):
        recorder.emit(
            request_id=f"request-{event_number}",
            stage="stage",
            event_name="event",
        )
    finalization = recorder.deactivate()
    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is False
    status = read_status(path)
    assert status["accepted_events"] == status["written_events"]
    assert status["dropped_events"] > 0
    assert status["complete"] is False
    assert "producer_buffer_overflow" in status["incomplete_reasons"]


def test_stop_drains_accepted_events_into_parseable_jsonl(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=128)
    path = recorder.start(run_id="drain", event_dir=str(tmp_path), stage="stage")
    for event_number in range(64):
        recorder.emit(
            request_id=f"request-{event_number}",
            stage="stage",
            event_name="event",
            metadata={"number": event_number},
        )

    recorder.stop(wait=True)
    events = read_events(path)
    assert len(events) == 64
    assert all(event["run_id"] == "drain" for event in events)
    status = read_status(path)
    assert status["accepted_events"] == 64
    assert status["processed_events"] == 64
    assert status["written_events"] == 64
    assert status["complete"] is True


def test_run_rotation_does_not_mix_events(
    tmp_path: Path,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=16)
    old_path = recorder.start(
        run_id="old", event_dir=str(tmp_path / "old"), stage="stage"
    )
    recorder.emit(request_id="old-request", stage="stage", event_name="event")
    recorder.stop(wait=True)

    new_path = recorder.start(
        run_id="new", event_dir=str(tmp_path / "new"), stage="stage"
    )
    recorder.emit(request_id="new-request", stage="stage", event_name="event")
    recorder.stop(wait=True)

    old_events = read_events(old_path)
    new_events = read_events(new_path)
    assert [event["request_id"] for event in old_events] == ["old-request"]
    assert [event["request_id"] for event in new_events] == ["new-request"]
    assert {event["run_id"] for event in old_events} == {"old"}
    assert {event["run_id"] for event in new_events} == {"new"}


def test_writer_replacement_after_atomic_exit_ownership_release(
    tmp_path: Path,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    first_path = recorder.start("first", str(tmp_path / "first"), "stage")
    first_finalization = recorder.deactivate()
    assert first_finalization is not None
    assert first_finalization.wait(timeout_seconds=5)
    second_path = recorder.start("second", str(tmp_path / "second"), "stage")
    recorder.emit(request_id="new", stage="stage", event_name="event")
    second_finalization = recorder.deactivate()
    assert second_finalization is not None
    assert second_finalization.wait(timeout_seconds=5)
    assert [event["run_id"] for event in read_events(first_path)] == []
    assert [event["run_id"] for event in read_events(second_path)] == ["second"]


def test_recorder_activity_hints_do_not_acquire_mutex() -> None:
    class LockMustNotBeAcquired:
        def __enter__(self) -> None:
            raise AssertionError("inactive recorder acquired its mutex")

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

    recorder = RequestEventRecorder()
    recorder._control_lock = LockMustNotBeAcquired()  # type: ignore[assignment]

    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"value": object()},
    )
    assert recorder.is_active() is False
    assert recorder.active_run_id() is None
    assert recorder.capture_active_run_timestamp_ns() is None

    recorder.active_state = RecorderRunState(
        run_id="active",
        stage="stage",
        stages={"stage"},
        active_stages={"stage"},
        path=Path("events.jsonl"),
        status_path=Path("run_status.json"),
        finalization=RequestEventFinalization(
            path="events.jsonl",
            status_path="run_status.json",
            completion_event=threading.Event(),
        ),
    )
    assert recorder.is_active() is True


def test_writer_failure_completes_active_and_retired_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    old_path = recorder.start("old", str(tmp_path / "old"), "stage")

    def crashing_write(
        queued_event: QueuedRequestEvent,
        file_handle: TextIOWrapper | None,
    ) -> TextIOWrapper | None:
        del queued_event, file_handle
        raise RuntimeError("synthetic writer failure")

    monkeypatch.setattr(recorder, "write_queued_event", crashing_write)
    recorder.emit(request_id="old", stage="stage", event_name="event")
    old_finalization = recorder.deactivate()
    assert old_finalization is not None
    assert old_finalization.wait(timeout_seconds=5) is False
    assert recorder.is_active() is False
    old_status = read_status_file(old_finalization.status_path)
    assert old_status["complete"] is False
    assert old_status["processed_events"] == old_status["accepted_events"] == 1
    assert old_status["written_events"] == 0
    assert "writer_callback_failure" in old_status["incomplete_reasons"]
    assert read_status(old_path)["unwritten_events"] >= 1

    monkeypatch.undo()
    new_path = recorder.start("new", str(tmp_path / "new"), "stage")
    recorder.emit(request_id="new", stage="stage", event_name="event")
    new_finalization = recorder.deactivate()
    assert new_finalization is not None
    assert new_finalization.wait(timeout_seconds=5)
    new_status = read_status_file(new_finalization.status_path)
    assert new_status["complete"] is True
    assert new_status["written_events"] == 1
    assert [event["request_id"] for event in read_events(new_path)] == ["new"]


def test_rotation_backlog_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    recorder.start(run_id="run-0", event_dir=str(tmp_path / "run-0"), stage="stage")
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)
    recorder.emit(request_id="request", stage="stage", event_name="event")
    finalization = recorder.deactivate()
    assert finalization is not None
    assert writer_entered.wait(timeout=5)
    with pytest.raises(RuntimeError, match="still draining"):
        recorder.start(run_id="run-1", event_dir=str(tmp_path / "run-1"), stage="stage")
    release_writer.set()
    assert finalization.wait(timeout_seconds=5)
    assert recorder.wait_for_finalizations(timeout_seconds=5)

    second_path = recorder.start(
        run_id="run-1", event_dir=str(tmp_path / "run-1"), stage="stage"
    )
    recorder.emit(request_id="second", stage="stage", event_name="event")
    recorder.stop(wait=True)
    assert [event["request_id"] for event in read_events(second_path)] == ["second"]


def test_nonblocking_stop_returns_while_writer_is_stalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    recorder.start(run_id="async-stop", event_dir=str(tmp_path), stage="stage")
    run_state = recorder.active_state
    assert run_state is not None
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)
    recorder.emit(request_id="request", stage="stage", event_name="event")
    stop_path = recorder.stop(wait=False)
    assert stop_path is not None
    finalization = run_state.finalization
    assert writer_entered.wait(timeout=5)
    assert finalization.completion_event.is_set() is False
    release_writer.set()
    assert finalization is not None
    assert finalization.wait(timeout_seconds=5)


def test_active_capture_has_no_writer_io_or_json_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sglang_omni.profiler import event_recorder_writer

    recorder = RequestEventRecorder(queue_capacity=8)
    json_calls: list[bool] = []
    writer_calls: list[str] = []
    original_dumps = event_recorder_writer.json.dumps
    original_write = recorder.write_queued_event

    def observe_dumps(*args: object, **kwargs: object) -> str:
        json_calls.append(True)
        return original_dumps(*args, **kwargs)

    def observe_write(
        queued_event: QueuedRequestEvent,
        file_handle: TextIOWrapper | None,
    ) -> TextIOWrapper | None:
        writer_calls.append(queued_event.event.request_id)
        return original_write(queued_event, file_handle)

    monkeypatch.setattr(event_recorder_writer.json, "dumps", observe_dumps)
    monkeypatch.setattr(recorder, "write_queued_event", observe_write)
    path = recorder.start(run_id="active", event_dir=str(tmp_path), stage="stage")
    recorder.emit(request_id="request", stage="stage", event_name="event")

    assert recorder.writer_thread is None
    assert recorder.writer_start_count == 0
    assert not hasattr(recorder, "event_queue")
    assert not hasattr(recorder, "lock")
    assert Path(path).exists() is False
    assert list(tmp_path.glob("run_status_*.json")) == []
    assert json_calls == []
    assert writer_calls == []

    recorder.stop(wait=True)
    assert recorder.writer_start_count == 1
    assert json_calls
    assert writer_calls == ["request"]


def test_cached_producer_emit_does_not_acquire_control_lock(
    tmp_path: Path,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(run_id="cached", event_dir=str(tmp_path), stage="stage")
    recorder.emit(request_id="first", stage="stage", event_name="event")
    original_lock = recorder._control_lock

    class LockMustNotBeAcquired:
        def __enter__(self) -> None:
            raise AssertionError("cached producer acquired the control lock")

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

    recorder._control_lock = LockMustNotBeAcquired()  # type: ignore[assignment]
    try:
        recorder.emit(request_id="second", stage="stage", event_name="event")
    finally:
        recorder._control_lock = original_lock

    recorder.stop(wait=True)
    assert [event["request_id"] for event in read_events(path)] == [
        "first",
        "second",
    ]


def test_disabled_emit_returns_before_timestamp_or_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.profiler import event_recorder_writer

    recorder = RequestEventRecorder()
    calls: list[str] = []

    def fail_time() -> int:
        calls.append("time")
        raise AssertionError("disabled emit captured a timestamp")

    def fail_snapshot(metadata: Mapping[str, object] | None) -> dict[str, object]:
        del metadata
        calls.append("snapshot")
        raise AssertionError("disabled emit copied metadata")

    monkeypatch.setattr(event_recorder_writer.time, "time_ns", fail_time)
    monkeypatch.setattr(event_recorder_writer, "snapshot_metadata", fail_snapshot)
    recorder.emit(
        request_id="disabled",
        stage="stage",
        event_name="event",
        metadata={"value": object()},
    )

    assert calls == []
    assert recorder.writer_thread is None
    assert not hasattr(recorder._producer_local, "handles")


def test_first_registration_race_is_accounted_after_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(
        run_id="registration-race", event_dir=str(tmp_path), stage="stage"
    )
    run_state = recorder.active_state
    assert run_state is not None
    registration_entered = threading.Event()
    release_registration = threading.Event()
    original_buffer_for = recorder._producer_buffer_for

    def blocked_buffer_for(candidate: RecorderRunState) -> ProducerBuffer | None:
        registration_entered.set()
        if not release_registration.wait(timeout=5):
            raise AssertionError("test did not release producer registration")
        else:
            pass
        return original_buffer_for(candidate)

    monkeypatch.setattr(recorder, "_producer_buffer_for", blocked_buffer_for)
    producer = threading.Thread(
        target=recorder.emit,
        kwargs={
            "request_id": "late",
            "stage": "stage",
            "event_name": "event",
        },
    )
    producer.start()
    assert registration_entered.wait(timeout=5)

    stop_path = recorder.stop(wait=False)
    assert stop_path == path
    assert run_state.finalization.completion_event.is_set() is False
    release_registration.set()
    producer.join(timeout=5)
    assert producer.is_alive() is False
    assert run_state.finalization.wait(timeout_seconds=5) is False

    status = read_status_file(run_state.finalization.status_path)
    assert status["accepted_events"] == 0
    assert status["dropped_events"] == 1
    assert status["complete"] is False
    assert "producer_buffer_registration_after_stop" in status["incomplete_reasons"]


def test_producer_registration_limit_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sglang_omni.profiler import event_recorder_writer

    monkeypatch.setattr(event_recorder_writer, "MAX_PRODUCER_BUFFERS", 2)
    recorder = RequestEventRecorder(queue_capacity=1)
    path = recorder.start(
        run_id="registration-limit", event_dir=str(tmp_path), stage="stage"
    )

    def emit_from_thread(index: int) -> None:
        recorder.emit(
            request_id=f"request-{index}",
            stage="stage",
            event_name="event",
        )

    for index in range(3):
        producer = threading.Thread(target=emit_from_thread, args=(index,))
        producer.start()
        producer.join(timeout=5)
        assert producer.is_alive() is False

    recorder.stop(wait=True)
    status = read_status(path)
    assert status["accepted_events"] == 2
    assert status["dropped_events"] == 1
    assert status["complete"] is False
    assert "producer_buffer_registration_limit" in status["incomplete_reasons"]


def test_status_is_published_after_event_file_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(run_id="close-order", event_dir=str(tmp_path), stage="stage")
    tracked_handles: list[TextIOWrapper] = []
    original_open = recorder.open_event_file
    original_publish = recorder._publish_status

    def observe_open(run_state: RecorderRunState) -> TextIOWrapper | None:
        file_handle = original_open(run_state)
        if file_handle is not None:
            tracked_handles.append(file_handle)
        else:
            pass
        return file_handle

    def observe_publish(run_state: RecorderRunState) -> bool:
        assert tracked_handles
        assert tracked_handles[-1].closed is True
        return original_publish(run_state)

    monkeypatch.setattr(recorder, "open_event_file", observe_open)
    monkeypatch.setattr(recorder, "_publish_status", observe_publish)
    recorder.emit(request_id="request", stage="stage", event_name="event")
    finalization = recorder.deactivate()
    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is True
    assert Path(path).exists()
