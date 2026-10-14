# SPDX-License-Identifier: Apache-2.0
"""Writer concurrency, drain, and failure tests for the request event recorder."""

from __future__ import annotations

import threading
from io import TextIOWrapper
from pathlib import Path

import pytest

from sglang_omni.profiler.event_recorder import RequestEventRecorder
from sglang_omni.profiler.event_recorder_types import (
    QueuedRequestEvent,
    RecorderRunState,
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
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)
    recorder.emit(request_id="first", stage="stage", event_name="event")
    assert writer_entered.wait(timeout=5)

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
    try:
        assert producer_finished.wait(timeout=5)
        finalization = recorder.deactivate()
        assert finalization is not None
    finally:
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
    recorder.stop()

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
    """An idle observation cannot finalize a run while its accepted count lags."""
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(run_id="idle-race", event_dir=str(tmp_path), stage="stage")
    run_state = recorder.active_state
    assert run_state is not None

    first_written = threading.Event()
    idle_entered = threading.Event()
    release_idle = threading.Event()
    open_calls: list[int] = []
    original_open = recorder.open_event_file
    original_idle = recorder.handle_writer_idle
    original_write = recorder.write_queued_event

    def counting_open(run_state: RecorderRunState) -> TextIOWrapper | None:
        open_calls.append(run_state.generation_id)
        return original_open(run_state)

    def observe_first_write(
        queued_event: QueuedRequestEvent,
        file_handle: TextIOWrapper | None,
    ) -> TextIOWrapper | None:
        result = original_write(queued_event, file_handle)
        if queued_event.event.request_id == "first":
            first_written.set()
        else:
            pass
        return result

    def pause_idle(
        open_state: RecorderRunState | None,
        file_handle: TextIOWrapper | None,
    ) -> tuple[RecorderRunState | None, TextIOWrapper | None]:
        if first_written.is_set() and not idle_entered.is_set():
            idle_entered.set()
            assert release_idle.wait(timeout=5)
        else:
            pass
        return original_idle(open_state, file_handle)

    monkeypatch.setattr(recorder, "open_event_file", counting_open)
    monkeypatch.setattr(recorder, "write_queued_event", observe_first_write)
    monkeypatch.setattr(recorder, "handle_writer_idle", pause_idle)

    recorder.emit(request_id="first", stage="stage", event_name="event")
    assert idle_entered.wait(timeout=5)

    # This enqueue happens after event #1 was processed but before the writer
    # is allowed to complete its idle decision.
    recorder.emit(request_id="second", stage="stage", event_name="event")
    finalization = recorder.deactivate(run_id="idle-race")
    assert finalization is not None
    release_idle.set()

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
    assert open_calls == [run_state.generation_id]


def test_full_queue_drops_events_and_marks_status_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=1)
    path = recorder.start(run_id="full", event_dir=str(tmp_path), stage="stage")
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)
    recorder.emit(request_id="first", stage="stage", event_name="event")
    assert writer_entered.wait(timeout=5)
    for event_number in range(8):
        recorder.emit(
            request_id=f"request-{event_number}",
            stage="stage",
            event_name="event",
        )
    finalization = recorder.deactivate()
    assert finalization is not None
    try:
        release_writer.set()
        assert finalization.wait(timeout_seconds=5)
    finally:
        release_writer.set()

    status = read_status(path)
    assert status["accepted_events"] == status["written_events"]
    assert status["dropped_events"] > 0
    assert status["complete"] is False
    assert "queue_overflow" in status["incomplete_reasons"]


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

    recorder.stop()
    events = read_events(path)
    assert len(events) == 64
    assert all(event["run_id"] == "drain" for event in events)
    status = read_status(path)
    assert status["accepted_events"] == 64
    assert status["processed_events"] == 64
    assert status["written_events"] == 64
    assert status["complete"] is True


def test_run_rotation_does_not_mix_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=16)
    old_path = recorder.start(
        run_id="old", event_dir=str(tmp_path / "old"), stage="stage"
    )
    writer_entered = threading.Event()
    release_writer = threading.Event()
    original_write = recorder.write_queued_event

    def stalled_old_write(
        queued_event: QueuedRequestEvent,
        file_handle: TextIOWrapper | None,
    ) -> TextIOWrapper | None:
        if queued_event.event.run_id == "old":
            writer_entered.set()
            if not release_writer.wait(timeout=5):
                raise AssertionError("test did not release the stalled writer")
            else:
                pass
        else:
            pass
        return original_write(queued_event, file_handle)

    monkeypatch.setattr(recorder, "write_queued_event", stalled_old_write)
    recorder.emit(request_id="old-request", stage="stage", event_name="event")
    assert writer_entered.wait(timeout=5)

    try:
        new_path = recorder.start(
            run_id="new", event_dir=str(tmp_path / "new"), stage="stage"
        )
        recorder.emit(request_id="new-request", stage="stage", event_name="event")
    finally:
        release_writer.set()
    recorder.stop()
    assert recorder.wait_for_finalizations(timeout_seconds=5)

    old_events = read_events(old_path)
    new_events = read_events(new_path)
    assert [event["request_id"] for event in old_events] == ["old-request"]
    assert [event["request_id"] for event in new_events] == ["new-request"]
    assert {event["run_id"] for event in old_events} == {"old"}
    assert {event["run_id"] for event in new_events} == {"new"}


def test_writer_replacement_after_atomic_exit_ownership_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    first_path = recorder.start("first", str(tmp_path), "stage")
    first_finalization = recorder.deactivate()
    assert first_finalization is not None

    exit_decided = threading.Event()
    release_old_writer = threading.Event()
    original_should_exit = recorder.writer_should_exit

    def gated_should_exit() -> bool:
        should_exit = original_should_exit()
        if should_exit:
            exit_decided.set()
            if not release_old_writer.wait(timeout=5):
                raise AssertionError("test did not release the retiring writer")
            else:
                pass
        else:
            pass
        return should_exit

    monkeypatch.setattr(recorder, "writer_should_exit", gated_should_exit)
    try:
        assert exit_decided.wait(timeout=5)
        second_path = recorder.start("second", str(tmp_path), "stage")
        recorder.emit(request_id="new", stage="stage", event_name="event")
        second_finalization = recorder.deactivate()
        assert second_finalization is not None
    finally:
        release_old_writer.set()

    assert first_finalization.wait(timeout_seconds=5)
    assert second_finalization.wait(timeout_seconds=5)
    assert [event["run_id"] for event in read_events(first_path)] == []
    assert [event["run_id"] for event in read_events(second_path)] == ["second"]


def test_inactive_recorder_does_not_acquire_mutex() -> None:
    class LockMustNotBeAcquired:
        def __enter__(self) -> None:
            raise AssertionError("inactive recorder acquired its mutex")

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

    recorder = RequestEventRecorder()
    recorder.lock = LockMustNotBeAcquired()  # type: ignore[assignment]

    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"value": object()},
    )
    assert recorder.is_active() is False
    assert recorder.active_run_id() is None
    assert recorder.capture_active_run_timestamp_ns() is None
    assert recorder.capture_active_run_identity_timestamp_ns() is None


def test_writer_failure_completes_active_and_retired_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    old_path = recorder.start("old", str(tmp_path / "old"), "stage")
    writer_entered = threading.Event()
    release_writer = threading.Event()

    def crashing_write(
        queued_event: QueuedRequestEvent,
        file_handle: TextIOWrapper | None,
    ) -> TextIOWrapper | None:
        del queued_event, file_handle
        writer_entered.set()
        if not release_writer.wait(timeout=5):
            raise AssertionError("test did not release the crashing writer")
        else:
            pass
        raise RuntimeError("synthetic writer failure")

    monkeypatch.setattr(recorder, "write_queued_event", crashing_write)
    recorder.emit(request_id="old", stage="stage", event_name="event")
    assert writer_entered.wait(timeout=5)

    old_finalization = recorder.deactivate()
    assert old_finalization is not None
    new_path = recorder.start("new", str(tmp_path / "new"), "stage")
    new_finalization = recorder.active_state.finalization  # type: ignore[union-attr]
    recorder.emit(request_id="new", stage="stage", event_name="event")
    try:
        release_writer.set()
        assert old_finalization.wait(timeout_seconds=5)
        assert new_finalization.wait(timeout_seconds=5)
    finally:
        release_writer.set()

    assert recorder.is_active() is False
    assert read_status_file(old_finalization.status_path)["complete"] is False
    assert read_status_file(new_finalization.status_path)["complete"] is False
    assert (
        "writer_thread_failure"
        in read_status_file(old_finalization.status_path)["incomplete_reasons"]
    )
    assert (
        "writer_thread_failure"
        in read_status_file(new_finalization.status_path)["incomplete_reasons"]
    )
    assert read_status(old_path)["unwritten_events"] >= 1
    assert read_status(new_path)["unwritten_events"] >= 1


def test_rotation_backlog_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    recorder.start(run_id="run-0", event_dir=str(tmp_path / "run-0"), stage="stage")
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)
    recorder.emit(request_id="request", stage="stage", event_name="event")
    assert writer_entered.wait(timeout=5)

    for run_number in range(1, 5):
        recorder.start(
            run_id=f"run-{run_number}",
            event_dir=str(tmp_path / f"run-{run_number}"),
            stage="stage",
        )
    with pytest.raises(RuntimeError, match="waiting for finalization"):
        recorder.start(
            run_id="run-5",
            event_dir=str(tmp_path / "run-5"),
            stage="stage",
        )

    finalization = recorder.deactivate()
    assert finalization is not None
    try:
        release_writer.set()
        assert finalization.wait(timeout_seconds=5)
    finally:
        release_writer.set()
    assert recorder.wait_for_finalizations(timeout_seconds=5)


def test_nonblocking_stop_returns_while_writer_is_stalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    recorder.start(run_id="async-stop", event_dir=str(tmp_path), stage="stage")
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)
    recorder.emit(request_id="request", stage="stage", event_name="event")
    assert writer_entered.wait(timeout=5)

    finalization = recorder.deactivate()
    assert finalization is not None
    try:
        assert finalization.completion_event.is_set() is False
    finally:
        release_writer.set()
    assert finalization.wait(timeout_seconds=5)
