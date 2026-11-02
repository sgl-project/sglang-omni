# SPDX-License-Identifier: Apache-2.0
"""Producer buffering and stop/write concurrency tests."""

from __future__ import annotations

import threading
from collections.abc import ItemsView
from io import TextIOWrapper
from pathlib import Path

import pytest

from sglang_omni.profiler import event_recorder_writer
from sglang_omni.profiler.event_recorder import RequestEventRecorder
from sglang_omni.profiler.event_recorder_types import (
    MAX_PRODUCER_BUFFERS,
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
