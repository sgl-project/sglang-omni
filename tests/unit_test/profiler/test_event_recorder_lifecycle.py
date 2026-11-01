# SPDX-License-Identifier: Apache-2.0
"""Run cutoff, drain, and status publication tests."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from sglang_omni.profiler import event_recorder
from sglang_omni.profiler.event_recorder import (
    RequestEventBuffer,
    RequestEventRecorder,
    RequestEventSnapshot,
    get_recorder,
)
from tests.unit_test.profiler.event_recorder_test_utils import (
    fail_status_replace,
    read_events,
    read_status_file,
)


def test_zero_event_run_publishes_complete_status(tmp_path: Path) -> None:
    recorder = RequestEventRecorder()
    recorder.start("zero", str(tmp_path), "stage")

    finalization = recorder.deactivate()

    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is True
    status = read_status_file(finalization.status_path)
    assert read_events(finalization.path) == []
    assert status["accepted_events"] == 0
    assert status["written_events"] == 0
    assert status["complete"] is True


def test_unflushed_instance_events_are_counted_as_lost(tmp_path: Path) -> None:
    recorder = get_recorder()
    buffer_a = RequestEventBuffer()
    buffer_b = RequestEventBuffer()
    path = recorder.start("deferred", str(tmp_path), "stage")
    buffer_a.capture(
        "event_a",
        (RequestEventSnapshot(request_id="request_a", metadata={"value": 1}),),
        {},
    )
    buffer_b.capture(
        "event_b",
        (RequestEventSnapshot(request_id="request_b", metadata={"value": 2}),),
        {},
    )
    buffer_a.flush(stage="stage_a")

    finalization = recorder.deactivate()

    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is True
    status = read_status_file(finalization.status_path)
    events = read_events(path)
    assert len(events) == 1
    assert events[0]["event_name"] == "event_a"
    assert events[0]["stage"] == "stage_a"
    assert status["accepted_events"] == 1
    assert status["written_events"] == 1
    assert status["dropped_events"] == 1
    assert status["complete"] is False
    assert "deferred_buffer_unflushed" in status["incomplete_reasons"]


def test_capture_paused_before_buffer_registration_loses_the_stop_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder()
    monkeypatch.setattr(event_recorder, "get_recorder", lambda: recorder)
    path = recorder.start("capture-cutoff", str(tmp_path), "stage")
    buffer = RequestEventBuffer()
    registration_entered = threading.Event()
    release_registration = threading.Event()

    class BlockingProducerLocal:
        def __init__(self) -> None:
            self.buffer = None

        def __getattribute__(self, name: str) -> object:
            if name == "buffer":
                registration_entered.set()
                if not release_registration.wait(timeout=5):
                    raise AssertionError("test did not release buffer registration")
                else:
                    pass
            else:
                pass
            return object.__getattribute__(self, name)

    recorder.producer_local = BlockingProducerLocal()
    consumed_snapshots: list[str] = []

    def snapshots() -> Iterator[RequestEventSnapshot]:
        consumed_snapshots.append("consumed")
        yield RequestEventSnapshot(request_id="request", metadata={"value": 1})

    producer = threading.Thread(
        target=buffer.capture,
        args=("event", snapshots(), {}),
    )
    producer.start()
    assert registration_entered.wait(timeout=5)

    finalization = recorder.deactivate()
    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is True
    release_registration.set()
    producer.join(timeout=5)
    assert producer.is_alive() is False

    status = read_status_file(finalization.status_path)
    assert consumed_snapshots == []
    assert read_events(path) == []
    assert status["accepted_events"] == 0
    assert status["complete"] is True


def test_status_published_success_is_separate_from_trace_completeness(
    tmp_path: Path,
) -> None:
    recorder = get_recorder()
    buffer = RequestEventBuffer()
    recorder.start("incomplete", str(tmp_path), "stage")
    buffer.capture(
        "event",
        (RequestEventSnapshot(request_id="pending", metadata={"value": 1}),),
        {},
    )

    finalization = recorder.deactivate()

    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is True
    status = read_status_file(finalization.status_path)
    assert status["complete"] is False
    assert recorder.wait_for_finalizations(timeout_seconds=5) is True


def test_status_publication_failure_is_local_to_its_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder()
    recorder.start("status-failure", str(tmp_path), "stage")
    fail_status_replace(monkeypatch)

    failed_finalization = recorder.deactivate()

    assert failed_finalization is not None
    assert failed_finalization.wait(timeout_seconds=5) is False
    assert recorder.wait_for_finalizations(timeout_seconds=5) is False
    assert list(tmp_path.glob("run_status_*.json")) == []

    monkeypatch.undo()
    next_directory = tmp_path / "next"
    recorder.start("status-next", str(next_directory), "stage")
    next_finalization = recorder.deactivate()

    assert next_finalization is not None
    assert next_finalization.wait(timeout_seconds=5) is True
    assert recorder.wait_for_finalizations(timeout_seconds=5) is True


def test_existing_output_fails_instead_of_creating_generation_suffix(
    tmp_path: Path,
) -> None:
    recorder = RequestEventRecorder()
    recorder.start("first", str(tmp_path), "stage")
    recorder.stop(wait=True)

    with pytest.raises(FileExistsError, match="fresh event_dir"):
        recorder.start("second", str(tmp_path), "stage")

    assert sorted(path.name for path in tmp_path.glob("events_*.jsonl")) == [
        f"events_stage_{recorder.pid}.jsonl"
    ]
