# SPDX-License-Identifier: Apache-2.0
"""Generation and finalization lifecycle tests for the request event recorder."""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.profiler.event_recorder import (
    RequestEventBuffer,
    RequestEventRecorder,
    RequestEventSnapshot,
    get_recorder,
)
from tests.unit_test.profiler.event_recorder_test_utils import (
    read_events,
    read_status_file,
)


def test_recording_generations_do_not_collide_in_one_directory(tmp_path: Path) -> None:
    recorder = RequestEventRecorder()
    first_path = recorder.start("first", str(tmp_path), "stage")
    recorder.emit(request_id="first", stage="stage", event_name="event")
    recorder.stop()

    second_path = recorder.start("second", str(tmp_path), "stage")
    recorder.emit(request_id="second", stage="stage", event_name="event")
    recorder.stop()

    assert first_path != second_path
    assert [event["run_id"] for event in read_events(first_path)] == ["first"]
    assert [event["run_id"] for event in read_events(second_path)] == ["second"]
    assert len(list(tmp_path.glob("events_*.jsonl"))) == 2
    assert len(list(tmp_path.glob("run_status_*.json"))) == 2


def test_reusing_run_id_creates_a_new_recording_generation(tmp_path: Path) -> None:
    recorder = RequestEventRecorder()
    first_path = recorder.start("reused", str(tmp_path), "stage")
    recorder.emit(request_id="first", stage="stage", event_name="event")
    recorder.stop()

    second_path = recorder.start("reused", str(tmp_path), "stage")
    recorder.emit(request_id="second", stage="stage", event_name="event")
    recorder.stop()

    assert first_path != second_path
    assert [event["request_id"] for event in read_events(first_path)] == ["first"]
    assert [event["request_id"] for event in read_events(second_path)] == ["second"]


def test_zero_event_retired_run_finalizes_complete(tmp_path: Path) -> None:
    recorder = RequestEventRecorder()
    path = recorder.start("zero", str(tmp_path), "stage")

    finalization = recorder.deactivate()

    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is True
    status = read_status_file(finalization.status_path)
    assert read_events(path) == []
    assert status["accepted_events"] == 0
    assert status["processed_events"] == 0
    assert status["written_events"] == 0
    assert status["unwritten_events"] == 0
    assert status["dropped_events"] == 0
    assert status["complete"] is True


def test_buffered_event_rejects_same_run_id_from_old_generation(tmp_path: Path) -> None:
    recorder = get_recorder()
    buffer = RequestEventBuffer()
    first_path = recorder.start("reused", str(tmp_path), "stage")
    buffer.capture(
        "event",
        (RequestEventSnapshot(request_id="stale", metadata={"value": 1}),),
        {},
    )
    recorder.stop()

    second_path = recorder.start("reused", str(tmp_path), "stage")
    buffer.flush(stage="stage")
    buffer.capture(
        "event",
        (RequestEventSnapshot(request_id="fresh", metadata={"value": 2}),),
        {},
    )
    buffer.flush(stage="stage")
    recorder.stop()

    assert read_events(first_path) == []
    assert [event["request_id"] for event in read_events(second_path)] == ["fresh"]


def test_stale_expected_run_id_cannot_enter_a_later_run(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    first_path = recorder.start(
        run_id="first", event_dir=str(tmp_path / "first"), stage="stage"
    )
    recorder.start(run_id="second", event_dir=str(tmp_path / "second"), stage="stage")
    recorder.emit(
        request_id="stale",
        stage="stage",
        event_name="event",
        expected_run_id="first",
    )
    second_path = recorder.active_path()
    recorder.stop()
    assert recorder.wait_for_finalizations(timeout_seconds=5)
    assert read_events(first_path) == []
    assert second_path is not None
    assert read_events(second_path) == []


def test_buffered_event_keeps_original_run_identity_after_rotation(
    tmp_path: Path,
) -> None:
    recorder = get_recorder()
    buffer = RequestEventBuffer()
    first_path = recorder.start(
        run_id="first", event_dir=str(tmp_path / "first"), stage="stage"
    )
    buffer.capture(
        "event",
        (RequestEventSnapshot(request_id="stale", metadata={"value": 1}),),
        {},
    )
    second_path = recorder.start(
        run_id="second", event_dir=str(tmp_path / "second"), stage="stage"
    )
    buffer.flush(stage="stage")
    recorder.stop()
    assert recorder.wait_for_finalizations(timeout_seconds=5)
    assert read_events(first_path) == []
    assert read_events(second_path) == []


def test_status_publication_failure_does_not_report_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    recorder.start(run_id="status", event_dir=str(tmp_path), stage="stage")

    from sglang_omni.profiler import event_recorder_writer

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("synthetic status publication failure")

    monkeypatch.setattr(event_recorder_writer.os, "replace", fail_replace)
    finalization = recorder.deactivate()
    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is False
    assert recorder.wait_for_finalizations(timeout_seconds=5) is False
    assert list(tmp_path.glob("run_status_*.json")) == []

    # A later successful run must not clear the recorder-level evidence that
    # this generation failed to publish its completion sidecar.
    monkeypatch.undo()
    recorder.start(run_id="status-next", event_dir=str(tmp_path), stage="stage")
    next_finalization = recorder.deactivate()
    assert next_finalization is not None
    assert next_finalization.wait(timeout_seconds=5) is True
    assert recorder.wait_for_finalizations(timeout_seconds=5) is False


def test_stop_returns_none_when_finalization_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder()
    recorder.start(run_id="status-stop", event_dir=str(tmp_path), stage="stage")

    from sglang_omni.profiler import event_recorder_writer

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("synthetic status publication failure")

    monkeypatch.setattr(event_recorder_writer.os, "replace", fail_replace)

    assert recorder.stop() is None
