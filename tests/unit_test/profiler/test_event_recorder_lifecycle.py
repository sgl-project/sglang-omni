# SPDX-License-Identifier: Apache-2.0
"""Run identity and finalization lifecycle tests for the request event recorder."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import pytest

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


def test_stale_expected_run_id_cannot_enter_a_later_run(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    first_path = recorder.start(
        run_id="first", event_dir=str(tmp_path / "first"), stage="stage"
    )
    recorder.stop(wait=True)
    recorder.start(run_id="second", event_dir=str(tmp_path / "second"), stage="stage")
    recorder.emit(
        request_id="stale",
        stage="stage",
        event_name="event",
        expected_run_id="first",
    )
    second_path = recorder.active_path()
    recorder.stop(wait=True)
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
    recorder.stop(wait=True)
    second_path = recorder.start(
        run_id="second", event_dir=str(tmp_path / "second"), stage="stage"
    )
    buffer.flush(stage="stage")
    recorder.stop(wait=True)
    assert recorder.wait_for_finalizations(timeout_seconds=5) is False
    assert read_events(first_path) == []
    assert read_events(second_path) == []


def test_delayed_buffer_flush_is_counted_as_incomplete(
    tmp_path: Path,
) -> None:
    recorder = get_recorder()
    buffer = RequestEventBuffer()
    path = recorder.start("delayed", str(tmp_path), "stage")
    buffer.capture(
        "event",
        (RequestEventSnapshot(request_id="pending", metadata={"value": 1}),),
        {},
    )

    finalization = recorder.deactivate()
    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is False
    status = read_status_file(finalization.status_path)
    assert status["accepted_events"] == 0
    assert status["dropped_events"] == 1
    assert status["complete"] is False
    assert "deferred_buffer_unflushed" in status["incomplete_reasons"]

    buffer.flush(stage="stage")
    assert read_events(path) == []


def test_buffer_capture_reserves_state_before_stop_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = get_recorder()
    buffer = RequestEventBuffer()
    path = recorder.start("capture-race", str(tmp_path), "stage")
    register_entered = threading.Event()
    release_register = threading.Event()
    original_register = recorder.register_deferred_buffer

    def delayed_register(run_id: str, finalizer: Callable[[], int]) -> bool:
        registered = original_register(run_id, finalizer)
        register_entered.set()
        if not release_register.wait(timeout=5):
            raise AssertionError("test did not release deferred-buffer registration")
        else:
            pass
        return registered

    monkeypatch.setattr(recorder, "register_deferred_buffer", delayed_register)
    capture_thread = threading.Thread(
        target=buffer.capture,
        args=(
            "event",
            (RequestEventSnapshot(request_id="pending", metadata={"value": 1}),),
            {},
        ),
    )
    capture_thread.start()
    assert register_entered.wait(timeout=5)

    finalization = recorder.deactivate()
    assert finalization is not None
    assert finalization.completion_event.is_set() is False
    release_register.set()
    capture_thread.join(timeout=5)
    assert capture_thread.is_alive() is False
    assert finalization.wait(timeout_seconds=5) is False

    status = read_status_file(finalization.status_path)
    assert read_events(path) == []
    assert status["accepted_events"] == 0
    assert status["dropped_events"] == 1
    assert status["complete"] is False
    assert "deferred_buffer_unflushed" in status["incomplete_reasons"]


def test_status_publication_failure_does_not_report_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    recorder.start(run_id="status", event_dir=str(tmp_path), stage="stage")

    fail_status_replace(monkeypatch)
    finalization = recorder.deactivate()
    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is False
    assert recorder.wait_for_finalizations(timeout_seconds=5) is False
    assert list(tmp_path.glob("run_status_*.json")) == []

    # A later successful run must not clear the recorder-level evidence that
    # the previous run failed to publish its completion sidecar.
    monkeypatch.undo()
    next_event_dir = tmp_path / "status-next"
    recorder.start(run_id="status-next", event_dir=str(next_event_dir), stage="stage")
    next_finalization = recorder.deactivate()
    assert next_finalization is not None
    assert next_finalization.wait(timeout_seconds=5) is True
    assert recorder.wait_for_finalizations(timeout_seconds=5) is False


def test_stop_returns_none_when_finalization_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder()
    recorder.start(run_id="status-stop", event_dir=str(tmp_path), stage="stage")

    fail_status_replace(monkeypatch)

    assert recorder.stop(wait=True) is None
