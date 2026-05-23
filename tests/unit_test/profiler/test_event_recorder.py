# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_omni.profiler.event_recorder."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from sglang_omni.profiler.event_recorder import (
    RequestEvent,
    RequestEventRecorder,
    emit,
    get_recorder,
)


@pytest.fixture(autouse=True)
def _reset_recorder():
    """Make sure the process-global recorder is closed before every test."""
    rec = get_recorder()
    if rec.is_active():
        rec.stop()
    yield
    if rec.is_active():
        rec.stop()


def _read_events(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def test_event_dataclass_roundtrip() -> None:
    ev = RequestEvent(
        request_id="r1",
        stage="thinker",
        event_name="thinker_first_token",
        timestamp_ns=1234,
        run_id="run_x",
        pid=99,
        metadata={"chunk_id": 7},
    )
    d = ev.to_dict()
    assert d["request_id"] == "r1"
    assert d["stage"] == "thinker"
    assert d["event_name"] == "thinker_first_token"
    assert d["timestamp_ns"] == 1234
    assert d["run_id"] == "run_x"
    assert d["metadata"] == {"chunk_id": 7}


def test_inactive_recorder_is_silent(tmp_path: Path) -> None:
    """emit() must be a no-op when start() was never called."""
    rec = RequestEventRecorder()
    rec.emit(request_id="r1", stage="s", event_name="anything")
    assert rec.is_active() is False


def test_start_writes_jsonl_per_pid_stage(tmp_path: Path) -> None:
    rec = get_recorder()
    path = rec.start(run_id="r0", event_dir=str(tmp_path), stage="encoder")
    try:
        assert rec.is_active()
        assert Path(path).parent == tmp_path
        # Filename must encode stage + pid
        name = Path(path).name
        assert name.startswith("events_encoder_")
        assert name.endswith(".jsonl")

        rec.emit(request_id="r1", stage="encoder", event_name="encoder_start")
        rec.emit(
            request_id="r1",
            stage="encoder",
            event_name="encoder_end",
            metadata={"items": 3},
        )
    finally:
        rec.stop()

    events = _read_events(path)
    assert len(events) == 2
    assert events[0]["event_name"] == "encoder_start"
    assert events[0]["run_id"] == "r0"
    assert events[0]["pid"] is not None
    assert events[1]["metadata"] == {"items": 3}


def test_default_stage_falls_back_to_active(tmp_path: Path) -> None:
    rec = get_recorder()
    rec.start(run_id="r0", event_dir=str(tmp_path), stage="thinker")
    try:
        rec.emit(request_id="r1", stage=None, event_name="scheduler_prefill_start")
    finally:
        path = rec.active_path()
        rec.stop()
    assert path is not None
    events = _read_events(path)
    assert events[0]["stage"] == "thinker"


def test_stop_with_mismatched_run_id_is_noop(tmp_path: Path) -> None:
    rec = get_recorder()
    rec.start(run_id="r0", event_dir=str(tmp_path), stage="s")
    try:
        rec.stop(run_id="other")  # should NOT close
        assert rec.is_active()
    finally:
        rec.stop()
    assert rec.is_active() is False


def test_concurrent_emits_are_safe(tmp_path: Path) -> None:
    rec = get_recorder()
    path = rec.start(run_id="r0", event_dir=str(tmp_path), stage="thinker")
    n_threads = 8
    n_per_thread = 50

    def worker(tid: int) -> None:
        for i in range(n_per_thread):
            rec.emit(
                request_id=f"req-{tid}",
                stage="thinker",
                event_name="stage_dispatch",
                metadata={"i": i},
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rec.stop()

    events = _read_events(path)
    assert len(events) == n_threads * n_per_thread
    # Every line must be valid JSON with required fields
    for ev in events:
        assert ev["event_name"] == "stage_dispatch"
        assert ev["request_id"].startswith("req-")
        assert ev["run_id"] == "r0"


def test_module_level_emit_uses_singleton(tmp_path: Path) -> None:
    rec = get_recorder()
    path = rec.start(run_id="r0", event_dir=str(tmp_path), stage="coord")
    emit(request_id="r1", stage=None, event_name="request_admission")
    rec.stop()
    events = _read_events(path)
    assert any(e["event_name"] == "request_admission" for e in events)


def test_multi_stage_same_process_share_one_file(tmp_path: Path) -> None:
    """Stages sharing one process must write to ONE JSONL file.

    The previous rotating-per-stage behavior caused data routing bugs
    when declarative topology co-located multiple non-AR stages in one
    OS process. The first stage to call ``start()`` wins the filename;
    later stages join the same file and rely on each event's ``stage``
    field for identity.
    """
    rec = get_recorder()
    p1 = rec.start(run_id="r0", event_dir=str(tmp_path), stage="preprocessing")
    p2 = rec.start(run_id="r0", event_dir=str(tmp_path), stage="image_encoder")
    p3 = rec.start(run_id="r0", event_dir=str(tmp_path), stage="thinker")
    assert p1 == p2 == p3, "shared-process stages must reuse one file"
    assert Path(p1).name.startswith("events_preprocessing_")

    rec.emit(request_id="r1", stage="preprocessing", event_name="preprocess_start")
    rec.emit(request_id="r1", stage="image_encoder", event_name="encoder_start")
    rec.emit(request_id="r1", stage="thinker", event_name="stage_dispatch")
    rec.stop()

    events = _read_events(p1)
    stages = {e["stage"] for e in events}
    assert stages == {"preprocessing", "image_encoder", "thinker"}
