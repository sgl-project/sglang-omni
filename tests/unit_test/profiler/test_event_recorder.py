# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_omni.profiler.event_recorder."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from io import TextIOWrapper
from pathlib import Path
from types import FrameType

import pytest

from sglang_omni.profiler import event_recorder
from sglang_omni.profiler.event_recorder import (
    RequestEvent,
    RequestEventBuffer,
    RequestEventRecorder,
    RequestEventSnapshot,
    emit,
    get_recorder,
    reset_active_stage,
    set_active_stage,
)
from sglang_omni.profiler.event_recorder_types import QueuedRequestEvent
from sglang_omni.profiler.views import reconstruct_timelines


@pytest.fixture(autouse=True)
def reset_recorder():
    """Make sure the process-global recorder is closed before every test."""
    rec = get_recorder()
    if rec.is_active():
        rec.stop(wait=True)
    # Clear any active-stage binding leaked from prior tests.
    reset_active_stage(None)
    yield
    if rec.is_active():
        rec.stop(wait=True)


def read_events(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def read_status(path: str) -> dict:
    status_paths = sorted(Path(path).parent.glob("run_status_*.json"))
    assert len(status_paths) == 1
    return json.loads(status_paths[0].read_text(encoding="utf-8"))


def read_status_file(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stall_writer(
    recorder: RequestEventRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[threading.Event, threading.Event]:
    writer_entered = threading.Event()
    release_writer = threading.Event()
    original_write = recorder.write_queued_event

    def stalled_write(
        queued_event: QueuedRequestEvent,
        file_handle: TextIOWrapper | None,
    ) -> TextIOWrapper | None:
        writer_entered.set()
        if not release_writer.wait(timeout=5):
            raise AssertionError("test did not release the stalled writer")
        else:
            pass
        return original_write(queued_event, file_handle)

    monkeypatch.setattr(recorder, "write_queued_event", stalled_write)
    return writer_entered, release_writer


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
        assert name == f"events_encoder_{rec.pid}.jsonl"

        rec.emit(request_id="r1", stage="encoder", event_name="encoder_start")
        rec.emit(
            request_id="r1",
            stage="encoder",
            event_name="encoder_end",
            metadata={"items": 3},
        )
    finally:
        rec.stop(wait=True)

    events = read_events(path)
    assert len(events) == 2
    assert events[0]["event_name"] == "encoder_start"
    assert events[0]["run_id"] == "r0"
    assert events[0]["pid"] is not None
    assert events[1]["metadata"] == {"items": 3}
    assert sorted(tmp_path.glob("run_status_*.json")) == [
        tmp_path / f"run_status_{rec.pid}.json"
    ]


def test_stop_with_mismatched_run_id_is_noop(tmp_path: Path) -> None:
    rec = get_recorder()
    rec.start(run_id="r0", event_dir=str(tmp_path), stage="s")
    try:
        rec.stop(run_id="other")  # should NOT close
        assert rec.is_active()
    finally:
        rec.stop(wait=True)
    assert rec.is_active() is False


def test_module_level_emit_uses_singleton(tmp_path: Path) -> None:
    rec = get_recorder()
    path = rec.start(run_id="r0", event_dir=str(tmp_path), stage="coord")
    emit(request_id="r1", stage=None, event_name="request_admission")
    rec.stop(wait=True)
    events = read_events(path)
    assert any(e["event_name"] == "request_admission" for e in events)


def test_buffer_flush_preserves_thread_ownership_and_capture_snapshot(
    tmp_path: Path,
) -> None:
    recorder = get_recorder()
    path = recorder.start(run_id="buffered", event_dir=str(tmp_path), stage="thinker")
    buffer = RequestEventBuffer()
    captured = threading.Event()
    release = threading.Event()

    def worker() -> tuple[int, int]:
        metadata: dict[str, int | float | str] = {"frames": 3}
        before_capture_ns = time.time_ns()
        buffer.capture(
            "decode_committed",
            (
                RequestEventSnapshot(request_id=request_id, metadata=metadata)
                for request_id in ("worker-a", "worker-b")
            ),
            {"samples": 12},
        )
        after_capture_ns = time.time_ns()
        metadata["frames"] = 7
        captured.set()
        assert release.wait(timeout=5)
        buffer.flush(stage="vocoder")
        buffer.flush(stage="vocoder")
        return before_capture_ns, after_capture_ns

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="decode") as executor:
        completion = executor.submit(worker)
        try:
            assert captured.wait(timeout=5)
            buffer.capture(
                "decode_enqueued",
                (RequestEventSnapshot(request_id="ingest", metadata={"frames": 1}),),
                {},
            )
            buffer.flush(stage="vocoder")
        finally:
            release.set()
        before_capture_ns, after_capture_ns = completion.result(timeout=5)

    recorder.stop(wait=True)
    events = read_events(path)
    assert [event["request_id"] for event in events] == [
        "ingest",
        "worker-a",
        "worker-b",
    ]
    assert before_capture_ns <= events[1]["timestamp_ns"] <= after_capture_ns
    assert events[1]["timestamp_ns"] == events[2]["timestamp_ns"]
    assert events[1]["timestamp_ns"] <= events[0]["timestamp_ns"]
    assert events[1]["metadata"]["monotonic_s"] == events[2]["metadata"]["monotonic_s"]
    assert [event["metadata"]["frames"] for event in events] == [1, 3, 3]
    assert [event["metadata"]["samples"] for event in events[1:]] == [12, 12]
    assert events[0]["metadata"]["worker"] == threading.current_thread().name
    assert all(event["metadata"]["worker"].startswith("decode") for event in events[1:])
    assert {event["stage"] for event in events} == {"vocoder"}
    assert {event["run_id"] for event in events} == {"buffered"}


def test_buffer_checks_activity_before_consuming_snapshots(tmp_path: Path) -> None:
    recorder = get_recorder()
    buffer = RequestEventBuffer()
    visited: list[str] = []

    def snapshots(request_id: str) -> Iterator[RequestEventSnapshot]:
        visited.append(request_id)
        yield RequestEventSnapshot(request_id=request_id, metadata={"frames": 1})

    buffer.capture("decode_enqueued", snapshots("inactive-capture"), {})
    buffer.flush(stage="vocoder")
    assert visited == []

    path = recorder.start(run_id="buffered", event_dir=str(tmp_path), stage="thinker")
    buffer.capture("decode_enqueued", snapshots("active"), {})
    assert visited == ["active"]
    buffer.flush(stage="vocoder")
    recorder.stop(wait=True)

    buffer.capture("decode_enqueued", snapshots("stopped"), {})
    assert visited == ["active"]
    events = read_events(path)
    assert [event["request_id"] for event in events] == ["active"]


def test_buffer_flush_after_stop_does_not_leak_into_next_run(tmp_path: Path) -> None:
    recorder = get_recorder()
    buffer = RequestEventBuffer()
    first_path = recorder.start("first", str(tmp_path / "first"), "vocoder")
    buffer.capture(
        "decode_enqueued",
        (RequestEventSnapshot(request_id="old", metadata={"frames": 1}),),
        {},
    )
    recorder.stop(wait=True)
    buffer.flush(stage="vocoder")

    second_path = recorder.start("second", str(tmp_path / "second"), "vocoder")
    buffer.capture(
        "decode_enqueued",
        (RequestEventSnapshot(request_id="new", metadata={"frames": 2}),),
        {},
    )
    buffer.flush(stage="vocoder")
    recorder.stop(wait=True)

    assert read_events(first_path) == []
    events = read_events(second_path)
    assert [event["request_id"] for event in events] == ["new"]
    assert events[0]["run_id"] == "second"
    assert events[0]["metadata"]["frames"] == 2


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
    rec.stop(wait=True)

    events = read_events(p1)
    stages = {e["stage"] for e in events}
    assert stages == {"preprocessing", "image_encoder", "thinker"}


def test_releasing_one_shared_stage_does_not_stop_live_stage(tmp_path: Path) -> None:
    recorder = RequestEventRecorder()
    path = recorder.start("shared", str(tmp_path), "first")
    recorder.start("shared", str(tmp_path), "second")
    recorder.emit(request_id="first", stage="first", event_name="event")

    assert recorder.release_stage("first") is None
    assert recorder.is_active() is True
    recorder.emit(request_id="second", stage="second", event_name="event")
    finalization = recorder.release_stage("second")
    assert finalization is not None
    assert finalization.wait(timeout_seconds=5)

    assert {event["stage"] for event in read_events(path)} == {"first", "second"}


# ---------------------------------------------------------------------------
# Active-stage attribution (review finding P1)
# ---------------------------------------------------------------------------


def test_emit_stage_none_uses_thread_local_active_stage(tmp_path: Path) -> None:
    """``emit(stage=None)`` must pick up the per-thread active stage, not
    the process-global ``_stage`` (which is the first stage to call start()
    in a shared-process topology).

    Regression for review finding P1: previously preprocessor / encoder /
    OmniScheduler / code2wav all called ``emit(stage=None)`` and got
    attributed to the first stage in the process.
    """
    rec = get_recorder()
    p = rec.start(run_id="r0", event_dir=str(tmp_path), stage="preprocessing")
    rec.start(run_id="r0", event_dir=str(tmp_path), stage="thinker")
    rec.start(run_id="r0", event_dir=str(tmp_path), stage="decode")

    # Main thread: no active stage → falls back to recorder's _stage
    # ("preprocessing", first to call start()).
    emit(request_id="r1", stage=None, event_name="from_main")

    # Worker threads: each binds its own active stage, then emits.
    def worker(stage_name: str) -> None:
        set_active_stage(stage_name)
        try:
            emit(request_id="r1", stage=None, event_name=f"from_{stage_name}")
        finally:
            reset_active_stage(None)

    t_thinker = threading.Thread(target=worker, args=("thinker",))
    t_decode = threading.Thread(target=worker, args=("decode",))
    t_thinker.start()
    t_decode.start()
    t_thinker.join()
    t_decode.join()

    rec.stop(wait=True)

    events = read_events(p)
    by_event = {e["event_name"]: e["stage"] for e in events}
    assert (
        by_event["from_main"] == "preprocessing"
    ), "main thread had no active stage; should fall back to recorder.stage"
    assert (
        by_event["from_thinker"] == "thinker"
    ), "worker thread's set_active_stage('thinker') was ignored"
    assert (
        by_event["from_decode"] == "decode"
    ), "worker thread's set_active_stage('decode') was ignored"


def test_emit_stage_none_uses_contextvar_in_asyncio_executor(
    tmp_path: Path,
) -> None:
    """The contextvar must propagate through asyncio.to_thread / executor.

    SimpleScheduler in concurrent mode runs each compute_fn via
    ``asyncio.to_thread``, which copies the context but NOT the parent
    thread's threading.local. The contextvar variant is what makes
    stage attribution work in that path.
    """
    import asyncio

    rec = get_recorder()
    p = rec.start(run_id="r0", event_dir=str(tmp_path), stage="first")
    rec.start(run_id="r0", event_dir=str(tmp_path), stage="encoder")

    seen: dict[str, str | None] = {}

    def compute() -> None:
        # Simulate what SimpleScheduler's worker does after stage binding.
        emit(request_id="r1", stage=None, event_name="from_executor")

    async def run() -> None:
        set_active_stage("encoder")
        try:
            await asyncio.to_thread(compute)
        finally:
            reset_active_stage(None)

    asyncio.run(run())
    rec.stop(wait=True)

    events = read_events(p)
    by_event = {e["event_name"]: e["stage"] for e in events}
    seen.update(by_event)
    assert (
        seen["from_executor"] == "encoder"
    ), "contextvar didn't propagate through asyncio.to_thread"


# ---------------------------------------------------------------------------
# Active-stage lifecycle (review finding P3)
# ---------------------------------------------------------------------------


def test_reset_active_stage_without_token_clears_both_thread_local_and_contextvar() -> (
    None
):
    """``reset_active_stage(None)`` must clear the contextvar too.

    Regression for review finding P3: the previous implementation only
    cleared the ``threading.local`` slot. The contextvar value persisted,
    so anyone using the helper to scrub state (test fixtures, ad-hoc
    cleanup) would silently leak the active stage into the next caller
    in the same context.
    """
    from sglang_omni.profiler import event_recorder

    set_active_stage("leaks")
    # Sanity: both backends are bound.
    assert (
        event_recorder._active_stage_cv.get() == "leaks"
    )  # noqa: leading-underscore  # production name
    assert (
        getattr(event_recorder._thread_active_stage, "stage", None) == "leaks"
    )  # noqa: leading-underscore  # production name

    reset_active_stage(None)

    # Both must now be empty.
    assert (
        event_recorder._active_stage_cv.get()
        is None  # noqa: leading-underscore  # production name
    ), "contextvar still bound after reset_active_stage(None)"
    assert (
        getattr(event_recorder._thread_active_stage, "stage", None) is None
    )  # noqa: leading-underscore  # production name
    # And the public accessor agrees.
    from sglang_omni.profiler.event_recorder import get_active_stage

    assert get_active_stage() is None


def test_reset_active_stage_with_token_restores_previous_value() -> None:
    """The ``Token`` form must still work — the standard ContextVar contract."""
    from sglang_omni.profiler.event_recorder import get_active_stage

    outer_token = set_active_stage("outer")
    try:
        inner_token = set_active_stage("inner")
        assert get_active_stage() == "inner"
        reset_active_stage(inner_token)
        # Reset restores the contextvar to "outer"; thread-local was
        # blanked, but the contextvar takes precedence in get_active_stage.
        assert get_active_stage() == "outer"
    finally:
        reset_active_stage(outer_token)
        # After the outer reset, contextvar is back to its initial None.
        assert get_active_stage() is None
def test_scheduler_model_path_activity_is_monotonic_and_default_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorder = event_recorder.get_recorder()
    recorder.stop(wait=True)
    event_recorder.emit_model_path_start("request-off")
    assert recorder.is_active() is False

    path = recorder.start("run-unit", str(tmp_path), "tts_engine")
    monkeypatch.setattr(event_recorder.time, "monotonic_ns", lambda: 123456)
    monkeypatch.setattr(event_recorder, "read_host_boot_id", lambda: "boot-test")
    event_recorder.emit_model_path_start("request-on")
    event_recorder.emit_model_path_end("request-on", status="success")
    recorder.stop(wait=True)
    emitted = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]
    assert [item["event_name"] for item in emitted] == [
        "model_path_start",
        "model_path_end",
    ]
    assert all(item["metadata"]["monotonic_ns"] == 123456 for item in emitted)
    assert all(item["metadata"]["clock"] == "CLOCK_MONOTONIC" for item in emitted)
    assert all(item["metadata"]["host_boot_id"] == "boot-test" for item in emitted)
    assert emitted[-1]["metadata"]["status"] == "success"


def test_model_path_activity_preserves_the_profiler_wall_clock_domain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorder = event_recorder.get_recorder()
    recorder.stop(wait=True)
    path = recorder.start("run-clock-domain", str(tmp_path), "tts_engine")
    monkeypatch.setattr(event_recorder.time, "time_ns", lambda: 2_000_000_000)
    monkeypatch.setattr(event_recorder.time, "monotonic_ns", lambda: 123456)
    event_recorder.emit(
        request_id="request-clock-domain",
        stage="tts_engine",
        event_name="request_started",
    )
    event_recorder.emit_model_path_start("request-clock-domain")
    event_recorder.emit_model_path_end(
        "request-clock-domain",
        status="success",
    )
    event_recorder.emit(
        request_id="request-clock-domain",
        stage="tts_engine",
        event_name="request_finished",
    )
    recorder.stop(wait=True)

    emitted = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]
    model_path_events = [
        item for item in emitted if item["event_name"].startswith("model_path_")
    ]
    assert all(item["timestamp_ns"] == 2_000_000_000 for item in emitted)
    assert all(item["metadata"]["monotonic_ns"] == 123456 for item in model_path_events)
    assert reconstruct_timelines(path)["request-clock-domain"].total_ms == 0.0

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


def test_full_queue_drops_events_and_marks_status_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    assert status["written_events"] == 64
    assert status["complete"] is True


def test_run_rotation_does_not_mix_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    recorder = RequestEventRecorder(queue_capacity=8)
    first_path = recorder.start("first", str(tmp_path), "stage")
    first_finalization = recorder.deactivate()
    assert first_finalization is not None
    assert first_finalization.wait(timeout_seconds=5)
    second_path = recorder.start("second", str(tmp_path), "stage")
    recorder.emit(request_id="new", stage="stage", event_name="event")
    second_finalization = recorder.deactivate()
    assert second_finalization is not None
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
    assert "writer_callback_failure" in old_status["incomplete_reasons"]
    assert read_status(old_path)["unwritten_events"] >= 1

    monkeypatch.undo()
    new_path = recorder.start("new", str(tmp_path / "new"), "stage")
    recorder.emit(request_id="new", stage="stage", event_name="event")
    new_finalization = recorder.deactivate()
    assert new_finalization is not None
    assert new_finalization.wait(timeout_seconds=5) is True
    assert read_status(new_path)["complete"] is True


def test_recording_generations_do_not_collide_in_one_directory(tmp_path: Path) -> None:
    recorder = RequestEventRecorder()
    first_path = recorder.start("first", str(tmp_path), "stage")
    recorder.emit(request_id="first", stage="stage", event_name="event")
    recorder.stop(wait=True)

    second_path = recorder.start("second", str(tmp_path), "stage")
    recorder.emit(request_id="second", stage="stage", event_name="event")
    recorder.stop(wait=True)

    assert first_path != second_path
    assert [event["run_id"] for event in read_events(first_path)] == ["first"]
    assert [event["run_id"] for event in read_events(second_path)] == ["second"]
    assert len(list(tmp_path.glob("events_*.jsonl"))) == 2
    assert len(list(tmp_path.glob("run_status_*.json"))) == 2


def test_reusing_run_id_creates_a_new_recording_generation(tmp_path: Path) -> None:
    recorder = RequestEventRecorder()
    first_path = recorder.start("reused", str(tmp_path), "stage")
    recorder.emit(request_id="first", stage="stage", event_name="event")
    recorder.stop(wait=True)

    second_path = recorder.start("reused", str(tmp_path), "stage")
    recorder.emit(request_id="second", stage="stage", event_name="event")
    recorder.stop(wait=True)

    assert first_path != second_path
    assert [event["request_id"] for event in read_events(first_path)] == ["first"]
    assert [event["request_id"] for event in read_events(second_path)] == ["second"]


def test_buffered_event_rejects_same_run_id_from_old_generation(tmp_path: Path) -> None:
    recorder = get_recorder()
    buffer = RequestEventBuffer()
    first_path = recorder.start("reused", str(tmp_path), "stage")
    buffer.capture(
        "event",
        (RequestEventSnapshot(request_id="stale", metadata={"value": 1}),),
        {},
    )
    recorder.stop(wait=True)

    second_path = recorder.start("reused", str(tmp_path), "stage")
    buffer.flush(stage="stage")
    buffer.capture(
        "event",
        (RequestEventSnapshot(request_id="fresh", metadata={"value": 2}),),
        {},
    )
    buffer.flush(stage="stage")
    recorder.stop(wait=True)

    assert read_events(first_path) == []
    assert [event["request_id"] for event in read_events(second_path)] == ["fresh"]


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


def test_nested_metadata_is_snapshotted_before_writer_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(run_id="metadata", event_dir=str(tmp_path), stage="stage")
    nested_metadata = {"values": [1, {"name": "before"}]}
    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"nested": nested_metadata},
    )
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)
    finalization = recorder.deactivate()
    assert finalization is not None
    assert writer_entered.wait(timeout=5)
    try:
        nested_metadata["values"][0] = 2
        nested_metadata["values"][1]["name"] = "after"
    finally:
        release_writer.set()
    assert finalization.wait(timeout_seconds=5)

    event = read_events(path)[0]
    assert event["metadata"] == {"nested": {"values": [1, {"name": "before"}]}}


def test_tensor_like_metadata_is_summarized_without_materialization(
    tmp_path: Path,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(run_id="tensor", event_dir=str(tmp_path), stage="stage")

    class FakeTensor:
        shape = (4, 8)
        dtype = "torch.float16"
        device = "cuda:0"

        def tolist(self):
            raise AssertionError("tensor metadata was materialized")

    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"nested": {"tensor": FakeTensor()}},
    )
    recorder.stop(wait=True)

    summary = read_events(path)[0]["metadata"]["nested"]["tensor"]
    assert summary["__tensor_summary__"] is True
    assert summary["shape"] == [4, 8]


def test_metadata_truncation_is_reported_in_status(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(
        run_id="metadata-limit", event_dir=str(tmp_path), stage="stage"
    )
    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={f"field-{index}": index for index in range(300)},
    )
    recorder.stop(wait=True)

    status = read_status(path)
    assert status["metadata_truncations"] == 1
    assert status["complete"] is False
    assert "metadata_truncated" in status["incomplete_reasons"]


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
    assert status["written_events"] == len(events)


def test_metadata_snapshot_failure_marks_run_incomplete(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(run_id="serialize", event_dir=str(tmp_path), stage="stage")

    class BrokenMetadata:
        def __repr__(self):
            raise RuntimeError("broken metadata")

    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"broken": BrokenMetadata()},
    )
    recorder.stop(wait=True)

    status = read_status(path)
    assert status["metadata_snapshot_failures"] == 1
    assert status["complete"] is False
    assert "metadata_snapshot_failure" in status["incomplete_reasons"]


def test_file_write_failure_marks_run_incomplete(tmp_path: Path, monkeypatch) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(run_id="write", event_dir=str(tmp_path), stage="stage")

    class BrokenFile:
        def write(self, serialized_event: str) -> None:
            raise OSError("broken file")

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    def open_broken_file(run_state):
        return BrokenFile()

    monkeypatch.setattr(recorder, "open_event_file", open_broken_file)
    recorder.emit(request_id="request", stage="stage", event_name="event")
    recorder.stop(wait=True)

    status = read_status(path)
    assert status["write_failures"] >= 1
    assert status["complete"] is False
    assert "event_file_write_failure" in status["incomplete_reasons"]


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


def test_rotation_backlog_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    first_path = recorder.start(
        run_id="run-0", event_dir=str(tmp_path / "run-0"), stage="stage"
    )
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)
    recorder.emit(request_id="request", stage="stage", event_name="event")
    finalization = recorder.deactivate()
    assert finalization is not None
    assert writer_entered.wait(timeout=5)
    with pytest.raises(RuntimeError, match="draining"):
        recorder.start(
            run_id="run-1",
            event_dir=str(tmp_path / "run-1"),
            stage="stage",
        )
    try:
        release_writer.set()
        assert finalization.wait(timeout_seconds=5)
    finally:
        release_writer.set()
    second_path = recorder.start("run-1", str(tmp_path / "run-1"), "stage")
    recorder.stop(wait=True)
    assert read_events(first_path)[0]["request_id"] == "request"
    assert read_events(second_path) == []


def test_disabled_recorder_does_not_snapshot_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    recorder = RequestEventRecorder()

    def fail_snapshot(metadata):
        raise AssertionError("disabled recorder constructed event metadata")

    from sglang_omni.profiler import (
        event_recorder_writer as event_recorder_writer_module,
    )

    monkeypatch.setattr(
        event_recorder_writer_module,
        "snapshot_metadata",
        fail_snapshot,
    )
    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"value": 1},
    )
    assert recorder.writer_thread is None


def test_nonblocking_stop_returns_while_writer_is_stalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    recorder.start(run_id="async-stop", event_dir=str(tmp_path), stage="stage")
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)
    recorder.emit(request_id="request", stage="stage", event_name="event")
    finalization = recorder.deactivate()
    assert finalization is not None
    try:
        assert writer_entered.wait(timeout=5)
        assert finalization.completion_event.is_set() is False
    finally:
        release_writer.set()
    assert finalization.wait(timeout_seconds=5)
