# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_omni.profiler.event_recorder."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
from tests.unit_test.profiler.event_recorder_test_utils import read_events


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
        rec.stop()

    events = read_events(path)
    assert len(events) == 2
    assert events[0]["event_name"] == "encoder_start"
    assert events[0]["run_id"] == "r0"
    assert events[0]["pid"] is not None
    assert events[1]["metadata"] == {"items": 3}
    assert sorted(tmp_path.glob("run_status_*.json")) == [
        tmp_path / f"run_status_{rec.pid}.json"
    ]


def test_default_stage_falls_back_to_active(tmp_path: Path) -> None:
    rec = get_recorder()
    rec.start(run_id="r0", event_dir=str(tmp_path), stage="thinker")
    try:
        rec.emit(request_id="r1", stage=None, event_name="scheduler_prefill_start")
    finally:
        path = rec.active_path()
        rec.stop()
    assert path is not None
    events = read_events(path)
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


def test_module_level_emit_uses_singleton(tmp_path: Path) -> None:
    rec = get_recorder()
    path = rec.start(run_id="r0", event_dir=str(tmp_path), stage="coord")
    emit(request_id="r1", stage=None, event_name="request_admission")
    rec.stop()
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

    recorder.stop()
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
    recorder.stop()

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
    recorder.stop()
    buffer.flush(stage="vocoder")

    second_path = recorder.start("second", str(tmp_path / "second"), "vocoder")
    buffer.capture(
        "decode_enqueued",
        (RequestEventSnapshot(request_id="new", metadata={"frames": 2}),),
        {},
    )
    buffer.flush(stage="vocoder")
    recorder.stop()

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
    rec.stop()

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

    rec.stop()

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
    rec.stop()

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
