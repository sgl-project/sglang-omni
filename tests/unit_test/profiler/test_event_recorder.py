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
    _json_default,
    emit,
    get_recorder,
    reset_active_stage,
    set_active_stage,
)


@pytest.fixture(autouse=True)
def _reset_recorder():
    """Make sure the process-global recorder is closed before every test."""
    rec = get_recorder()
    if rec.is_active():
        rec.stop()
    # Clear any active-stage binding leaked from prior tests.
    reset_active_stage(None)
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
    def _worker(stage_name: str) -> None:
        set_active_stage(stage_name)
        try:
            emit(request_id="r1", stage=None, event_name=f"from_{stage_name}")
        finally:
            reset_active_stage(None)

    t_thinker = threading.Thread(target=_worker, args=("thinker",))
    t_decode = threading.Thread(target=_worker, args=("decode",))
    t_thinker.start()
    t_decode.start()
    t_thinker.join()
    t_decode.join()

    rec.stop()

    events = _read_events(p)
    by_event = {e["event_name"]: e["stage"] for e in events}
    assert (
        by_event["from_main"] == "preprocessing"
    ), "main thread had no active stage; should fall back to recorder._stage"
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

    def _compute() -> None:
        # Simulate what SimpleScheduler's worker does after stage binding.
        emit(request_id="r1", stage=None, event_name="from_executor")

    async def _run() -> None:
        set_active_stage("encoder")
        try:
            await asyncio.to_thread(_compute)
        finally:
            reset_active_stage(None)

    asyncio.run(_run())
    rec.stop()

    events = _read_events(p)
    by_event = {e["event_name"]: e["stage"] for e in events}
    seen.update(by_event)
    assert (
        seen["from_executor"] == "encoder"
    ), "contextvar didn't propagate through asyncio.to_thread"


# ---------------------------------------------------------------------------
# Safe metadata serialization (review finding P2)
# ---------------------------------------------------------------------------


def test_json_default_summarizes_tensor_without_materializing() -> None:
    """``_json_default`` must NOT call .tolist() on tensors / arrays.

    Regression for review finding P2: previously any object with .tolist()
    got fully expanded; on a GPU tensor this synchronizes, copies to CPU,
    and serializes potentially MBs of data inline — violating the "large
    blobs stay out of metadata" rule.
    """

    class FakeTensor:
        shape = (32, 1024, 4096)
        dtype = "torch.float16"
        device = "cuda:0"

        def tolist(self):  # pragma: no cover - must not be called
            raise AssertionError(
                "_json_default called tolist() on a tensor — this is the "
                "exact bug we're regressing against"
            )

        def item(self):  # pragma: no cover
            raise AssertionError("item() called on multi-dim tensor")

    out = _json_default(FakeTensor())
    assert isinstance(out, dict)
    assert out["__tensor_summary__"] is True
    assert out["type"] == "FakeTensor"
    assert out["shape"] == [32, 1024, 4096]
    assert out["dtype"] == "torch.float16"
    assert out["device"] == "cuda:0"


def test_json_default_unwraps_zero_d_tensor_as_scalar() -> None:
    """0-D tensors / numpy scalars should serialize as plain scalars."""

    class FakeScalar:
        shape = ()
        dtype = "float32"

        def item(self):
            return 3.14

    assert _json_default(FakeScalar()) == 3.14


def test_emit_with_tensor_metadata_does_not_materialize(tmp_path: Path) -> None:
    """End-to-end: even if a caller hands us a tensor in metadata, the
    JSONL line must stay small and never call .tolist()."""

    rec = get_recorder()
    p = rec.start(run_id="r0", event_dir=str(tmp_path), stage="encoder")

    class HugeTensor:
        shape = (1024, 4096)
        dtype = "torch.bfloat16"
        device = "cuda:0"

        def tolist(self):  # pragma: no cover
            raise AssertionError("hot-path emit called tolist()")

    rec.emit(
        request_id="r1",
        stage="encoder",
        event_name="encoder_end",
        metadata={"hidden_states": HugeTensor()},
    )
    rec.stop()

    events = _read_events(p)
    assert len(events) == 1
    summary = events[0]["metadata"]["hidden_states"]
    assert summary["__tensor_summary__"] is True
    assert summary["shape"] == [1024, 4096]


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
    assert event_recorder._active_stage_cv.get() == "leaks"
    assert getattr(event_recorder._thread_active_stage, "stage", None) == "leaks"

    reset_active_stage(None)

    # Both must now be empty.
    assert (
        event_recorder._active_stage_cv.get() is None
    ), "contextvar still bound after reset_active_stage(None)"
    assert getattr(event_recorder._thread_active_stage, "stage", None) is None
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
