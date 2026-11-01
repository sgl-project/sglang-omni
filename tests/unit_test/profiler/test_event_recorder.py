# SPDX-License-Identifier: Apache-2.0
"""Public API and co-located stage behavior for the request event recorder."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from sglang_omni.profiler import event_recorder
from sglang_omni.profiler.event_recorder import (
    RequestEvent,
    RequestEventBuffer,
    RequestEventRecorder,
    RequestEventSnapshot,
    emit,
    get_active_stage,
    get_recorder,
    reset_active_stage,
    set_active_stage,
)
from sglang_omni.profiler.views import reconstruct_timelines
from sglang_omni.utils.json import JsonValue
from tests.unit_test.profiler.event_recorder_test_utils import read_events, read_status


@pytest.fixture(autouse=True)
def reset_recorder() -> Iterator[None]:
    recorder = get_recorder()
    if recorder.is_active():
        recorder.stop(wait=True)
    else:
        pass
    recorder.wait_for_finalizations(timeout_seconds=5)
    reset_active_stage(None)
    yield
    if recorder.is_active():
        recorder.stop(wait=True)
    else:
        pass
    recorder.wait_for_finalizations(timeout_seconds=5)
    reset_active_stage(None)


def test_event_dataclass_roundtrip() -> None:
    request_event = RequestEvent(
        request_id="request-1",
        stage="thinker",
        event_name="first_token",
        timestamp_ns=1234,
        run_id="run-1",
        pid=99,
        metadata={"chunk_id": 7},
    )

    assert request_event.to_dict() == {
        "request_id": "request-1",
        "stage": "thinker",
        "event_name": "first_token",
        "timestamp_ns": 1234,
        "run_id": "run-1",
        "pid": 99,
        "metadata": {"chunk_id": 7},
    }


def test_inactive_recorder_is_silent() -> None:
    recorder = RequestEventRecorder()
    recorder.emit(request_id="request", stage="stage", event_name="event")

    assert recorder.is_active() is False


def test_start_writes_one_process_file_and_status(tmp_path: Path) -> None:
    recorder = RequestEventRecorder()
    path = recorder.start("run-1", str(tmp_path), "encoder")
    recorder.emit(
        request_id="request",
        stage="encoder",
        event_name="encoder_start",
        metadata={"items": 3},
    )
    finalization = recorder.deactivate()

    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is True
    assert Path(path).name == f"events_encoder_{recorder.pid}.jsonl"
    assert read_events(path)[0]["metadata"] == {"items": 3}
    status = read_status(path)
    assert status["complete"] is True
    assert status["written_events"] == 1


def test_module_emit_uses_process_singleton(tmp_path: Path) -> None:
    recorder = get_recorder()
    path = recorder.start("singleton", str(tmp_path), "coordinator")

    emit(request_id="request", stage=None, event_name="request_admission")
    recorder.stop(wait=True)

    assert [event["event_name"] for event in read_events(path)] == ["request_admission"]


def test_deferred_buffer_snapshots_metadata_once_before_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = get_recorder()
    path = recorder.start("deferred", str(tmp_path), "thinker")
    buffer = RequestEventBuffer()
    source_metadata: dict[str, JsonValue] = {"frames": [3]}
    cohort_metadata: dict[str, JsonValue] = {"sample_count": 12}
    original_snapshot = event_recorder.snapshot_metadata
    snapshot_calls = 0

    def count_snapshots(
        metadata: Mapping[str, object] | None,
    ) -> dict[str, JsonValue]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_snapshot(metadata)

    monkeypatch.setattr(event_recorder, "snapshot_metadata", count_snapshots)
    buffer.capture(
        "decode_committed",
        (RequestEventSnapshot(request_id="request", metadata=source_metadata),),
        cohort_metadata,
    )
    source_metadata["frames"][0] = 99
    cohort_metadata["sample_count"] = 0
    buffer.flush(stage="vocoder")
    recorder.stop(wait=True)

    events = read_events(path)
    assert snapshot_calls == 1
    assert events[0]["stage"] == "vocoder"
    assert events[0]["metadata"]["frames"] == [3]
    assert events[0]["metadata"]["sample_count"] == 12


def test_co_located_stages_share_file_and_release_independently(
    tmp_path: Path,
) -> None:
    recorder = RequestEventRecorder()
    path = recorder.start("shared", str(tmp_path), "preprocess")
    assert recorder.start("shared", str(tmp_path), "encoder") == path
    recorder.emit(
        request_id="request", stage="preprocess", event_name="preprocess_start"
    )

    assert recorder.release_stage("preprocess") is None
    assert recorder.is_active() is True
    recorder.emit(request_id="request", stage="encoder", event_name="encoder_start")
    finalization = recorder.release_stage("encoder")

    assert finalization is not None
    assert finalization.wait(timeout_seconds=5) is True
    assert {event["stage"] for event in read_events(path)} == {
        "preprocess",
        "encoder",
    }


def test_active_stage_flows_through_threads_and_asyncio_executor(
    tmp_path: Path,
) -> None:
    recorder = get_recorder()
    path = recorder.start("stages", str(tmp_path), "coordinator")
    recorder.start("stages", str(tmp_path), "encoder")

    def worker() -> None:
        token = set_active_stage("encoder")
        try:
            emit(request_id="thread", stage=None, event_name="thread_event")
        finally:
            reset_active_stage(token)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert thread.is_alive() is False

    async def emit_from_executor() -> None:
        token = set_active_stage("encoder")
        try:
            await asyncio.to_thread(
                emit,
                request_id="async",
                stage=None,
                event_name="async_event",
            )
        finally:
            reset_active_stage(token)

    asyncio.run(emit_from_executor())
    recorder.stop(wait=True)

    stages = {event["event_name"]: event["stage"] for event in read_events(path)}
    assert stages == {"thread_event": "encoder", "async_event": "encoder"}


def test_active_stage_reset_restores_outer_context() -> None:
    outer_token = set_active_stage("outer")
    try:
        inner_token = set_active_stage("inner")
        assert get_active_stage() == "inner"
        reset_active_stage(inner_token)
        assert get_active_stage() == "outer"
    finally:
        reset_active_stage(outer_token)

    assert get_active_stage() is None


def test_model_path_events_keep_wall_and_monotonic_clocks_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = get_recorder()
    path = recorder.start("model-path", str(tmp_path), "tts_engine")
    monkeypatch.setattr(event_recorder.time, "time_ns", lambda: 2_000_000_000)
    monkeypatch.setattr(event_recorder.time, "monotonic_ns", lambda: 123456)
    monkeypatch.setattr(event_recorder, "read_host_boot_id", lambda: "boot-test")

    event_recorder.emit_model_path_start("request")
    event_recorder.emit_model_path_end("request", status="success")
    recorder.stop(wait=True)

    events = read_events(path)
    assert [event["event_name"] for event in events] == [
        "model_path_start",
        "model_path_end",
    ]
    assert all(event["timestamp_ns"] == 2_000_000_000 for event in events)
    assert all(event["metadata"]["monotonic_ns"] == 123456 for event in events)
    assert all(event["metadata"]["host_boot_id"] == "boot-test" for event in events)
    assert reconstruct_timelines(path)["request"].total_ms == 0.0
