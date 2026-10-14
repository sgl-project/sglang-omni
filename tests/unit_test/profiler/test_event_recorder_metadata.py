# SPDX-License-Identifier: Apache-2.0
"""Metadata snapshot and storage failure tests for the request event recorder."""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.profiler.event_recorder import (
    RequestEventRecorder,
    get_recorder,
    json_default,
)
from tests.unit_test.profiler.event_recorder_test_utils import (
    read_events,
    read_status,
    stall_writer,
)


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

    out = json_default(FakeTensor())
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

    assert json_default(FakeScalar()) == 3.14


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

    events = read_events(p)
    assert len(events) == 1
    summary = events[0]["metadata"]["hidden_states"]
    assert summary["__tensor_summary__"] is True
    assert summary["shape"] == [1024, 4096]


def test_nested_metadata_is_snapshotted_before_writer_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(run_id="metadata", event_dir=str(tmp_path), stage="stage")
    writer_entered, release_writer = stall_writer(recorder, monkeypatch)
    nested_metadata = {"values": [1, {"name": "before"}]}
    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"nested": nested_metadata},
    )
    assert writer_entered.wait(timeout=5)
    try:
        nested_metadata["values"][0] = 2
        nested_metadata["values"][1]["name"] = "after"
    finally:
        release_writer.set()
    recorder.stop()

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
    recorder.stop()

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
    recorder.stop()

    status = read_status(path)
    assert status["metadata_truncations"] == 1
    assert status["complete"] is False
    assert "metadata_truncated" in status["incomplete_reasons"]


def test_metadata_snapshot_failure_marks_run_incomplete(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(run_id="serialize", event_dir=str(tmp_path), stage="stage")
    repr_called = False

    class BrokenMetadata:
        def __repr__(self) -> str:
            nonlocal repr_called
            repr_called = True
            raise RuntimeError("broken metadata")

    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"broken": BrokenMetadata()},
    )
    recorder.stop()

    assert repr_called is False
    status = read_status(path)
    assert status["metadata_snapshot_failures"] == 1
    assert status["complete"] is False
    assert "metadata_snapshot_failure" in status["incomplete_reasons"]


def test_metadata_string_is_bounded_and_marks_run_incomplete(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(
        run_id="metadata-string", event_dir=str(tmp_path), stage="stage"
    )
    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"large": "x" * 10000},
    )
    recorder.stop()

    event = read_events(path)[0]
    assert len(event["metadata"]["large"]["value"]) <= 4096
    status = read_status(path)
    assert status["metadata_truncations"] == 1
    assert status["complete"] is False


def test_bytes_metadata_is_rejected_without_repr(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start(
        run_id="metadata-bytes", event_dir=str(tmp_path), stage="stage"
    )
    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"payload": b"x" * 100000},
    )
    recorder.stop()

    assert read_events(path) == []
    status = read_status(path)
    assert status["metadata_snapshot_failures"] == 1
    assert status["complete"] is False


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
    recorder.stop()

    status = read_status(path)
    assert status["write_failures"] >= 1
    assert status["complete"] is False
    assert "event_file_write_failure" in status["incomplete_reasons"]


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
