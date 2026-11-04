# SPDX-License-Identifier: Apache-2.0
"""Metadata bounds, tensor safety, and writer failure tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.profiler.event_recorder import RequestEventRecorder, json_default
from sglang_omni.profiler.event_recorder_types import RecorderRunState
from tests.unit_test.profiler.event_recorder_test_utils import read_events, read_status


def test_tensor_metadata_is_summarized_without_materializing(tmp_path: Path) -> None:
    class FakeTensor:
        shape = (32, 1024, 4096)
        dtype = "torch.float16"
        device = "cuda:0"

        def tolist(self) -> None:
            raise AssertionError("tensor contents must not be materialized")

    summary = json_default(FakeTensor())
    assert summary["__tensor_summary__"] is True
    assert summary["type"] == "FakeTensor"
    assert summary["shape"] == [32, 1024, 4096]

    class FakeHighRankTensor:
        shape = tuple(range(64))
        dtype = "float32"

    high_rank_summary = json_default(FakeHighRankTensor())
    assert len(high_rank_summary["shape"]) <= 31
    assert high_rank_summary["shape_truncated"] is True

    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start("tensor", str(tmp_path), "stage")
    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"tensor": FakeTensor()},
    )
    recorder.stop(wait=True)

    event = read_events(path)[0]
    assert event["metadata"]["tensor"]["shape"] == [32, 1024, 4096]


def test_cpu_scalar_is_copied_and_device_scalar_is_not_read() -> None:
    class CpuScalar:
        shape = ()
        dtype = "float32"

        def item(self) -> float:
            return 3.14

    class DeviceScalar:
        shape = ()
        dtype = "float32"
        device = "cuda:0"

        def item(self) -> float:
            raise AssertionError("device scalar was read on the producer")

    assert json_default(CpuScalar()) == 3.14
    assert json_default(DeviceScalar())["shape"] == []


def test_nested_metadata_is_owned_by_the_capture(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start("metadata-copy", str(tmp_path), "stage")
    nested_metadata = {"values": [1, {"name": "before"}]}
    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"nested": nested_metadata},
    )
    nested_metadata["values"][0] = 2
    nested_metadata["values"][1]["name"] = "after"
    recorder.stop(wait=True)

    assert read_events(path)[0]["metadata"] == {
        "nested": {"values": [1, {"name": "before"}]}
    }


def test_metadata_item_and_character_limits_mark_status_incomplete(
    tmp_path: Path,
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start("metadata-limit", str(tmp_path), "stage")
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

    long_path = tmp_path / "long"
    long_recorder = RequestEventRecorder(queue_capacity=8)
    long_event_path = long_recorder.start("long-string", str(long_path), "stage")
    long_recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"large": "x" * 10000},
    )
    long_recorder.stop(wait=True)

    event = read_events(long_event_path)[0]
    assert len(event["metadata"]["large"]["value"]) <= 512
    assert read_status(long_event_path)["complete"] is False

    integer_recorder = RequestEventRecorder(queue_capacity=8)
    integer_path = integer_recorder.start(
        "large-integer", str(tmp_path / "large-integer"), "stage"
    )
    integer_recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"large": 1 << 1_000_000},
    )
    integer_recorder.stop(wait=True)

    integer_event = read_events(integer_path)[0]
    assert integer_event["metadata"]["large"]["reason"] == "integer_limit"
    assert read_status(integer_path)["metadata_truncations"] == 1


def test_unsupported_metadata_is_dropped_without_repr(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start("bad-metadata", str(tmp_path), "stage")
    repr_called = False

    class UnsupportedMetadata:
        def __repr__(self) -> str:
            nonlocal repr_called
            repr_called = True
            raise RuntimeError("metadata repr must not run")

    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"broken": UnsupportedMetadata()},
    )
    recorder.stop(wait=True)

    assert repr_called is False
    assert read_events(path) == []
    status = read_status(path)
    assert status["metadata_snapshot_failures"] == 1
    assert status["dropped_events"] == 1
    assert status["complete"] is False


def test_bytes_metadata_is_rejected_without_copying(tmp_path: Path) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start("bytes", str(tmp_path), "stage")
    recorder.emit(
        request_id="request",
        stage="stage",
        event_name="event",
        metadata={"payload": b"x" * 100000},
    )
    recorder.stop(wait=True)

    assert read_events(path) == []
    status = read_status(path)
    assert status["metadata_snapshot_failures"] == 1
    assert status["complete"] is False


def test_file_write_failure_marks_status_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = RequestEventRecorder(queue_capacity=8)
    path = recorder.start("write-failure", str(tmp_path), "stage")

    class BrokenFile:
        closed = False

        def write(self, serialized_event: str) -> None:
            del serialized_event
            raise OSError("synthetic event write failure")

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    def open_broken_file(run_state: RecorderRunState) -> BrokenFile:
        del run_state
        return BrokenFile()

    monkeypatch.setattr(recorder, "open_event_file", open_broken_file)
    recorder.emit(request_id="request", stage="stage", event_name="event")
    recorder.stop(wait=True)

    status = read_status(path)
    assert status["write_failures"] >= 1
    assert status["complete"] is False
    assert "event_file_write_failure" in status["incomplete_reasons"]
