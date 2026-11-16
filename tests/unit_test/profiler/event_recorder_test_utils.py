# SPDX-License-Identifier: Apache-2.0
"""Small helpers shared by request event recorder tests."""

from __future__ import annotations

import json
import threading
from io import TextIOWrapper
from pathlib import Path

import pytest

from sglang_omni.profiler import event_recorder_writer
from sglang_omni.profiler.event_recorder_types import RecorderRunState, RequestEvent
from sglang_omni.profiler.event_recorder_writer import RequestEventRecorder
from sglang_omni.utils.json import JsonValue


def read_events(path: str | Path) -> list[dict[str, JsonValue]]:
    with open(path, "r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def read_status_file(path: str | Path) -> dict[str, JsonValue]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_status(path: str | Path) -> dict[str, JsonValue]:
    status_paths = sorted(Path(path).parent.glob("run_status_*.json"))
    assert len(status_paths) == 1
    return read_status_file(status_paths[0])


def stall_writer(
    recorder: RequestEventRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[threading.Event, threading.Event]:
    """Pause the first post-stop write and return entry/release events."""
    writer_entered = threading.Event()
    release_writer = threading.Event()
    original_write = recorder.write_event

    def stalled_write(
        request_event: RequestEvent,
        file_handle: TextIOWrapper,
        run_state: RecorderRunState,
    ) -> None:
        writer_entered.set()
        if not release_writer.wait(timeout=5):
            raise AssertionError("test did not release the stalled writer")
        else:
            pass
        original_write(request_event, file_handle, run_state)

    monkeypatch.setattr(recorder, "write_event", stalled_write)
    return writer_entered, release_writer


def fail_status_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make atomic status publication fail in one finalization test."""

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("synthetic status publication failure")

    monkeypatch.setattr(event_recorder_writer.os, "replace", fail_replace)
