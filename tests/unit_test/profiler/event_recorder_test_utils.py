# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for request event recorder tests."""

from __future__ import annotations

import json
import threading
from io import TextIOWrapper
from pathlib import Path

import pytest

from sglang_omni.profiler.event_recorder_types import QueuedRequestEvent
from sglang_omni.profiler.event_recorder_writer import RequestEventRecorder
from sglang_omni.utils.json import JsonValue


def read_events(path: str | Path) -> list[dict[str, JsonValue]]:
    with open(path, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def read_status(path: str | Path) -> dict[str, JsonValue]:
    status_paths = sorted(Path(path).parent.glob("run_status_*.json"))
    assert len(status_paths) == 1
    return json.loads(status_paths[0].read_text(encoding="utf-8"))


def read_status_file(path: str | Path) -> dict[str, JsonValue]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stall_writer(
    recorder: RequestEventRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[threading.Event, threading.Event]:
    """Pause one writer attempt and return entry/release barriers."""
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
