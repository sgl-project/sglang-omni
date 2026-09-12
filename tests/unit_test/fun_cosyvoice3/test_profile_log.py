# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sglang_omni.models.fun_cosyvoice3.profile_log import (
    log_cosy_profile,
    request_ids_of,
)
from sglang_omni.profiler.event_recorder import get_recorder, reset_active_stage


def _read_events(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def test_request_ids_of_reads_scheduler_request_id() -> None:
    requests = [
        SimpleNamespace(request_id="r-a"),
        SimpleNamespace(request_id="r-b"),
    ]
    assert request_ids_of(requests) == ["r-a", "r-b"]


def test_inactive_log_is_silent(tmp_path: Path) -> None:
    rec = get_recorder()
    if rec.is_active():
        rec.stop()
    log_cosy_profile("ar_decode", request_ids=["r1"], batch=4)
    assert list(tmp_path.iterdir()) == []


def test_active_log_writes_one_jsonl_event(tmp_path: Path) -> None:
    rec = get_recorder()
    if rec.is_active():
        rec.stop()
    reset_active_stage(None)
    path = rec.start(run_id="cosy-test", event_dir=str(tmp_path), stage="tts_engine")
    try:
        log_cosy_profile(
            "ar_decode",
            request_ids=["r1", "r2"],
            batch=2,
        )
    finally:
        rec.stop()
    events = _read_events(path)
    assert len(events) == 1
    event = events[0]
    assert event["event_name"] == "cosy_ar_decode"
    assert event["request_id"] == "r1"
    assert event["metadata"]["batch"] == 2
    assert event["metadata"]["request_ids"] == ["r1", "r2"]
    assert event["metadata"]["clock"] == "CLOCK_MONOTONIC"
    assert isinstance(event["timestamp_ns"], int)
    assert isinstance(event["metadata"]["monotonic_ns"], int)
