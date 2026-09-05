# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sglang_omni.models.moss_tts_realtime.observability import (
    emit_realtime_event,
    realtime_identity_metadata,
)
from sglang_omni.profiler.event_recorder import get_recorder


def test_realtime_event_keeps_wall_clock_and_adds_monotonic_metadata(
    tmp_path: Path,
) -> None:
    recorder = get_recorder()
    recorder.stop()
    path = Path(
        recorder.start(
            run_id="moss-realtime-test",
            event_dir=str(tmp_path),
            stage="tts_engine",
        )
    )
    try:
        assert emit_realtime_event(
            request_id="request-1",
            stage=None,
            event_name="prefill_gate_ready",
            metadata={"stable_token_count": 12, "monotonic_ns": "stale"},
            monotonic_ns=123,
        )
    finally:
        recorder.stop(run_id="moss-realtime-test")

    event = json.loads(path.read_text(encoding="utf-8").strip())
    assert event["request_id"] == "request-1"
    assert event["stage"] == "tts_engine"
    assert event["event_name"] == "moss_tts_realtime_prefill_gate_ready"
    assert isinstance(event["timestamp_ns"], int)
    assert event["metadata"]["monotonic_ns"] == 123
    assert event["metadata"]["stable_token_count"] == 12


def test_realtime_identity_metadata_preserves_zero_turn_index() -> None:
    assert realtime_identity_metadata(
        SimpleNamespace(session_id="session-1", turn_id="turn-1", turn_index=0)
    ) == {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "turn_index": 0,
    }


def test_realtime_event_is_a_noop_without_an_active_recorder() -> None:
    recorder = get_recorder()
    recorder.stop()

    assert not emit_realtime_event(
        request_id="request-inactive",
        stage="tts_engine",
        event_name="prefill_gate_ready",
    )
