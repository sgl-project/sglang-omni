# SPDX-License-Identifier: Apache-2.0
"""Model-free session pipeline coverage for Full Duplex profiler wiring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from sglang_omni.profiler.duplex_metrics import build_duplex_session_metrics
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.profiler.profiler_control import ProfilerControlClient
from sglang_omni.proto import OmniRequest
from tests.unit_test.fixtures.session_pipeline import chunk, pipeline


@pytest.mark.asyncio
async def test_session_pipeline_emits_joinable_duplex_trace(tmp_path: Path) -> None:
    event_dir = tmp_path / "events"
    recorder = get_recorder()
    recorder.stop()
    async with pipeline(tmp_path, stage_count=3) as (coordinator, _, _):
        run_id = "session-observability-test"
        recorder.start(run_id, str(event_dir), "coordinator")
        control = ProfilerControlClient(
            {name: info.control_endpoint for name, info in coordinator.stages.items()}
        )
        profiler_started = False
        try:
            await control.broadcast_start(
                run_id=run_id,
                trace_path_template=str(tmp_path / "trace"),
                event_dir=str(event_dir),
                enable_torch=False,
            )
            profiler_started = True
            await asyncio.sleep(0.1)
            session_identity = await coordinator.open_session(
                OmniRequest(None), stages=["source", "middle", "sink"]
            )
            outputs = coordinator.session_outputs(session_identity)
            await coordinator.append_session(session_identity, chunk(0, eos=True))
            output_kinds: list[str] = []
            while True:
                output = await asyncio.wait_for(anext(outputs), 5)
                output_kinds.append(output.kind)
                if output.kind == "input_done":
                    break
                else:
                    pass
            assert output_kinds == ["data", "input_done"]
            await outputs.aclose()
        finally:
            if profiler_started:
                await control.broadcast_stop(run_id=run_id)
            else:
                pass
            await control.close()
            recorder.stop()

    events = [
        json.loads(line)
        for path in sorted(event_dir.glob("events_*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    request_id = next(
        event["request_id"]
        for event in events
        if event["event_name"] == "session_unit_admitted"
    )
    unit_events = [event for event in events if event["request_id"] == request_id]
    event_names = {event["event_name"] for event in unit_events}
    assert {
        "session_unit_admitted",
        "stage_dispatch",
        "session_stage_started",
        "stage_complete",
        "session_output_emitted",
        "session_unit_finished",
    }.issubset(event_names)
    assert "session_stage_queued" not in event_names
    assert "session_stage_finished" not in event_names

    report = build_duplex_session_metrics(event_dir)
    unit = report["units"][0]
    stages = unit["stages"]
    assert unit["session_id"] == session_identity.id
    assert unit["input_seq"] == 0
    assert [stage["stage"] for stage in stages] == ["source", "middle", "sink"]
    assert all(
        stage["queued_timestamp_ns"] is not None
        and stage["started_timestamp_ns"] is not None
        and stage["finished_timestamp_ns"] is not None
        for stage in stages
    )
    assert all(
        stage["queue_ms"] is not None
        and stage["service_ms"] is not None
        and stage["queue_ms"] >= 0
        and stage["service_ms"] >= 0
        for stage in stages
    )
    assert all(
        (
            stage["handoff_from_previous_ms"] is None
            if index == 0
            else stage["handoff_from_previous_ms"] is not None
            and stage["handoff_from_previous_ms"] >= 0
        )
        for index, stage in enumerate(stages)
    )
