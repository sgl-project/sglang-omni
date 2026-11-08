# SPDX-License-Identifier: Apache-2.0
"""Tests of the realtime session lifecycle stress benchmark against its own mock server."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from benchmarks.duplex.lifecycle_stress.mock_server import (
    MockSettings,
    serve_in_process,
)
from benchmarks.duplex.lifecycle_stress.records import (
    CycleSample,
    ServerSnapshot,
    SessionOutcome,
    StressSamples,
)
from benchmarks.duplex.lifecycle_stress.scenarios import RunSettings, run_stress
from benchmarks.duplex.lifecycle_stress.session_client import debug_url_of
from benchmarks.duplex.lifecycle_stress.summary import render_report, summarize
from sglang_omni.audio import smart_turn as smart_turn_module
from sglang_omni.serve.realtime import session as session_module
from sglang_omni.serve.realtime import turn_detector as turn_detector_module

CLEANUP_WAIT_S = 5.0
ABNORMAL_CLOSE_CODE = 1006


def settings_for(url: str, output: Path, **overrides: int | str) -> RunSettings:
    base = RunSettings(
        url=url,
        scenario="churn",
        concurrency=2,
        cycles=1,
        appends_per_session=1,
        close_style="event",
        drop_point="none",
        drop_style="abort",
        max_connections=None,
        server_pid=None,
        debug_url=debug_url_of(url),
        cleanup_wait_s=CLEANUP_WAIT_S,
        output=output,
    )
    for name, value in overrides.items():
        setattr(base, name, value)
    return base


async def stress_against_mock(
    mock: MockSettings, output: Path, **overrides: int | str
) -> StressSamples:
    async with serve_in_process(mock, "127.0.0.1", 0) as url:
        return await run_stress(settings_for(url, output, **overrides))


def test_churn_cycle_completes_every_session_and_the_server_releases_them(
    tmp_path: Path,
) -> None:
    """A clean open-update-append-close cycle ends with code 1000 and leaves no session behind."""
    samples = asyncio.run(
        stress_against_mock(MockSettings(max_connections=4), tmp_path)
    )
    cycle = samples.cycles[0]
    assert [session.outcome for session in cycle.sessions] == ["completed", "completed"]
    assert cycle.close_codes == {"1000": 2}
    assert cycle.cleanup_latency_ms is not None and not cycle.is_cleanup_timed_out
    assert cycle.snapshot_after is not None
    assert cycle.snapshot_after.active_sessions == 0
    assert cycle.snapshot_after.mode == "deployment"
    assert samples.final_probe is not None
    assert samples.final_probe.outcome == "completed"
    assert samples.has_debug_route


def test_drop_after_append_is_a_reset_the_shared_session_releases(
    tmp_path: Path,
) -> None:
    """A TCP reset after the last append must count as dropped, not completed, and free the slot."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=4),
            tmp_path,
            scenario="drop",
            drop_point="append",
            max_connections=4,
        )
    )
    cycle = samples.cycles[0]
    assert {session.outcome for session in cycle.sessions} == {"dropped"}
    assert cycle.close_codes == {str(ABNORMAL_CLOSE_CODE): 2}
    assert cycle.snapshot_after is not None
    assert cycle.snapshot_after.active_sessions == 0
    assert cycle.capacity_probe is not None
    assert cycle.capacity_probe.denied == 0
    assert cycle.capacity_recovery_ms is not None


def test_admission_denial_is_an_outcome_not_an_exception(tmp_path: Path) -> None:
    """With a cap of one, the second concurrent connection is denied with 503 and the run goes on."""
    samples = asyncio.run(
        stress_against_mock(MockSettings(max_connections=1), tmp_path, concurrency=2)
    )
    outcomes = sorted(session.outcome for session in samples.cycles[0].sessions)
    assert outcomes == ["completed", "denied"]
    assert samples.final_probe is not None
    assert samples.final_probe.outcome == "completed"


def test_legacy_mock_serves_a_frame_closed_session_without_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy mode must run the per-connection session with the VAD and Smart Turn loads stubbed out."""
    monkeypatch.setattr(session_module, "StreamingVAD", session_module.StreamingVAD)
    monkeypatch.setattr(
        turn_detector_module, "StreamingVAD", turn_detector_module.StreamingVAD
    )
    monkeypatch.setattr(
        smart_turn_module, "load_smart_turn", smart_turn_module.load_smart_turn
    )
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(mode="legacy"), tmp_path, concurrency=1, close_style="frame"
        )
    )
    cycle = samples.cycles[0]
    assert cycle.sessions[0].outcome == "completed"
    assert cycle.snapshot_after is not None
    assert cycle.snapshot_after.mode == "legacy"
    assert cycle.snapshot_after.active_sessions == 0


def stress_samples(
    active_sessions_left: int, rss_final_megabytes: float
) -> StressSamples:
    outcome = SessionOutcome(
        outcome="completed", close_code=1000, wall_ms=5.0, ended_s=1.0
    )
    return StressSamples(
        scenario="churn",
        url="ws://example/v1/realtime",
        concurrency=1,
        cycles_requested=2,
        appends_per_session=1,
        close_style="event",
        drop_point="none",
        drop_style="abort",
        max_connections=None,
        has_debug_route=True,
        server_pid=None,
        baseline=ServerSnapshot(
            taken_s=0.0, mode="deployment", active_sessions=0, rss_megabytes=100.0
        ),
        cycles=[
            CycleSample(
                cycle_index=index,
                wall_ms=10.0,
                sessions=[outcome],
                close_codes={"1000": 1},
                cleanup_latency_ms=1.0,
                is_cleanup_timed_out=False,
                capacity_probe=None,
                capacity_recovery_ms=None,
                snapshot_after=ServerSnapshot(
                    taken_s=1.0,
                    mode="deployment",
                    active_sessions=active_sessions_left,
                    rss_megabytes=rss_final_megabytes,
                ),
            )
            for index in range(2)
        ],
        final_probe=outcome,
        client_wall_s=1.0,
        client_cpu_s=0.1,
        client_peak_rss_megabytes=50.0,
        python="3.12",
        websockets="17",
        platform="test",
    )


def test_summary_targets_name_leftover_sessions_and_rss_growth() -> None:
    """Leftover sessions and a megabyte per cycle of RSS growth each fail their own target."""
    clean = summarize([stress_samples(0, 100.5)])
    leaking = summarize([stress_samples(3, 104.0)])
    assert clean["unmet_targets"] == []
    assert leaking["unmet_targets"] == [
        "no sessions left after the last cycle",
        "server RSS growth per cycle under budget",
    ]
    assert "every capacity probe fully admitted" in clean["targets_not_measurable"]
    report = render_report([clean, leaking])
    assert report.count("\n") == 4
    assert "| 3 |" in report
