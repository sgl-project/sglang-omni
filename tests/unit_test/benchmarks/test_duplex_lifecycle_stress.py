# SPDX-License-Identifier: Apache-2.0
"""Tests of the realtime session lifecycle stress benchmark against its own mock server."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import JsonValue

from benchmarks.duplex.lifecycle_stress.mock_server import (
    MockSettings,
    serve_in_process,
)
from benchmarks.duplex.lifecycle_stress.records import (
    CapacityProbe,
    CycleSample,
    IdlePoint,
    Scenario,
    ServerSnapshot,
    SessionOutcome,
    StressSamples,
)
from benchmarks.duplex.lifecycle_stress.scenarios import RunSettings, run_stress
from benchmarks.duplex.lifecycle_stress.session_client import (
    DEBUG_ROUTE_PATH,
    http_url_of,
)
from benchmarks.duplex.lifecycle_stress.summary import (
    RSS_WARMUP_CYCLES,
    render_report,
    summarize,
)
from sglang_omni.audio import smart_turn as smart_turn_module
from sglang_omni.serve.realtime import session as session_module
from sglang_omni.serve.realtime import turn_detector as turn_detector_module

CLEANUP_WAIT_S = 5.0
ABNORMAL_CLOSE_CODE = 1006
POLICY_CLOSE_CODE = 1008
MS_PER_SECOND = 1000
SERVER_REVISION = "0123456789abcdef0123456789abcdef01234567"
SERVER: dict[str, JsonValue] = {
    "revision": SERVER_REVISION,
    "model": "lifecycle-stress-mock",
}


def settings_for(url: str, output: Path, **overrides: int | str) -> RunSettings:
    base = RunSettings(
        url=url,
        server=SERVER,
        scenario="churn",
        concurrency=2,
        cycles=1,
        appends_per_session=1,
        close_style="event",
        drop_point="none",
        drop_style="abort",
        idle_point="none",
        reconnects=0,
        max_connections=None,
        server_pid=None,
        debug_url=http_url_of(url, DEBUG_ROUTE_PATH),
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
    assert samples.server == SERVER
    assert "harness_git_head" in samples.source


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


def test_in_session_admission_rejection_is_a_denial_that_frees_its_socket(
    tmp_path: Path,
) -> None:
    """A model with one slot rejects the second session.update; the client counts a denial and closes."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=4, model_slots=1),
            tmp_path,
            concurrency=2,
            appends_per_session=3,
        )
    )
    cycle = samples.cycles[0]
    sessions = sorted(cycle.sessions, key=lambda session: session.outcome)
    assert [session.outcome for session in sessions] == ["completed", "denied"]
    assert sessions[1].error is not None
    assert "admission_rejected" in sessions[1].error
    assert cycle.snapshot_after is not None
    assert cycle.snapshot_after.active_sessions == 0
    assert samples.final_probe is not None
    assert samples.final_probe.outcome == "completed"


def test_fill_denies_every_connection_above_the_cap_and_recovers(
    tmp_path: Path,
) -> None:
    """While the cap is held open no further connection gets in, and the whole cap is free again once the holders close."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2),
            tmp_path,
            scenario="fill",
            concurrency=3,
            max_connections=2,
        )
    )
    cycle = samples.cycles[0]
    assert [session.outcome for session in cycle.sessions] == ["completed", "completed"]
    assert cycle.overflow_probe is not None
    assert (cycle.overflow_probe.admitted, cycle.overflow_probe.denied) == (0, 3)
    assert cycle.capacity_probe is not None
    assert cycle.capacity_probe.denied == 0
    assert summarize([samples]).unmet_targets == []


@pytest.mark.parametrize(
    ("idle_point", "close_reason"),
    [("created", "session_update_timeout"), ("open", "idle_timeout")],
)
def test_idle_sessions_are_closed_by_the_timeout_the_server_publishes(
    tmp_path: Path, idle_point: IdlePoint, close_reason: str
) -> None:
    """A silent client is closed with the policy code for its idle point, no earlier than the published limit, and released."""
    timeout_s = 0.2
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(
                max_connections=4,
                idle_timeout_s=timeout_s,
                session_update_timeout_s=timeout_s,
            ),
            tmp_path,
            scenario="idle",
            idle_point=idle_point,
        )
    )
    cycle = samples.cycles[0]
    assert samples.server_timeout_s == timeout_s
    assert {session.outcome for session in cycle.sessions} == {"expired"}
    assert {
        (session.close_code, session.close_reason) for session in cycle.sessions
    } == {(POLICY_CLOSE_CODE, close_reason)}
    assert all(
        session.expire_ms is not None and session.expire_ms >= timeout_s * MS_PER_SECOND
        for session in cycle.sessions
    )
    assert cycle.snapshot_after is not None
    assert cycle.snapshot_after.active_sessions == 0
    assert summarize([samples]).unmet_targets == []


def test_reconnect_at_a_full_cap_is_denied_until_the_old_session_is_released(
    tmp_path: Path,
) -> None:
    """A client that resets and reconnects while the server still closes its old session is denied first, then admitted."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2, close_delay_ms=100.0),
            tmp_path,
            scenario="reconnect",
            drop_point="append",
            reconnects=2,
        )
    )
    cycle = samples.cycles[0]
    assert [session.outcome for session in cycle.sessions] == ["dropped"] * 6
    reconnected = [
        session for session in cycle.sessions if session.reconnect_denials is not None
    ]
    assert len(reconnected) == 4
    assert all(session.reconnect_denials >= 1 for session in reconnected)
    assert cycle.snapshot_after is not None
    assert cycle.snapshot_after.active_sessions == 0
    assert summarize([samples]).reconnects_denied_first == 4


def test_cancel_ends_the_reply_in_progress_and_the_session_closes_normally(
    tmp_path: Path,
) -> None:
    """response.cancel during a slow reply gets a cancelled response.done, nothing of that reply after it, and a clean close."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=4, reply_chunks=5, chunk_delay_ms=50.0),
            tmp_path,
            scenario="cancel",
            appends_per_session=2,
        )
    )
    cycle = samples.cycles[0]
    assert [session.outcome for session in cycle.sessions] == ["completed", "completed"]
    assert {session.cancel_status for session in cycle.sessions} == {"cancelled"}
    assert {session.late_outputs for session in cycle.sessions} == {0}
    assert cycle.snapshot_after is not None
    assert cycle.snapshot_after.active_sessions == 0
    assert summarize([samples]).unmet_targets == []


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


COMPLETED_SESSION = SessionOutcome(
    outcome="completed", close_code=1000, wall_ms=5.0, ended_s=1.0
)


def stress_samples(
    active_sessions_left: int,
    rss_megabytes_after_cycles: list[float],
    scenario: Scenario = "churn",
    outcome: SessionOutcome = COMPLETED_SESSION,
    overflow_probe: CapacityProbe | None = None,
    server_timeout_s: float | None = None,
    max_connections: int | None = None,
) -> StressSamples:
    return StressSamples(
        scenario=scenario,
        url="ws://example/v1/realtime",
        server=SERVER,
        source={"harness_git_head": None},
        platform="test",
        concurrency=1,
        cycles_requested=len(rss_megabytes_after_cycles),
        appends_per_session=1,
        close_style="event",
        drop_point="none",
        drop_style="abort",
        server_timeout_s=server_timeout_s,
        max_connections=max_connections,
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
                overflow_probe=overflow_probe,
                snapshot_after=ServerSnapshot(
                    taken_s=1.0,
                    mode="deployment",
                    active_sessions=active_sessions_left,
                    rss_megabytes=rss_megabytes,
                ),
            )
            for index, rss_megabytes in enumerate(rss_megabytes_after_cycles)
        ],
        final_probe=COMPLETED_SESSION,
        client_wall_s=1.0,
        client_cpu_s=0.1,
        client_peak_rss_megabytes=50.0,
    )


def test_summary_targets_name_leftover_sessions_and_rss_growth() -> None:
    """Leftover sessions and steady RSS growth each fail their own target; growth during warm-up fails none."""
    warmed_then_flat = [103.0, 106.0, 108.0, 110.0, 110.0, 110.2, 110.4, 110.5]
    growing_every_cycle = [102.0 + 2.0 * index for index in range(8)]
    clean = summarize([stress_samples(0, warmed_then_flat)])
    leaking = summarize([stress_samples(3, growing_every_cycle)])
    short = summarize([stress_samples(0, warmed_then_flat[:RSS_WARMUP_CYCLES])])
    assert clean.unmet_targets == []
    assert clean.rss_growth_per_cycle_megabytes == pytest.approx(0.5 / 3)
    assert leaking.rss_growth_per_cycle_megabytes == pytest.approx(2.0)
    assert "server RSS growth per cycle under budget" in short.targets_not_measurable
    assert leaking.unmet_targets == [
        "no sessions left after the last cycle",
        "server RSS growth per cycle under budget",
    ]
    assert "every capacity probe fully admitted" in clean.targets_not_measurable
    report = render_report([clean, leaking])
    assert report.count("\n") == 4
    assert "| 3 |" in report
    assert f"| {SERVER_REVISION[:9]} |" in report


@pytest.mark.parametrize(
    ("samples", "unmet_target"),
    [
        (
            stress_samples(0, [100.0], max_connections=4),
            "every capacity probe fully admitted",
        ),
        (
            stress_samples(
                0,
                [100.0],
                scenario="fill",
                overflow_probe=CapacityProbe(
                    admitted=1, denied=2, errored=0, wall_ms=1.0
                ),
            ),
            "every connection above the cap denied",
        ),
        (
            stress_samples(
                0,
                [100.0],
                scenario="idle",
                outcome=COMPLETED_SESSION.model_copy(
                    update={"outcome": "expired", "expire_ms": 150.0}
                ),
                server_timeout_s=0.2,
            ),
            "no session closed before the server's timeout",
        ),
        (
            stress_samples(
                0,
                [100.0],
                scenario="reconnect",
                outcome=COMPLETED_SESSION.model_copy(update={"outcome": "denied"}),
            ),
            "no admission denied below the connection cap",
        ),
        (
            stress_samples(
                0,
                [100.0],
                scenario="cancel",
                outcome=COMPLETED_SESSION.model_copy(
                    update={"cancel_status": "cancelled", "late_outputs": 2}
                ),
            ),
            "no output of a cancelled reply after its end",
        ),
        (
            stress_samples(
                0,
                [100.0],
                scenario="cancel",
                outcome=COMPLETED_SESSION.model_copy(
                    update={"cancel_status": "completed", "late_outputs": 0}
                ),
            ),
            "every cancelled reply ended as cancelled",
        ),
    ],
)
def test_each_scenario_fails_the_target_named_after_its_own_fault(
    samples: StressSamples, unmet_target: str
) -> None:
    """A cap never probed back in full, an over-admitted connection, an early timeout, a reconnect never admitted, a late output and a reply that outran its cancel each fail one target."""
    assert summarize([samples]).unmet_targets == [unmet_target]
