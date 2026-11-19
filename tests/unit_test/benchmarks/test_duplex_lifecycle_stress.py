# SPDX-License-Identifier: Apache-2.0
"""Tests of the realtime session lifecycle stress benchmark against its own mock server."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import JsonValue

from benchmarks.duplex.client import PACKET_BYTES
from benchmarks.duplex.lifecycle_stress.mock_server import (
    MockSettings,
    serve_in_process,
)
from benchmarks.duplex.lifecycle_stress.records import (
    CycleSample,
    IdlePoint,
    LeavePoint,
    LeaveStyle,
    Scenario,
    ServerSnapshot,
    SessionOutcome,
    StressSamples,
)
from benchmarks.duplex.lifecycle_stress.scenarios import RunSettings, run_stress
from benchmarks.duplex.lifecycle_stress.session_client import probe_capacity
from benchmarks.duplex.lifecycle_stress.summary import WARMUP_CYCLES, summarize
from sglang_omni.audio import smart_turn as smart_turn_module
from sglang_omni.serve.realtime import session as session_module
from sglang_omni.serve.realtime import turn_detector as turn_detector_module

CLEANUP_WAIT_S = 5.0
PACKETS_PER_SESSION = 2
SHORT_TIMEOUT_S = 0.3
SERVER: dict[str, JsonValue] = {"revision": "0" * 40, "model": "lifecycle-stress-mock"}


async def stress_against_mock(
    mock: MockSettings,
    output: Path,
    scenario: Scenario = "churn",
    concurrency: int = 2,
    leave_point: LeavePoint = "end",
    leave_style: LeaveStyle = "goodbye",
    idle_point: IdlePoint = "none",
    reconnects: int = 0,
    max_connections: int | None = None,
    has_debug_route: bool = True,
) -> StressSamples:
    """One cycle of the given scenario against a mock served on this event loop."""
    async with serve_in_process(mock, "127.0.0.1", 0) as url:
        return await run_stress(
            RunSettings(
                url=url,
                server=SERVER,
                scenario=scenario,
                protocol=mock.protocol,
                concurrency=concurrency,
                cycles=1,
                pcm=bytes(PACKETS_PER_SESSION * PACKET_BYTES),
                is_input_silence=True,
                packets_per_session=PACKETS_PER_SESSION,
                leave_point=leave_point,
                leave_style=leave_style,
                idle_point=idle_point,
                reconnects=reconnects,
                max_connections=max_connections,
                server_pid=None,
                has_debug_route=has_debug_route,
                cleanup_wait_s=CLEANUP_WAIT_S,
                output=output,
            )
        )


def test_churn_cycle_completes_and_every_check_passes(tmp_path: Path) -> None:
    """Sessions that say goodbye the recorder's way all complete, and the server is left empty with its slots back."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2), tmp_path, max_connections=2
        )
    )
    cycle = samples.cycles[0]
    assert [session.outcome for session in cycle.sessions] == ["completed"] * 2
    assert cycle.cleanup_latency_ms is not None
    assert cycle.capacity_probe is not None and cycle.capacity_probe.admitted == 2
    assert summarize(samples).checks == {
        "no_sessions_left": True,
        "slots_recovered": True,
        "final_session_completed": True,
        "no_errored_sessions": True,
    }


@pytest.mark.parametrize("leave_point", ["update", "append", "reply"])
@pytest.mark.parametrize("leave_style", ["reset", "close_frame"])
def test_a_client_that_leaves_early_is_released_by_the_shared_session(
    tmp_path: Path, leave_point: LeavePoint, leave_style: LeaveStyle
) -> None:
    """Leaving while the server still has a reply to send or to finish leaves no session behind; the mock holds the reply of those two sessions, so the reply point is certain."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(
                max_connections=2,
                held_first_replies=2 if leave_point == "reply" else 0,
            ),
            tmp_path,
            scenario="drop",
            leave_point=leave_point,
            leave_style=leave_style,
        )
    )
    cycle = samples.cycles[0]
    assert [session.outcome for session in cycle.sessions] == ["dropped"] * 2
    assert cycle.snapshot_after is not None
    assert cycle.snapshot_after.active_sessions == 0
    assert samples.final_session.outcome == "completed"


def test_cancel_ends_the_reply_in_progress_and_nothing_of_it_arrives_later(
    tmp_path: Path,
) -> None:
    """The mock holds the first reply of both sessions until the cancel arrives, so the cancel always meets a reply in progress."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2, held_first_replies=2),
            tmp_path,
            scenario="cancel",
        )
    )
    sessions = samples.cycles[0].sessions
    assert [session.outcome for session in sessions] == ["completed"] * 2
    assert [session.cancel_status for session in sessions] == ["cancelled"] * 2
    assert [session.late_outputs for session in sessions] == [0, 0]


@pytest.mark.parametrize("idle_point", ["created", "open"])
def test_idle_sessions_expire_no_earlier_than_the_published_timeout(
    tmp_path: Path, idle_point: IdlePoint
) -> None:
    """A silent session is closed by the server with the name of its limit, and not before that limit."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(
                max_connections=2,
                idle_timeout_s=SHORT_TIMEOUT_S,
                session_update_timeout_s=SHORT_TIMEOUT_S,
            ),
            tmp_path,
            scenario="idle",
            idle_point=idle_point,
        )
    )
    summary = summarize(samples)
    assert samples.server_timeout_s == SHORT_TIMEOUT_S
    assert summary.session_outcomes == {"expired": 2}
    assert summary.early_expiries == 0
    assert summary.sessions_left == 0


def test_a_session_the_model_has_no_room_for_is_denied_not_errored(
    tmp_path: Path,
) -> None:
    """An idle session keeps the only model slot, so the other one is rejected on session.update; that counts as a denial."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(
                max_connections=2, model_slots=1, idle_timeout_s=SHORT_TIMEOUT_S
            ),
            tmp_path,
            scenario="idle",
            idle_point="open",
        )
    )
    summary = summarize(samples)
    assert summary.session_outcomes == {"expired": 1, "denied": 1}
    assert summary.checks["no_errored_sessions"] is True


def test_reconnect_after_a_reset_is_denied_until_the_old_session_is_gone(
    tmp_path: Path,
) -> None:
    """With one slot, the mock keeps the reset session open until it has denied the reconnect once."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=1, should_close_after_denial=True),
            tmp_path,
            scenario="reconnect",
            concurrency=1,
            leave_point="append",
            leave_style="reset",
            reconnects=1,
        )
    )
    first, second = samples.cycles[0].sessions
    assert first.reconnect_denials is None
    assert second.reconnect_denials is not None and second.reconnect_denials >= 1
    assert second.readmit_ms is not None and second.readmit_ms > 0


def test_capacity_probe_reports_a_slot_that_is_still_taken() -> None:
    """Three connections held together against a cap of two: the probe must see the missing slot."""

    async def probe() -> tuple[int, int]:
        async with serve_in_process(
            MockSettings(max_connections=2), "127.0.0.1", 0
        ) as url:
            result = await probe_capacity(url, 3, CLEANUP_WAIT_S)
            return result.admitted, result.denied

    assert asyncio.run(probe()) == (2, 1)


def test_long_scenario_runs_the_recorder_and_keeps_its_trace(tmp_path: Path) -> None:
    """A whole conversation goes through the duplex recorder, whose trace is the raw record."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2), tmp_path, scenario="long", concurrency=1
        )
    )
    assert samples.cycles[0].sessions[0].outcome == "completed"
    assert (tmp_path / "traces/cycle-000/session-000/trace.jsonl").exists()


def test_without_the_debug_route_the_session_count_is_unmeasured_not_passed(
    tmp_path: Path,
) -> None:
    """A real server has no debug route; the check it would feed must be reported as unmeasured."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2),
            tmp_path,
            max_connections=2,
            has_debug_route=False,
        )
    )
    summary = summarize(samples)
    assert summary.unmeasured_checks == ["no_sessions_left"]
    assert summary.checks["slots_recovered"] is True
    assert summary.cleanup_latency_ms.count == 0


def test_legacy_sessions_close_with_a_frame_and_are_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy protocol has no session.close and no connection cap; a close frame is its goodbye."""
    # note (Yi Zhao): the legacy mock replaces these module attributes; registering them restores the real ones after the test.
    monkeypatch.setattr(session_module, "StreamingVAD", session_module.StreamingVAD)
    monkeypatch.setattr(
        turn_detector_module, "StreamingVAD", turn_detector_module.StreamingVAD
    )
    monkeypatch.setattr(
        smart_turn_module, "load_smart_turn", smart_turn_module.load_smart_turn
    )
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(protocol="legacy"), tmp_path, leave_style="close_frame"
        )
    )
    summary = summarize(samples)
    assert summary.session_outcomes == {"completed": 2}
    assert summary.sessions_left == 0
    assert summary.unmeasured_checks == ["slots_recovered"]


def test_summary_fails_on_leftover_sessions_and_measures_growth_after_warmup() -> None:
    """Sessions still on the server fail the run, and memory growth is the slope after the warmup cycles only."""
    measured_cycles = 4
    resident_megabytes = [100.0] * WARMUP_CYCLES + [
        100.0 + 2.0 * (index + 1) for index in range(measured_cycles)
    ]
    samples = StressSamples(
        scenario="drop",
        protocol="legacy",
        url="ws://127.0.0.1:1/v1/realtime",
        server=SERVER,
        source={},
        platform="test",
        concurrency=1,
        cycles_requested=len(resident_megabytes),
        packets_per_session=PACKETS_PER_SESSION,
        input_seconds=0.16,
        is_input_silence=True,
        leave_point="update",
        leave_style="close_frame",
        idle_point="none",
        should_cancel_reply=False,
        reconnects=0,
        server_timeout_s=None,
        max_connections=None,
        cleanup_wait_s=CLEANUP_WAIT_S,
        cleanup_poll_interval_s=0.005,
        has_debug_route=True,
        server_pid=None,
        baseline=None,
        cycles=[
            CycleSample(
                cycle_index=index,
                sessions=[SessionOutcome(outcome="dropped", ended_s=float(index))],
                cleanup_latency_ms=None,
                is_cleanup_timed_out=True,
                capacity_probe=None,
                capacity_recovery_ms=None,
                snapshot_after=ServerSnapshot(
                    active_sessions=index + 1, rss_megabytes=megabytes
                ),
            )
            for index, megabytes in enumerate(resident_megabytes)
        ],
        final_session=SessionOutcome(outcome="completed", ended_s=0.0),
    )
    summary = summarize(samples)
    assert summary.failed_checks == ["no_sessions_left"]
    assert summary.sessions_left == len(resident_megabytes)
    assert summary.cleanup_timeouts == len(resident_megabytes)
    assert summary.rss_growth_megabytes_per_cycle == pytest.approx(2.0)
