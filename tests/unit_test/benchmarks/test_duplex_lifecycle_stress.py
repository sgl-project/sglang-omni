# SPDX-License-Identifier: Apache-2.0
"""Tests of the realtime session lifecycle stress benchmark against its own mock server."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import JsonValue
from websockets.asyncio.client import ClientConnection

from benchmarks.duplex.client import PACKET_BYTES, PACKET_MS
from benchmarks.duplex.lifecycle_stress import host_metrics
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
    ServerLog,
    ServerSnapshot,
    SessionOutcome,
    StressSamples,
)
from benchmarks.duplex.lifecycle_stress.scenarios import (
    TRACEBACK_MARKER,
    RunSettings,
    run_stress,
)
from benchmarks.duplex.lifecycle_stress.session_client import (
    probe_capacity,
    run_recorded_session,
    take_slot,
)
from benchmarks.duplex.lifecycle_stress.summary import WARMUP_CYCLES, summarize
from sglang_omni.audio import smart_turn as smart_turn_module
from sglang_omni.serve.realtime import session as session_module
from sglang_omni.serve.realtime import turn_detector as turn_detector_module

CLEANUP_WAIT_S = 5.0
PACKETS_PER_SESSION = 2
SHORT_TIMEOUT_S = 0.3
STAND_IN_PID = 4242
# note (Yi Zhao): any nonzero sample is speech to the stand-in pipeline.
SPEECH_PACKET = b"\x01\x00" * (PACKET_BYTES // 2)
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
    server_pid: int | None = None,
    cleanup_wait_s: float = CLEANUP_WAIT_S,
    has_device_memory: bool = False,
    pcm: bytes = bytes(PACKETS_PER_SESSION * PACKET_BYTES),
    server_log: Path | None = None,
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
                pcm=pcm,
                is_input_silence=True,
                audio_file=None,
                audio_sha256=None,
                audio_conversion=None,
                packets_per_session=len(pcm) // PACKET_BYTES,
                leave_point=leave_point,
                leave_style=leave_style,
                idle_point=idle_point,
                reconnects=reconnects,
                max_connections=max_connections,
                server_pid=server_pid,
                has_debug_route=has_debug_route,
                has_device_memory=has_device_memory,
                server_log=server_log,
                cleanup_wait_s=cleanup_wait_s,
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
    assert cycle.capacity_recovery_ms is None
    assert cycle.snapshot_after is not None
    assert cycle.snapshot_after.device_megabytes is None
    assert samples.device_models is None
    assert samples.mock_settings is not None
    assert samples.mock_settings["max_connections"] == 2
    assert samples.duration_s > 0
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


def test_a_model_that_does_not_cancel_refuses_and_the_session_still_completes(
    tmp_path: Path,
) -> None:
    """The mock grants no cancel, as PersonaPlex does; the refusal is the expected answer and not an errored session."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2, supports_cancel=False),
            tmp_path,
            scenario="cancel",
        )
    )
    summary = summarize(samples)
    assert samples.server_supports_cancel is False
    assert summary.session_outcomes == {"completed": 2}
    assert summary.cancel_statuses == {"not_supported": 2}
    assert summary.late_outputs is None
    assert summary.failed_checks == []


def test_a_client_that_cancels_and_resets_at_once_is_released(tmp_path: Path) -> None:
    """Cancel and disconnect meet in the same teardown; the session must still leave the server."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2, held_first_replies=2),
            tmp_path,
            scenario="cancel",
            leave_point="reply",
            leave_style="reset",
        )
    )
    summary = summarize(samples)
    assert summary.session_outcomes == {"dropped": 2}
    assert summary.sessions_left == 0
    assert summary.checks["final_session_completed"] is True


def test_cancel_waits_for_the_whole_first_unit_of_a_model_with_a_longer_unit(
    tmp_path: Path,
) -> None:
    """A model whose unit spans two packets cannot reply after the first; the client reads the unit the server publishes and cancels once the reply exists."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(
                max_connections=2,
                unit_ms=PACKETS_PER_SESSION * PACKET_MS,
                held_first_replies=2,
            ),
            tmp_path,
            scenario="cancel",
        )
    )
    assert samples.server_unit_ms == PACKETS_PER_SESSION * PACKET_MS
    assert summarize(samples).cancel_statuses == {"cancelled": 2}


@pytest.mark.parametrize("has_debug_route", [True, False])
def test_a_server_that_keeps_a_closed_session_fails_the_run(
    tmp_path: Path, has_debug_route: bool
) -> None:
    """The mock keeps one closed session in its table; the run must fail on the session count where it can read it, and on the missing slot either way."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=3, leaked_sessions=1),
            tmp_path,
            max_connections=3,
            has_debug_route=has_debug_route,
            cleanup_wait_s=SHORT_TIMEOUT_S,
        )
    )
    summary = summarize(samples)
    probe = samples.cycles[0].capacity_probe
    assert probe is not None and (probe.admitted, probe.denied) == (2, 1)
    assert summary.checks["slots_recovered"] is False
    assert summary.checks["final_session_completed"] is True
    if has_debug_route:
        assert summary.sessions_left == 1
        assert summary.checks["no_sessions_left"] is False
    else:
        assert summary.unmeasured_checks == ["no_sessions_left"]


def test_a_session_the_probe_leaves_behind_is_not_in_its_cycle_but_fails_the_run(
    tmp_path: Path,
) -> None:
    """With no clients the probe is the only load; the mock keeps one of its sessions, which the cycle's reading taken before the probe must not hold and the closing reading must."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2, leaked_sessions=1),
            tmp_path,
            concurrency=0,
            max_connections=2,
            cleanup_wait_s=SHORT_TIMEOUT_S,
        )
    )
    cycle = samples.cycles[0]
    assert cycle.sessions == []
    assert cycle.snapshot_after is not None
    assert cycle.snapshot_after.active_sessions == 0
    summary = summarize(samples)
    assert summary.sessions_left == 1
    assert summary.failed_checks == ["no_sessions_left"]


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


@pytest.mark.parametrize(
    "mock",
    [MockSettings(max_connections=2), MockSettings(max_connections=3, model_slots=2)],
)
def test_capacity_probe_reports_a_slot_that_is_still_taken(mock: MockSettings) -> None:
    """Three sessions held together against two slots: the probe must see the missing one, whether the server refuses the connect or, as a turn-based pipeline does, the session.update."""

    async def probe() -> tuple[int, int]:
        async with serve_in_process(mock, "127.0.0.1", 0) as url:
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


def test_a_recorded_session_the_server_has_no_slot_for_is_denied(
    tmp_path: Path,
) -> None:
    """The recorder gives up after its retries without an error line; with the only slot held, that must read as a denial."""

    async def record_against_a_full_server() -> SessionOutcome:
        async with serve_in_process(
            MockSettings(max_connections=1), "127.0.0.1", 0
        ) as url:
            held = await take_slot(url, CLEANUP_WAIT_S)
            assert isinstance(held, ClientConnection)
            outcome = await run_recorded_session(
                url,
                bytes(PACKETS_PER_SESSION * PACKET_BYTES),
                tmp_path / "session" / "trace.jsonl",
                CLEANUP_WAIT_S,
                None,
            )
            await held.close()
            return outcome

    assert asyncio.run(record_against_a_full_server()).outcome == "denied"


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
    assert summary.capacity_recovery_ms.count == 1


def test_device_memory_is_the_sum_over_gpus_and_joins_every_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stand-in nvidia-smi reports two GPUs; their sum has to reach the baseline, the cycle and the closing reading next to what the debug route gives, and the run has to name both GPUs."""
    tool_directory = tmp_path / "tools"
    tool_directory.mkdir()
    stand_in = tool_directory / "nvidia-smi"
    stand_in.write_text("#!/bin/sh\necho 1000\necho 2000\n")
    stand_in.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tool_directory}{os.pathsep}{os.environ['PATH']}")
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2),
            tmp_path / "run",
            has_device_memory=True,
        )
    )
    cycle_snapshot = samples.cycles[0].snapshot_after
    assert samples.baseline is not None and samples.closing is not None
    assert cycle_snapshot is not None and cycle_snapshot.active_sessions == 0
    assert [
        samples.baseline.device_megabytes,
        cycle_snapshot.device_megabytes,
        samples.closing.device_megabytes,
    ] == [3000.0] * 3
    assert summarize(samples).device_megabytes_peak == 3000.0
    assert samples.device_models == ["1000", "2000"]


@pytest.mark.parametrize(
    ("speech_packets", "interrupted_replies"), [((0, 3), 1), ((0,), 0)]
)
def test_speech_over_a_reply_is_counted_as_an_interruption_and_its_control_is_not(
    tmp_path: Path, speech_packets: tuple[int, ...], interrupted_replies: int
) -> None:
    """The stand-in pipeline starts a reply on the silent packet after the speech and needs three more silent packets to finish it; speech in the fourth packet meets it, and the same input without that speech does not."""
    packets = [bytes(PACKET_BYTES)] * 6
    for packet_index in speech_packets:
        packets[packet_index] = SPEECH_PACKET
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2, takes_turns=True, reply_chunks=3),
            tmp_path,
            scenario="interrupt",
            pcm=b"".join(packets),
        )
    )
    summary = summarize(samples)
    assert summary.session_outcomes == {"completed": 2}
    assert summary.interrupted_replies == 2 * interrupted_replies
    assert summary.interrupted_sessions == 2 * interrupted_replies
    assert summary.late_outputs == 0
    assert summary.failed_checks == []


@pytest.mark.parametrize(
    ("scenario", "leave_point", "leave_style", "outcomes", "cancel_statuses"),
    [
        ("cancel", "end", "goodbye", {"completed": 2}, {"cancelled": 2}),
        ("drop", "reply", "reset", {"dropped": 2}, {}),
    ],
)
def test_a_reply_that_only_comes_after_the_turn_has_ended_is_still_met(
    tmp_path: Path,
    scenario: Scenario,
    leave_point: LeavePoint,
    leave_style: LeaveStyle,
    outcomes: dict[str, int],
    cancel_statuses: dict[str, int],
) -> None:
    """The stand-in pipeline answers only on the silent packet after the speech; a client that stopped sending to wait for the reply would never get one."""
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2, takes_turns=True, reply_chunks=8),
            tmp_path,
            scenario=scenario,
            leave_point=leave_point,
            leave_style=leave_style,
            pcm=SPEECH_PACKET + bytes(5 * PACKET_BYTES),
        )
    )
    summary = summarize(samples)
    assert summary.session_outcomes == outcomes
    assert summary.cancel_statuses == cancel_statuses
    assert summary.sessions_left == 0


def test_only_what_the_server_logs_during_the_run_is_kept_and_counted(
    tmp_path: Path,
) -> None:
    """A traceback from before the run must not count; the mock here logs nothing, so the test appends the run's lines itself once the run has taken its starting point."""
    log_path = tmp_path / "server.log"
    log_path.write_text(f"{TRACEBACK_MARKER}\nERROR old failure\n")
    logged_during_run = (
        f"INFO session 1 opened\n{TRACEBACK_MARKER}\n"
        "ERROR unknown session open index 17\nERROR unknown session open index 52\n"
    )

    async def stress_while_the_server_logs() -> StressSamples:
        run = asyncio.create_task(
            stress_against_mock(
                MockSettings(max_connections=2), tmp_path / "run", server_log=log_path
            )
        )
        while not (tmp_path / "run" / "traces").exists():
            await asyncio.sleep(0)
        with log_path.open("a") as log_file:
            log_file.write(logged_during_run)
        return await run

    samples = asyncio.run(stress_while_the_server_logs())
    assert samples.server_log == ServerLog(
        lines=4,
        tracebacks=1,
        error_lines=2,
        top_error_lines={"ERROR unknown session open index #": 2},
    )
    assert (tmp_path / "run" / "server.log").read_text() == logged_during_run


def test_a_model_server_on_this_host_gives_its_threads_and_open_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a debug route the counts come from /proc of the server's process tree; a stand-in /proc holds a server with seven threads and three descriptors and a stage process with two and one."""
    for pid, threads, descriptors in ((STAND_IN_PID, 7, 3), (STAND_IN_PID + 1, 2, 1)):
        process_directory = tmp_path / "proc" / str(pid)
        (process_directory / "fd").mkdir(parents=True)
        for descriptor in range(descriptors):
            (process_directory / "fd" / str(descriptor)).touch()
        (process_directory / "status").write_text(f"Threads:\t{threads}\n")
        (process_directory / "statm").write_text("0 256 0 0 0 0 0\n")
    monkeypatch.setattr(host_metrics, "PROC_ROOT", tmp_path / "proc")
    monkeypatch.setattr(
        host_metrics, "process_tree", lambda pid: [STAND_IN_PID, STAND_IN_PID + 1]
    )
    samples = asyncio.run(
        stress_against_mock(
            MockSettings(max_connections=2),
            tmp_path / "run",
            has_debug_route=False,
            server_pid=STAND_IN_PID,
        )
    )
    snapshot = samples.cycles[0].snapshot_after
    assert snapshot is not None
    assert (snapshot.process_count, snapshot.thread_count, snapshot.open_files) == (
        2,
        9,
        4,
    )
    assert snapshot.rss_megabytes == 2 * 256 * os.sysconf("SC_PAGE_SIZE") / 2**20


def test_the_process_tree_holds_a_child_this_process_started() -> None:
    """A model server's stages are its child processes; the tree of this process has to hold one it has just started."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        tree = host_metrics.process_tree(os.getpid())
    finally:
        child.kill()
        child.wait()
    assert tree[0] == os.getpid()
    assert child.pid in tree


@pytest.fixture
def restored_legacy_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy mock replaces these module attributes; registering them restores the real ones after the test."""
    monkeypatch.setattr(session_module, "StreamingVAD", session_module.StreamingVAD)
    monkeypatch.setattr(
        turn_detector_module, "StreamingVAD", turn_detector_module.StreamingVAD
    )
    monkeypatch.setattr(
        smart_turn_module, "load_smart_turn", smart_turn_module.load_smart_turn
    )


def test_the_idle_scenario_is_refused_on_the_legacy_protocol(
    tmp_path: Path, restored_legacy_modules: None
) -> None:
    """The legacy server publishes no idle limits, so an expiry could not be told from an early close; the run has to say so before it opens a session."""
    with pytest.raises(ValueError, match="native protocol"):
        asyncio.run(
            stress_against_mock(
                MockSettings(protocol="legacy"),
                tmp_path,
                scenario="idle",
                leave_style="close_frame",
                idle_point="open",
            )
        )


def test_legacy_sessions_close_with_a_frame_and_are_released(
    tmp_path: Path, restored_legacy_modules: None
) -> None:
    """The legacy protocol has no session.close and no connection cap; a close frame is its goodbye."""
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
    """Sessions still on the server fail the run, and memory growth is the fitted slope of the cycles after the warmup, which no warmup reading can turn negative."""
    # note (Yi Zhao): the warmup cycles read far higher than the rest, as a server that has just started does.
    resident_megabytes = [300.0] * WARMUP_CYCLES + [
        100.0 + 2.0 * index for index in range(4)
    ]
    samples = StressSamples(
        scenario="drop",
        protocol="legacy",
        url="ws://127.0.0.1:1/v1/realtime",
        started_at="2026-11-20T00:00:00+00:00",
        duration_s=1.0,
        server=SERVER,
        mock_settings=None,
        source={},
        platform="test",
        concurrency=1,
        cycles_requested=len(resident_megabytes),
        packets_per_session=PACKETS_PER_SESSION,
        input_seconds=0.16,
        is_input_silence=True,
        audio_file=None,
        audio_sha256=None,
        audio_conversion=None,
        leave_point="update",
        leave_style="close_frame",
        idle_point="none",
        should_cancel_reply=False,
        reconnects=0,
        server_unit_ms=None,
        server_supports_cancel=None,
        server_timeout_s=None,
        max_connections=None,
        cleanup_wait_s=CLEANUP_WAIT_S,
        cleanup_poll_interval_s=0.005,
        has_debug_route=True,
        server_pid=None,
        has_device_memory=True,
        device_models=None,
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
                    active_sessions=index + 1,
                    open_files=10 + 3 * index,
                    rss_megabytes=megabytes,
                    device_megabytes=10 * megabytes,
                ),
            )
            for index, megabytes in enumerate(resident_megabytes)
        ],
        final_session=SessionOutcome(outcome="completed", ended_s=0.0),
        server_log=None,
        closing=ServerSnapshot(active_sessions=len(resident_megabytes)),
    )
    summary = summarize(samples)
    assert summary.failed_checks == ["no_sessions_left"]
    assert summary.sessions_left == len(resident_megabytes)
    assert summary.cleanup_timeouts == len(resident_megabytes)
    assert summary.rss_growth_megabytes_per_cycle == pytest.approx(2.0)
    assert summary.device_growth_megabytes_per_cycle == pytest.approx(20.0)
    assert summary.device_megabytes_peak == 3000.0
    assert summary.open_files_growth == 9
    assert summary.thread_count_growth is None
