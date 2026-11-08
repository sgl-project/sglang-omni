# SPDX-License-Identifier: Apache-2.0
"""Repeated cycles of concurrent sessions, with the server's recovery measured after each."""

from __future__ import annotations

import asyncio
import platform
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import websockets

from benchmarks.duplex.lifecycle_stress.host_metrics import peak_rss_megabytes
from benchmarks.duplex.lifecycle_stress.records import (
    SAMPLES_FILE,
    CapacityProbe,
    CloseStyle,
    CycleSample,
    DropPoint,
    DropStyle,
    Scenario,
    StressSamples,
)
from benchmarks.duplex.lifecycle_stress.session_client import (
    SessionPlan,
    SnapshotSource,
    probe_capacity,
    run_lifecycle_session,
)

MS_PER_SECOND = 1000
# Above the server's default cleanup_timeout_s of 30, so a slow adapter close is not a timeout here.
CLEANUP_WAIT_S = 35.0
CLEANUP_POLL_INTERVAL_S = 0.02
RECOVERY_POLL_INTERVAL_S = 0.1
# A capacity probe's own sessions need to leave before the next cycle when no debug route can confirm it.
PROBE_SETTLE_S = 0.5


@dataclass(kw_only=True)
class RunSettings:
    url: str
    scenario: Scenario
    concurrency: int
    cycles: int
    appends_per_session: int
    close_style: CloseStyle
    drop_point: DropPoint
    drop_style: DropStyle
    max_connections: int | None
    server_pid: int | None
    debug_url: str | None
    cleanup_wait_s: float = CLEANUP_WAIT_S
    output: Path


def session_plan(settings: RunSettings) -> SessionPlan:
    return SessionPlan(
        appends=settings.appends_per_session,
        drop_point=settings.drop_point,
        drop_style=settings.drop_style,
        close_style=settings.close_style,
    )


async def await_cleanup(
    source: SnapshotSource, last_ended_s: float, deadline_s: float
) -> tuple[float | None, bool]:
    """Poll until the server holds no session; (latency_ms, timed_out)."""
    while True:
        snapshot = await source.fetch()
        if snapshot is None or snapshot.active_sessions is None:
            return None, False
        elif snapshot.active_sessions == 0:
            return (snapshot.taken_s - last_ended_s) * MS_PER_SECOND, False
        elif time.perf_counter() >= deadline_s:
            return None, True
        else:
            await asyncio.sleep(CLEANUP_POLL_INTERVAL_S)


async def await_capacity(
    settings: RunSettings, last_ended_s: float, deadline_s: float
) -> tuple[CapacityProbe | None, float | None]:
    """Probe the cap until every slot is admitted; (last probe, recovery_ms)."""
    if settings.max_connections is None:
        return None, None
    else:
        pass
    while True:
        probe_started_s = time.perf_counter()
        probe = await probe_capacity(
            settings.url, settings.max_connections, settings.cleanup_wait_s
        )
        if probe.denied == 0:
            return probe, (probe_started_s - last_ended_s) * MS_PER_SECOND
        elif time.perf_counter() >= deadline_s:
            return probe, None
        else:
            await asyncio.sleep(RECOVERY_POLL_INTERVAL_S)


async def run_cycle(
    settings: RunSettings, source: SnapshotSource, cycle_index: int
) -> CycleSample:
    cycle_started_s = time.perf_counter()
    plan = session_plan(settings)
    outcomes = await asyncio.gather(
        *(
            run_lifecycle_session(settings.url, plan)
            for _ in range(settings.concurrency)
        )
    )
    last_ended_s = max(outcome.ended_s for outcome in outcomes)
    deadline_s = last_ended_s + settings.cleanup_wait_s
    cleanup_latency_ms, is_cleanup_timed_out = await await_cleanup(
        source, last_ended_s, deadline_s
    )
    probe, capacity_recovery_ms = await await_capacity(
        settings, last_ended_s, deadline_s
    )
    if probe is not None:
        probe_cleanup_ms, _ = await await_cleanup(
            source, time.perf_counter(), deadline_s
        )
        if probe_cleanup_ms is None:
            await asyncio.sleep(PROBE_SETTLE_S)
        else:
            pass
    else:
        pass
    return CycleSample(
        cycle_index=cycle_index,
        wall_ms=(time.perf_counter() - cycle_started_s) * MS_PER_SECOND,
        sessions=list(outcomes),
        close_codes=dict(Counter(str(outcome.close_code) for outcome in outcomes)),
        cleanup_latency_ms=cleanup_latency_ms,
        is_cleanup_timed_out=is_cleanup_timed_out,
        capacity_probe=probe,
        capacity_recovery_ms=capacity_recovery_ms,
        snapshot_after=await source.fetch(),
    )


async def run_stress(settings: RunSettings) -> StressSamples:
    """Run every cycle, then one ordinary session that must still complete."""
    client_started_s = time.perf_counter()
    client_cpu_started_s = time.process_time()
    source = SnapshotSource(
        debug_url=settings.debug_url, server_pid=settings.server_pid
    )
    baseline = await source.fetch()
    cycles: list[CycleSample] = []
    for cycle_index in range(settings.cycles):
        cycles.append(await run_cycle(settings, source, cycle_index))
    final_plan = SessionPlan(
        appends=settings.appends_per_session, close_style=settings.close_style
    )
    final_probe = await run_lifecycle_session(settings.url, final_plan)
    samples = StressSamples(
        scenario=settings.scenario,
        url=settings.url,
        concurrency=settings.concurrency,
        cycles_requested=settings.cycles,
        appends_per_session=settings.appends_per_session,
        close_style=settings.close_style,
        drop_point=settings.drop_point,
        drop_style=settings.drop_style,
        max_connections=settings.max_connections,
        has_debug_route=source.debug_url is not None and source.has_debug_route,
        server_pid=settings.server_pid,
        baseline=baseline,
        cycles=cycles,
        final_probe=final_probe,
        client_wall_s=time.perf_counter() - client_started_s,
        client_cpu_s=time.process_time() - client_cpu_started_s,
        client_peak_rss_megabytes=peak_rss_megabytes(),
        python=platform.python_version(),
        websockets=websockets.__version__,
        platform=platform.platform(),
    )
    settings.output.mkdir(parents=True, exist_ok=True)
    (settings.output / SAMPLES_FILE).write_text(
        samples.model_dump_json(indent=2) + "\n"
    )
    return samples
