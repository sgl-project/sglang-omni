# SPDX-License-Identifier: Apache-2.0
"""Repeated cycles of concurrent sessions, with the server's recovery measured after each."""

from __future__ import annotations

import asyncio
import platform
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
from pydantic import JsonValue

from benchmarks.duplex.artifacts import source_fingerprint
from benchmarks.duplex.lifecycle_stress.host_metrics import peak_rss_megabytes
from benchmarks.duplex.lifecycle_stress.records import (
    SAMPLES_FILE,
    CapacityProbe,
    CloseStyle,
    CycleSample,
    DropPoint,
    DropStyle,
    IdlePoint,
    Scenario,
    SessionOutcome,
    StressSamples,
)
from benchmarks.duplex.lifecycle_stress.session_client import (
    CAPABILITIES_ROUTE_PATH,
    EXPIRY_LIMITS,
    SESSION_TIMEOUT_S,
    SNAPSHOT_TIMEOUT_S,
    Hold,
    SessionPlan,
    SnapshotSource,
    http_url_of,
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
HOLD_POLL_INTERVAL_S = 0.005
RECONNECT_RETRY_INTERVAL_S = 0.02


@dataclass(kw_only=True)
class RunSettings:
    url: str
    server: dict[str, JsonValue]
    scenario: Scenario
    concurrency: int
    cycles: int
    appends_per_session: int
    close_style: CloseStyle
    drop_point: DropPoint
    drop_style: DropStyle
    idle_point: IdlePoint
    reconnects: int
    max_connections: int | None
    server_pid: int | None
    debug_url: str | None
    cleanup_wait_s: float = CLEANUP_WAIT_S
    output: Path


async def await_cleanup(
    snapshot_source: SnapshotSource, last_ended_s: float, deadline_s: float
) -> tuple[float | None, bool]:
    """Poll until the server holds no session; (latency_ms, timed_out)."""
    while True:
        snapshot = await snapshot_source.fetch()
        if snapshot is None or snapshot.active_sessions is None:
            return None, False
        elif snapshot.active_sessions == 0:
            return (snapshot.taken_s - last_ended_s) * MS_PER_SECOND, False
        elif time.perf_counter() >= deadline_s:
            return None, True
        else:
            await asyncio.sleep(CLEANUP_POLL_INTERVAL_S)


async def fill_connection_cap(
    settings: RunSettings, plan: SessionPlan
) -> tuple[list[SessionOutcome], CapacityProbe]:
    """Hold the whole connection cap open, try to exceed it, then let the holders close."""
    assert settings.max_connections is not None, "fill needs the connection cap"
    release = asyncio.Event()
    holds = [Hold(release=release) for _ in range(settings.max_connections)]
    holders = [
        asyncio.create_task(
            run_lifecycle_session(settings.url, replace(plan, hold=hold))
        )
        for hold in holds
    ]
    while not all(
        hold.is_admitted or holder.done() for hold, holder in zip(holds, holders)
    ):
        await asyncio.sleep(HOLD_POLL_INTERVAL_S)
    overflow_probe = await probe_capacity(
        settings.url, settings.concurrency, settings.cleanup_wait_s
    )
    release.set()
    return list(await asyncio.gather(*holders)), overflow_probe


async def run_reconnecting_client(
    settings: RunSettings, plan: SessionPlan
) -> list[SessionOutcome]:
    """One client's sessions back to back; a denied reconnect retries until the server has room again."""
    outcomes = [await run_lifecycle_session(settings.url, plan)]
    for _ in range(settings.reconnects):
        left_s = outcomes[-1].ended_s
        denials = 0
        while True:
            attempt_started_s = time.perf_counter()
            outcome = await run_lifecycle_session(settings.url, plan)
            if (
                outcome.outcome == "denied"
                and time.perf_counter() < left_s + settings.cleanup_wait_s
            ):
                denials += 1
                await asyncio.sleep(RECONNECT_RETRY_INTERVAL_S)
            else:
                break
        outcomes.append(
            outcome.model_copy(
                update={
                    "reconnect_denials": denials,
                    "readmit_ms": (attempt_started_s - left_s) * MS_PER_SECOND,
                }
            )
        )
    return outcomes


async def run_cycle(
    settings: RunSettings,
    plan: SessionPlan,
    snapshot_source: SnapshotSource,
    cycle_index: int,
) -> CycleSample:
    cycle_started_s = time.perf_counter()
    overflow_probe: CapacityProbe | None = None
    if settings.scenario == "fill":
        outcomes, overflow_probe = await fill_connection_cap(settings, plan)
    elif settings.scenario == "reconnect":
        client_outcomes = await asyncio.gather(
            *(
                run_reconnecting_client(settings, plan)
                for _ in range(settings.concurrency)
            )
        )
        outcomes = [outcome for client in client_outcomes for outcome in client]
    else:
        outcomes = list(
            await asyncio.gather(
                *(
                    run_lifecycle_session(settings.url, plan)
                    for _ in range(settings.concurrency)
                )
            )
        )
    last_ended_s = max(outcome.ended_s for outcome in outcomes)
    deadline_s = last_ended_s + settings.cleanup_wait_s
    cleanup_latency_ms, is_cleanup_timed_out = await await_cleanup(
        snapshot_source, last_ended_s, deadline_s
    )
    probe: CapacityProbe | None = None
    capacity_recovery_ms: float | None = None
    if settings.max_connections is not None:
        while capacity_recovery_ms is None:
            probe_started_s = time.perf_counter()
            probe = await probe_capacity(
                settings.url, settings.max_connections, settings.cleanup_wait_s
            )
            if probe.denied == 0:
                capacity_recovery_ms = (probe_started_s - last_ended_s) * MS_PER_SECOND
            elif time.perf_counter() >= deadline_s:
                break
            else:
                await asyncio.sleep(RECOVERY_POLL_INTERVAL_S)
        probe_cleanup_ms, _ = await await_cleanup(
            snapshot_source, time.perf_counter(), deadline_s
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
        sessions=outcomes,
        close_codes=dict(Counter(str(outcome.close_code) for outcome in outcomes)),
        cleanup_latency_ms=cleanup_latency_ms,
        is_cleanup_timed_out=is_cleanup_timed_out,
        capacity_probe=probe,
        capacity_recovery_ms=capacity_recovery_ms,
        overflow_probe=overflow_probe,
        snapshot_after=await snapshot_source.fetch(),
    )


async def run_stress(settings: RunSettings) -> StressSamples:
    """Run every cycle, then one ordinary session that must still complete."""
    client_started_s = time.perf_counter()
    client_cpu_started_s = time.process_time()
    snapshot_source = SnapshotSource(
        debug_url=settings.debug_url, server_pid=settings.server_pid
    )
    server_timeout_s: float | None = None
    if settings.scenario == "idle":
        # note (Yi Zhao): the client can only tell an early close from a late one against the limit the server itself publishes.
        async with httpx.AsyncClient(timeout=SNAPSHOT_TIMEOUT_S) as http:
            response = await http.get(
                http_url_of(settings.url, CAPABILITIES_ROUTE_PATH)
            )
        response.raise_for_status()
        limit_name = f"{EXPIRY_LIMITS[settings.idle_point]}_s"
        server_timeout_s = float(response.json()["limits"][limit_name])
    else:
        pass
    plan = SessionPlan(
        appends=settings.appends_per_session,
        drop_point=settings.drop_point,
        drop_style=settings.drop_style,
        close_style=settings.close_style,
        idle_point=settings.idle_point,
        should_cancel_reply=settings.scenario == "cancel",
        timeout_s=SESSION_TIMEOUT_S + (server_timeout_s or 0.0),
    )
    baseline = await snapshot_source.fetch()
    cycles: list[CycleSample] = []
    for cycle_index in range(settings.cycles):
        cycles.append(await run_cycle(settings, plan, snapshot_source, cycle_index))
    final_plan = SessionPlan(
        appends=settings.appends_per_session, close_style=settings.close_style
    )
    final_probe = await run_lifecycle_session(settings.url, final_plan)
    samples = StressSamples(
        scenario=settings.scenario,
        url=settings.url,
        server=settings.server,
        source=source_fingerprint(),
        platform=platform.platform(),
        concurrency=settings.concurrency,
        cycles_requested=settings.cycles,
        appends_per_session=settings.appends_per_session,
        close_style=settings.close_style,
        drop_point=settings.drop_point,
        drop_style=settings.drop_style,
        idle_point=settings.idle_point,
        reconnects=settings.reconnects,
        server_timeout_s=server_timeout_s,
        max_connections=settings.max_connections,
        has_debug_route=(
            snapshot_source.debug_url is not None and snapshot_source.has_debug_route
        ),
        server_pid=settings.server_pid,
        baseline=baseline,
        cycles=cycles,
        final_probe=final_probe,
        client_wall_s=time.perf_counter() - client_started_s,
        client_cpu_s=time.process_time() - client_cpu_started_s,
        client_peak_rss_megabytes=peak_rss_megabytes(),
    )
    settings.output.mkdir(parents=True, exist_ok=True)
    (settings.output / SAMPLES_FILE).write_text(
        samples.model_dump_json(indent=2) + "\n"
    )
    return samples
