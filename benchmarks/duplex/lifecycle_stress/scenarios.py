# SPDX-License-Identifier: Apache-2.0
"""Repeated cycles of concurrent sessions, with the server's recovery measured after each."""

from __future__ import annotations

import asyncio
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, JsonValue

from benchmarks.duplex.artifacts import source_fingerprint
from benchmarks.duplex.client import PACKET_BYTES, SAMPLE_RATE
from benchmarks.duplex.lifecycle_stress.host_metrics import read_rss_megabytes
from benchmarks.duplex.lifecycle_stress.records import (
    DEBUG_ROUTE_PATH,
    MS_PER_SECOND,
    SAMPLES_FILE,
    CapacityProbe,
    CycleSample,
    IdlePoint,
    LeavePoint,
    LeaveStyle,
    Protocol,
    Scenario,
    ServerSnapshot,
    SessionOutcome,
    StressSamples,
)
from benchmarks.duplex.lifecycle_stress.session_client import (
    EXPIRY_LIMITS,
    SessionPlan,
    probe_capacity,
    run_lifecycle_session,
    run_recorded_session,
)

CAPABILITIES_ROUTE_PATH = "/v1/realtime/capabilities"
SESSION_TIMEOUT_S = 30.0
HTTP_TIMEOUT_S = 5.0
CLEANUP_POLL_INTERVAL_S = 0.005
RECOVERY_POLL_INTERVAL_S = 0.1
RECONNECT_RETRY_INTERVAL_S = 0.02
# note (Yi Zhao): a capacity probe's own sessions need to leave before the next cycle when no debug route can confirm it.
PROBE_SETTLE_S = 0.5


@dataclass(kw_only=True)
class RunSettings:
    url: str
    server: dict[str, JsonValue]
    scenario: Scenario
    protocol: Protocol
    concurrency: int
    cycles: int
    pcm: bytes
    is_input_silence: bool
    packets_per_session: int
    leave_point: LeavePoint
    leave_style: LeaveStyle
    idle_point: IdlePoint
    reconnects: int
    max_connections: int | None
    server_pid: int | None
    has_debug_route: bool
    cleanup_wait_s: float
    output: Path


class PublishedLimits(BaseModel):
    """The two idle limits a deployment publishes at its capabilities route."""

    idle_timeout_s: float
    session_update_timeout_s: float


class PublishedCapabilities(BaseModel):
    limits: PublishedLimits


@dataclass(kw_only=True)
class ServerObserver:
    """Where server state comes from: the mock's debug route, a local pid, or nothing."""

    http: httpx.AsyncClient
    realtime_url: str
    server_pid: int | None
    has_debug_route: bool

    def http_url(self, route_path: str) -> str:
        parts = urlsplit(self.realtime_url)
        scheme = "https" if parts.scheme == "wss" else "http"
        return urlunsplit((scheme, parts.netloc, route_path, "", ""))

    async def snapshot(self) -> ServerSnapshot | None:
        if self.has_debug_route:
            response = await self.http.get(self.http_url(DEBUG_ROUTE_PATH))
            response.raise_for_status()
            return ServerSnapshot.model_validate_json(response.text)
        elif self.server_pid is not None:
            return ServerSnapshot(rss_megabytes=read_rss_megabytes(self.server_pid))
        else:
            return None

    async def await_no_sessions(
        self, since_s: float, deadline_s: float
    ) -> tuple[float | None, bool]:
        """Poll until the server holds no session; (ms since since_s on this client's clock, timed out)."""
        if not self.has_debug_route:
            return None, False
        else:
            pass
        while True:
            snapshot = await self.snapshot()
            observed_s = time.perf_counter()
            assert snapshot is not None, "debug route without a snapshot"
            if snapshot.active_sessions == 0:
                return (observed_s - since_s) * MS_PER_SECOND, False
            elif observed_s >= deadline_s:
                return None, True
            else:
                await asyncio.sleep(CLEANUP_POLL_INTERVAL_S)


async def run_client(
    settings: RunSettings, plan: SessionPlan, trace_directory: Path
) -> list[SessionOutcome]:
    """One client's sessions back to back; a denied reconnect retries until the server has room again."""
    if settings.scenario == "long":
        return [
            await run_recorded_session(
                settings.url,
                settings.pcm,
                trace_directory / "trace.jsonl",
                plan.timeout_s,
            )
        ]
    else:
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
    settings: RunSettings, plan: SessionPlan, observer: ServerObserver, cycle_index: int
) -> CycleSample:
    cycle_directory = settings.output / "traces" / f"cycle-{cycle_index:03d}"
    client_outcomes = await asyncio.gather(
        *(
            run_client(settings, plan, cycle_directory / f"session-{index:03d}")
            for index in range(settings.concurrency)
        )
    )
    outcomes = [outcome for client in client_outcomes for outcome in client]
    last_ended_s = max(outcome.ended_s for outcome in outcomes)
    deadline_s = last_ended_s + settings.cleanup_wait_s
    cleanup_latency_ms, is_cleanup_timed_out = await observer.await_no_sessions(
        last_ended_s, deadline_s
    )
    probe: CapacityProbe | None = None
    capacity_recovery_ms: float | None = None
    if settings.max_connections is not None:
        while capacity_recovery_ms is None:
            probe_started_s = time.perf_counter()
            probe = await probe_capacity(
                settings.url, settings.max_connections, settings.cleanup_wait_s
            )
            # note (Yi Zhao): a probe whose connects failed on the client host admitted fewer than the cap without one denial; that is not a recovered server.
            if probe.admitted == settings.max_connections:
                capacity_recovery_ms = (probe_started_s - last_ended_s) * MS_PER_SECOND
            elif time.perf_counter() >= deadline_s:
                break
            else:
                await asyncio.sleep(RECOVERY_POLL_INTERVAL_S)
        probe_cleanup_ms, _ = await observer.await_no_sessions(
            time.perf_counter(), time.perf_counter() + settings.cleanup_wait_s
        )
        if probe_cleanup_ms is None:
            await asyncio.sleep(PROBE_SETTLE_S)
        else:
            pass
    else:
        pass
    return CycleSample(
        cycle_index=cycle_index,
        sessions=outcomes,
        cleanup_latency_ms=cleanup_latency_ms,
        is_cleanup_timed_out=is_cleanup_timed_out,
        capacity_probe=probe,
        capacity_recovery_ms=capacity_recovery_ms,
        snapshot_after=await observer.snapshot(),
    )


async def run_stress(settings: RunSettings) -> StressSamples:
    """Run every cycle, then one ordinary session that must still complete."""
    input_seconds = len(settings.pcm) / (2 * SAMPLE_RATE)
    assert settings.packets_per_session * PACKET_BYTES <= len(
        settings.pcm
    ), "input audio shorter than the packets of one session"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as http:
        observer = ServerObserver(
            http=http,
            realtime_url=settings.url,
            server_pid=settings.server_pid,
            has_debug_route=settings.has_debug_route,
        )
        server_timeout_s: float | None = None
        if settings.idle_point != "none":
            # note (Yi Zhao): the client can only tell an early close from a late one against the limit the server itself publishes.
            response = await http.get(observer.http_url(CAPABILITIES_ROUTE_PATH))
            response.raise_for_status()
            limits = PublishedCapabilities.model_validate_json(response.text).limits
            server_timeout_s = limits.model_dump()[
                f"{EXPIRY_LIMITS[settings.idle_point]}_s"
            ]
            session_timeout_s = SESSION_TIMEOUT_S + server_timeout_s
        else:
            session_timeout_s = SESSION_TIMEOUT_S + input_seconds
        plan = SessionPlan(
            protocol=settings.protocol,
            pcm=settings.pcm,
            packets=settings.packets_per_session,
            leave_point=settings.leave_point,
            leave_style=settings.leave_style,
            idle_point=settings.idle_point,
            should_cancel_reply=settings.scenario == "cancel",
            timeout_s=session_timeout_s,
        )
        baseline = await observer.snapshot()
        cycles: list[CycleSample] = []
        for cycle_index in range(settings.cycles):
            cycles.append(await run_cycle(settings, plan, observer, cycle_index))
        if settings.protocol == "native":
            final_session = await run_recorded_session(
                settings.url,
                settings.pcm[: settings.packets_per_session * PACKET_BYTES],
                settings.output / "traces" / "final" / "trace.jsonl",
                session_timeout_s,
            )
        else:
            # note (Yi Zhao): the recorder's legacy profile needs a turn detector this benchmark does not configure.
            final_session = await run_lifecycle_session(
                settings.url,
                SessionPlan(
                    protocol=settings.protocol,
                    pcm=settings.pcm,
                    packets=settings.packets_per_session,
                    leave_point="end",
                    leave_style="close_frame",
                    idle_point="none",
                    should_cancel_reply=False,
                    timeout_s=session_timeout_s,
                ),
            )
    samples = StressSamples(
        scenario=settings.scenario,
        protocol=settings.protocol,
        url=settings.url,
        server=settings.server,
        source=source_fingerprint(),
        platform=platform.platform(),
        concurrency=settings.concurrency,
        cycles_requested=settings.cycles,
        packets_per_session=settings.packets_per_session,
        input_seconds=input_seconds,
        is_input_silence=settings.is_input_silence,
        leave_point=settings.leave_point,
        leave_style=settings.leave_style,
        idle_point=settings.idle_point,
        should_cancel_reply=plan.should_cancel_reply,
        reconnects=settings.reconnects,
        server_timeout_s=server_timeout_s,
        max_connections=settings.max_connections,
        cleanup_wait_s=settings.cleanup_wait_s,
        cleanup_poll_interval_s=CLEANUP_POLL_INTERVAL_S,
        has_debug_route=settings.has_debug_route,
        server_pid=settings.server_pid,
        baseline=baseline,
        cycles=cycles,
        final_session=final_session,
    )
    settings.output.mkdir(parents=True, exist_ok=True)
    (settings.output / SAMPLES_FILE).write_text(
        samples.model_dump_json(indent=2) + "\n"
    )
    return samples
