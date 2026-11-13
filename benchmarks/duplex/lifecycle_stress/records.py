# SPDX-License-Identifier: Apache-2.0
"""Records shared by the lifecycle stress client, its mock server and the summary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

SAMPLES_FILE = "samples.json"
SUMMARY_FILE = "summary.json"
REPORT_FILE = "report.md"

Scenario = Literal["churn", "drop", "fill", "idle", "reconnect", "cancel"]
CloseStyle = Literal["event", "frame"]
DropPoint = Literal["none", "append", "update"]
DropStyle = Literal["abort", "close"]
IdlePoint = Literal["none", "created", "open"]
ServerMode = Literal["deployment", "legacy"]
Outcome = Literal["completed", "dropped", "denied", "errored", "expired"]


class ServerSnapshot(BaseModel):
    """What the server knows about itself; every field but the clock is None on a real server."""

    model_config = ConfigDict(extra="ignore")
    taken_s: float
    mode: ServerMode | None = None
    active_sessions: int | None = None
    rss_megabytes: float | None = None
    peak_rss_megabytes: float | None = None
    task_count: int | None = None
    thread_count: int | None = None
    open_fd_count: int | None = None


class SessionOutcome(BaseModel):
    outcome: Outcome
    close_code: int | None = None
    close_reason: str | None = None
    error: str | None = None
    connect_ms: float | None = None
    admit_ms: float | None = None
    expire_ms: float | None = None
    reconnect_denials: int | None = None
    readmit_ms: float | None = None
    cancel_ms: float | None = None
    cancel_status: str | None = None
    late_outputs: int | None = None
    wall_ms: float
    ended_s: float


class CapacityProbe(BaseModel):
    admitted: int
    denied: int
    errored: int
    wall_ms: float


class CycleSample(BaseModel):
    cycle_index: int
    wall_ms: float
    sessions: list[SessionOutcome]
    close_codes: dict[str, int]
    cleanup_latency_ms: float | None
    is_cleanup_timed_out: bool
    capacity_probe: CapacityProbe | None
    capacity_recovery_ms: float | None
    overflow_probe: CapacityProbe | None = None
    snapshot_after: ServerSnapshot | None


class StressSamples(BaseModel):
    scenario: Scenario
    url: str
    server: dict[str, JsonValue]
    source: dict[str, JsonValue]
    platform: str
    concurrency: int
    cycles_requested: int
    appends_per_session: int
    close_style: CloseStyle
    drop_point: DropPoint
    drop_style: DropStyle
    idle_point: IdlePoint = "none"
    reconnects: int = 0
    server_timeout_s: float | None = None
    max_connections: int | None
    has_debug_route: bool
    server_pid: int | None
    baseline: ServerSnapshot | None
    cycles: list[CycleSample]
    final_probe: SessionOutcome | None
    client_wall_s: float
    client_cpu_s: float
    client_peak_rss_megabytes: float
