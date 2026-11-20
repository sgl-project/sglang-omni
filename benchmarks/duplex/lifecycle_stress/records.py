# SPDX-License-Identifier: Apache-2.0
"""Records shared by the lifecycle stress client, its mock server and the summary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, JsonValue

SAMPLES_FILE = "samples.json"
SUMMARY_FILE = "summary.json"
SERVER_LOG_FILE = "server.log"
DEBUG_ROUTE_PATH = "/debug/lifecycle"
MS_PER_SECOND = 1000

Scenario = Literal["churn", "drop", "cancel", "idle", "reconnect", "long", "interrupt"]
Protocol = Literal["native", "legacy"]
LeavePoint = Literal["end", "update", "append", "reply"]
LeaveStyle = Literal["goodbye", "close_frame", "reset"]
IdlePoint = Literal["none", "created", "open"]
Outcome = Literal["completed", "dropped", "denied", "errored", "expired"]


class ServerSnapshot(BaseModel):
    """What the mock's debug route reports; a real server gives the resident memory and the device memory at most."""

    protocol: Protocol | None = None
    active_sessions: int | None = None
    rss_megabytes: float | None = None
    peak_rss_megabytes: float | None = None
    device_megabytes: float | None = None
    task_count: int | None = None
    thread_count: int | None = None


class SessionOutcome(BaseModel):
    outcome: Outcome
    ended_s: float
    close_code: int | None = None
    close_reason: str | None = None
    error: str | None = None
    expire_ms: float | None = None
    cancel_ms: float | None = None
    cancel_status: str | None = None
    late_outputs: int | None = None
    interrupted_replies: int | None = None
    reconnect_denials: int | None = None
    readmit_ms: float | None = None


class ServerLog(BaseModel):
    """What the server logged while the run lasted; counts only, because what a healthy model server logs is not known yet."""

    lines: int
    tracebacks: int
    error_lines: int
    top_error_lines: dict[str, int]


class CapacityProbe(BaseModel):
    """Connections held open together; every one admitted means no slot is still taken."""

    admitted: int
    denied: int
    errored: int


class CycleSample(BaseModel):
    cycle_index: int
    sessions: list[SessionOutcome]
    cleanup_latency_ms: float | None
    is_cleanup_timed_out: bool
    capacity_probe: CapacityProbe | None
    capacity_recovery_ms: float | None
    snapshot_after: ServerSnapshot | None


class StressSamples(BaseModel):
    scenario: Scenario
    protocol: Protocol
    url: str
    server: dict[str, JsonValue]
    source: dict[str, JsonValue]
    platform: str
    concurrency: int
    cycles_requested: int
    packets_per_session: int
    input_seconds: float
    is_input_silence: bool
    audio_file: str | None
    audio_sha256: str | None
    audio_conversion: dict[str, JsonValue] | None
    leave_point: LeavePoint
    leave_style: LeaveStyle
    idle_point: IdlePoint
    should_cancel_reply: bool
    reconnects: int
    server_unit_ms: int | None
    server_supports_cancel: bool | None
    server_timeout_s: float | None
    max_connections: int | None
    cleanup_wait_s: float
    cleanup_poll_interval_s: float
    has_debug_route: bool
    server_pid: int | None
    has_device_memory: bool
    device_models: list[str] | None
    baseline: ServerSnapshot | None
    cycles: list[CycleSample]
    final_session: SessionOutcome
    server_log: ServerLog | None
    closing: ServerSnapshot | None
