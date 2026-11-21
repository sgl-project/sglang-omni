# SPDX-License-Identifier: Apache-2.0
"""Records shared by the lifecycle stress client, its mock server and the summary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, JsonValue

SAMPLES_FILE = "samples.json"
SUMMARY_FILE = "summary.json"
SERVER_LOG_FILE = "server.log"
DEBUG_ROUTE_PATH = "/debug/lifecycle"
DEBUG_SETTINGS_PATH = "/debug/lifecycle/settings"
MS_PER_SECOND = 1000

Scenario = Literal["churn", "drop", "cancel", "idle", "reconnect", "long", "interrupt"]
Protocol = Literal["native", "legacy"]
LeavePoint = Literal["end", "update", "append", "reply"]
LeaveStyle = Literal["goodbye", "close_frame", "reset"]
IdlePoint = Literal["none", "created", "open"]
Outcome = Literal["completed", "dropped", "denied", "errored", "expired"]


class ServerSnapshot(BaseModel):
    """What the mock's debug route reports; a real server on this host gives its resident memory, threads, open files and the device memory."""

    protocol: Protocol | None = None
    active_sessions: int | None = None
    rss_megabytes: float | None = None
    peak_rss_megabytes: float | None = None
    device_megabytes: float | None = None
    task_count: int | None = None
    thread_count: int | None = None
    open_files: int | None = None
    process_count: int | None = None


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
    # note (Yi Zhao): of a session that left at the reply: whether the client had not yet read the end of that reply when it left. False means the reply was already over and the session did not test a leave mid-reply.
    is_left_mid_reply: bool | None = None
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
    """One more session than the cap, held open together."""

    admitted: int
    denied: int
    errored: int

    def confirms(self, slots: int) -> bool:
        """Every slot is back and the cap is the server's: the cap admitted and the one above it denied. Without that denial a cap set too low would hide leaked slots behind the room above it."""
        return self.admitted == slots and self.denied == 1 and self.errored == 0


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
    started_at: str
    duration_s: float
    server: dict[str, JsonValue]
    mock_settings: dict[str, JsonValue] | None
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
    # note (Yi Zhao): native or turn_based, as the deployment publishes it; every session of a run against a turn-based one asks for its voice detection.
    server_interaction: str | None
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
    # note (Yi Zhao): empty in the file a run writes after each cycle, so a run that dies keeps the cycles it finished.
    final_session: SessionOutcome | None
    server_log: ServerLog | None
    # note (Yi Zhao): a last probe after the final session, because a probe shows only the leaks before it and nothing else follows the last cycle's probe and the final session.
    closing_probe: CapacityProbe | None
    closing: ServerSnapshot | None
