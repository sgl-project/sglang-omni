# SPDX-License-Identifier: Apache-2.0
"""Turn the raw records of one stress run into its numbers and its yes or no checks."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, JsonValue

from benchmarks.duplex.lifecycle_stress.records import (
    MS_PER_SECOND,
    SUMMARY_FILE,
    Protocol,
    Scenario,
    ServerLog,
    StressSamples,
)

# note (Yi Zhao): the server's resident memory still climbs over its first cycles on a healthy run, so growth is measured after them.
WARMUP_CYCLES = 5
MAX_REPORTED_ERRORS = 5
MIN_GROWTH_POINTS = 2


class Distribution(BaseModel):
    count: int
    p50: float | None
    p99: float | None
    maximum: float | None


class StressSummary(BaseModel):
    scenario: Scenario
    protocol: Protocol
    started_at: str
    duration_s: float
    server: dict[str, JsonValue]
    mock_settings: dict[str, JsonValue] | None
    server_unit_ms: int | None
    server_supports_cancel: bool | None
    server_interaction: str | None
    device_models: list[str] | None
    input_seconds: float
    audio_file: str | None
    audio_sha256: str | None
    audio_conversion: dict[str, JsonValue] | None
    concurrency: int
    cycles: int
    session_outcomes: dict[str, int]
    unexpected_denials: int
    errors: list[str]
    sessions_left: int | None
    cleanup_latency_ms: Distribution
    cleanup_timeouts: int
    capacity_recovery_ms: Distribution
    capacity_unrecovered_cycles: int | None
    # note (Yi Zhao): probes that got more sessions in than --max-connections: the cap given is not the server's, and such a probe proves nothing about leaked slots.
    probes_above_cap: int | None
    rss_megabytes_first: float | None
    rss_megabytes_last: float | None
    rss_megabytes_peak: float | None
    rss_growth_megabytes_per_cycle: float | None
    device_megabytes_first: float | None
    device_megabytes_last: float | None
    device_megabytes_peak: float | None
    device_growth_megabytes_per_cycle: float | None
    task_count_growth: int | None
    thread_count_growth: int | None
    open_files_growth: int | None
    process_count_growth: int | None
    cancel_ms: Distribution
    cancel_statuses: dict[str, int]
    late_outputs: int | None
    drops_mid_reply: int | None
    drops_after_reply: int | None
    interrupted_replies: int | None
    interrupted_sessions: int | None
    server_log: ServerLog | None
    expire_overshoot_ms: Distribution
    early_expiries: int
    reconnects_denied_first: int
    readmit_ms: Distribution
    checks: dict[str, bool | None]
    failed_checks: list[str]
    unmeasured_checks: list[str]


def distribution(values: list[float]) -> Distribution:
    """Nearest-rank percentiles; every field is None when nothing was measured."""
    if not values:
        return Distribution(count=0, p50=None, p99=None, maximum=None)
    else:
        ordered = sorted(values)
        return Distribution(
            count=len(ordered),
            p50=ordered[math.ceil(0.5 * len(ordered)) - 1],
            p99=ordered[math.ceil(0.99 * len(ordered)) - 1],
            maximum=ordered[-1],
        )


def growth_per_cycle(readings: list[float]) -> float | None:
    """Slope of a line fitted to one reading per cycle; None below two readings."""
    if len(readings) >= MIN_GROWTH_POINTS:
        # note (Yi Zhao): a fitted line, because the difference of two readings moves with either one of them.
        return statistics.linear_regression(range(len(readings)), readings).slope
    else:
        return None


def count_growth(counts: list[int | None]) -> int | None:
    """The last reading of a count minus the first; None below two readings."""
    readings = [count for count in counts if count is not None]
    if len(readings) >= MIN_GROWTH_POINTS:
        return readings[-1] - readings[0]
    else:
        return None


def summarize(samples: StressSamples) -> StressSummary:
    sessions = [session for cycle in samples.cycles for session in cycle.sessions]
    snapshots = [
        cycle.snapshot_after
        for cycle in samples.cycles
        if cycle.snapshot_after is not None
    ]
    sessions_left = None if samples.closing is None else samples.closing.active_sessions

    rss_series = [
        snapshot.rss_megabytes
        for snapshot in snapshots
        if snapshot.rss_megabytes is not None
    ]
    measured_snapshots = snapshots[WARMUP_CYCLES:]
    measured_rss = [
        snapshot.rss_megabytes
        for snapshot in measured_snapshots
        if snapshot.rss_megabytes is not None
    ]
    device_series = [
        snapshot.device_megabytes
        for snapshot in snapshots
        if snapshot.device_megabytes is not None
    ]
    measured_device = [
        snapshot.device_megabytes
        for snapshot in measured_snapshots
        if snapshot.device_megabytes is not None
    ]
    peaks = [
        snapshot.peak_rss_megabytes
        for snapshot in snapshots
        if snapshot.peak_rss_megabytes is not None
    ]

    capacity_unrecovered_cycles: int | None
    probes_above_cap: int | None
    if samples.max_connections is None:
        capacity_unrecovered_cycles = None
        probes_above_cap = None
    else:
        # note (Yi Zhao): the probe after the final session counts with those of the cycles; a run that has not got to it yet has not shown its slots back.
        probes = [cycle.capacity_probe for cycle in samples.cycles] + [
            samples.closing_probe
        ]
        capacity_unrecovered_cycles = sum(
            probe is None or not probe.confirms(samples.max_connections)
            for probe in probes
        )
        probes_above_cap = sum(
            probe is not None and probe.admitted > samples.max_connections
            for probe in probes
        )

    expire_overshoots_ms: list[float] = []
    for session in sessions:
        if session.expire_ms is not None and session.outcome == "expired":
            assert (
                samples.server_timeout_s is not None
            ), "expiry without the server's timeout"
            expire_overshoots_ms.append(
                session.expire_ms - samples.server_timeout_s * MS_PER_SECOND
            )
        else:
            pass

    cancelled = [session for session in sessions if session.cancel_status is not None]
    late_output_counts = [
        session.late_outputs for session in sessions if session.late_outputs is not None
    ]
    interruption_counts = [
        session.interrupted_replies
        for session in sessions
        if session.interrupted_replies is not None
    ]
    reply_leaves = [
        session.is_left_mid_reply
        for session in sessions
        if session.is_left_mid_reply is not None
    ]
    reconnected = [session for session in sessions if session.readmit_ms is not None]
    errored = [session for session in sessions if session.outcome == "errored"]

    # note (Yi Zhao): a denial is the right answer only to the clients above the cap, and a server that admits nobody answers every session with one; without a cap to go by, no denial is expected.
    expected_denials_per_cycle = (
        0
        if samples.max_connections is None
        else max(0, samples.concurrency - samples.max_connections)
    )
    unexpected_denials = sum(
        max(
            0,
            sum(session.outcome == "denied" for session in cycle.sessions)
            - expected_denials_per_cycle,
        )
        for cycle in samples.cycles
    )
    checks: dict[str, bool | None] = {
        "no_sessions_left": None if sessions_left is None else sessions_left == 0,
        "slots_recovered": (
            None
            if capacity_unrecovered_cycles is None
            else capacity_unrecovered_cycles == 0
        ),
        "final_session_completed": samples.final_session is not None
        and samples.final_session.outcome == "completed",
        "no_errored_sessions": not errored,
        "no_unexpected_denials": unexpected_denials == 0,
    }
    return StressSummary(
        scenario=samples.scenario,
        protocol=samples.protocol,
        started_at=samples.started_at,
        duration_s=samples.duration_s,
        server=samples.server,
        mock_settings=samples.mock_settings,
        server_unit_ms=samples.server_unit_ms,
        server_supports_cancel=samples.server_supports_cancel,
        server_interaction=samples.server_interaction,
        device_models=samples.device_models,
        input_seconds=samples.input_seconds,
        audio_file=samples.audio_file,
        audio_sha256=samples.audio_sha256,
        audio_conversion=samples.audio_conversion,
        concurrency=samples.concurrency,
        cycles=len(samples.cycles),
        session_outcomes=dict(Counter(session.outcome for session in sessions)),
        unexpected_denials=unexpected_denials,
        errors=sorted({session.error for session in errored if session.error})[
            :MAX_REPORTED_ERRORS
        ],
        sessions_left=sessions_left,
        cleanup_latency_ms=distribution(
            [
                cycle.cleanup_latency_ms
                for cycle in samples.cycles
                if cycle.cleanup_latency_ms is not None
            ]
        ),
        cleanup_timeouts=sum(cycle.is_cleanup_timed_out for cycle in samples.cycles),
        capacity_recovery_ms=distribution(
            [
                cycle.capacity_recovery_ms
                for cycle in samples.cycles
                if cycle.capacity_recovery_ms is not None
            ]
        ),
        capacity_unrecovered_cycles=capacity_unrecovered_cycles,
        probes_above_cap=probes_above_cap,
        rss_megabytes_first=rss_series[0] if rss_series else None,
        rss_megabytes_last=rss_series[-1] if rss_series else None,
        rss_megabytes_peak=max(peaks) if peaks else None,
        rss_growth_megabytes_per_cycle=growth_per_cycle(measured_rss),
        device_megabytes_first=device_series[0] if device_series else None,
        device_megabytes_last=device_series[-1] if device_series else None,
        device_megabytes_peak=max(device_series) if device_series else None,
        device_growth_megabytes_per_cycle=growth_per_cycle(measured_device),
        task_count_growth=count_growth(
            [snapshot.task_count for snapshot in measured_snapshots]
        ),
        thread_count_growth=count_growth(
            [snapshot.thread_count for snapshot in measured_snapshots]
        ),
        open_files_growth=count_growth(
            [snapshot.open_files for snapshot in measured_snapshots]
        ),
        process_count_growth=count_growth(
            [snapshot.process_count for snapshot in measured_snapshots]
        ),
        cancel_ms=distribution(
            [
                session.cancel_ms
                for session in cancelled
                if session.cancel_ms is not None
            ]
        ),
        cancel_statuses=dict(
            Counter(str(session.cancel_status) for session in cancelled)
        ),
        late_outputs=sum(late_output_counts) if late_output_counts else None,
        drops_mid_reply=sum(reply_leaves) if reply_leaves else None,
        drops_after_reply=(
            len(reply_leaves) - sum(reply_leaves) if reply_leaves else None
        ),
        interrupted_replies=sum(interruption_counts) if interruption_counts else None,
        interrupted_sessions=(
            sum(count > 0 for count in interruption_counts)
            if interruption_counts
            else None
        ),
        server_log=samples.server_log,
        expire_overshoot_ms=distribution(expire_overshoots_ms),
        early_expiries=sum(overshoot_ms < 0 for overshoot_ms in expire_overshoots_ms),
        reconnects_denied_first=sum(
            bool(session.reconnect_denials) for session in reconnected
        ),
        readmit_ms=distribution(
            [
                session.readmit_ms
                for session in reconnected
                if session.readmit_ms is not None
            ]
        ),
        checks=checks,
        failed_checks=[name for name, passed in checks.items() if passed is False],
        unmeasured_checks=[name for name, passed in checks.items() if passed is None],
    )


def write_summary(samples: StressSamples, output: Path) -> StressSummary:
    summary = summarize(samples)
    (output / SUMMARY_FILE).write_text(summary.model_dump_json(indent=2) + "\n")
    return summary
