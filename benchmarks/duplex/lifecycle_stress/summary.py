# SPDX-License-Identifier: Apache-2.0
"""Turn the raw records of one stress run into its numbers and its yes or no checks."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, JsonValue

from benchmarks.duplex.lifecycle_stress.records import (
    MS_PER_SECOND,
    SUMMARY_FILE,
    Protocol,
    Scenario,
    StressSamples,
)

# note (Yi Zhao): the server's resident memory still climbs over its first cycles on a healthy run, so growth is measured after them.
WARMUP_CYCLES = 5
MAX_REPORTED_ERRORS = 5


class Distribution(BaseModel):
    count: int
    p50: float | None
    p99: float | None
    maximum: float | None


class StressSummary(BaseModel):
    scenario: Scenario
    protocol: Protocol
    server: dict[str, JsonValue]
    concurrency: int
    cycles: int
    session_outcomes: dict[str, int]
    errors: list[str]
    sessions_left: int | None
    cleanup_latency_ms: Distribution
    cleanup_timeouts: int
    capacity_recovery_ms: Distribution
    capacity_unrecovered_cycles: int | None
    rss_megabytes_first: float | None
    rss_megabytes_last: float | None
    rss_megabytes_peak: float | None
    rss_growth_megabytes_per_cycle: float | None
    task_count_growth: int | None
    thread_count_growth: int | None
    cancel_ms: Distribution
    cancel_statuses: dict[str, int]
    late_outputs: int | None
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


def summarize(samples: StressSamples) -> StressSummary:
    sessions = [session for cycle in samples.cycles for session in cycle.sessions]
    snapshots = [
        cycle.snapshot_after
        for cycle in samples.cycles
        if cycle.snapshot_after is not None
    ]
    last_snapshot = snapshots[-1] if snapshots else None
    sessions_left = None if last_snapshot is None else last_snapshot.active_sessions

    rss_series = [
        snapshot.rss_megabytes
        for snapshot in snapshots
        if snapshot.rss_megabytes is not None
    ]
    rss_growth_megabytes_per_cycle: float | None = None
    task_count_growth: int | None = None
    thread_count_growth: int | None = None
    if len(snapshots) > WARMUP_CYCLES:
        warm_snapshot = snapshots[WARMUP_CYCLES - 1]
        assert last_snapshot is not None, "snapshots without a last one"
        measured_cycles = len(snapshots) - WARMUP_CYCLES
        if (
            warm_snapshot.rss_megabytes is not None
            and last_snapshot.rss_megabytes is not None
        ):
            rss_growth_megabytes_per_cycle = (
                last_snapshot.rss_megabytes - warm_snapshot.rss_megabytes
            ) / measured_cycles
        else:
            pass
        if (
            warm_snapshot.task_count is not None
            and last_snapshot.task_count is not None
        ):
            task_count_growth = last_snapshot.task_count - warm_snapshot.task_count
        else:
            pass
        if (
            warm_snapshot.thread_count is not None
            and last_snapshot.thread_count is not None
        ):
            thread_count_growth = (
                last_snapshot.thread_count - warm_snapshot.thread_count
            )
        else:
            pass
    else:
        pass
    peaks = [
        snapshot.peak_rss_megabytes
        for snapshot in snapshots
        if snapshot.peak_rss_megabytes is not None
    ]

    capacity_unrecovered_cycles: int | None
    if samples.max_connections is None:
        capacity_unrecovered_cycles = None
    else:
        capacity_unrecovered_cycles = sum(
            cycle.capacity_recovery_ms is None for cycle in samples.cycles
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
    reconnected = [session for session in sessions if session.readmit_ms is not None]
    errored = [session for session in sessions if session.outcome == "errored"]

    checks: dict[str, bool | None] = {
        "no_sessions_left": None if sessions_left is None else sessions_left == 0,
        "slots_recovered": (
            None
            if capacity_unrecovered_cycles is None
            else capacity_unrecovered_cycles == 0
        ),
        "final_session_completed": samples.final_session.outcome == "completed",
        "no_errored_sessions": not errored,
    }
    return StressSummary(
        scenario=samples.scenario,
        protocol=samples.protocol,
        server=samples.server,
        concurrency=samples.concurrency,
        cycles=len(samples.cycles),
        session_outcomes=dict(Counter(session.outcome for session in sessions)),
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
        rss_megabytes_first=rss_series[0] if rss_series else None,
        rss_megabytes_last=rss_series[-1] if rss_series else None,
        rss_megabytes_peak=max(peaks) if peaks else None,
        rss_growth_megabytes_per_cycle=rss_growth_megabytes_per_cycle,
        task_count_growth=task_count_growth,
        thread_count_growth=thread_count_growth,
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
