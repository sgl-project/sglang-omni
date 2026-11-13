# SPDX-License-Identifier: Apache-2.0
"""Pool lifecycle stress samples into one summary with pass/fail targets and a report table."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from pydantic import BaseModel, JsonValue

from benchmarks.duplex.lifecycle_stress.records import (
    SUMMARY_FILE,
    CloseStyle,
    DropPoint,
    DropStyle,
    IdlePoint,
    Scenario,
    ServerMode,
    ServerSnapshot,
    StressSamples,
)

# note (Yi Zhao): a steady server should not grow a full megabyte per cycle of churn.
RSS_GROWTH_BUDGET_MEGABYTES_PER_CYCLE = 1.0
# note (Yi Zhao): the allocator and the thread pool fill once during the first cycles; on the mock that growth ends by the fifth.
RSS_WARMUP_CYCLES = 5
REVISION_DISPLAY_LENGTH = 9
MS_PER_SECOND = 1000
UNMEASURED_SNAPSHOT = ServerSnapshot(taken_s=0.0)
REPORT_COLUMNS = (
    "scenario",
    "mode",
    "server revision",
    "client leaves",
    "processes x concurrency x cycles",
    "completed / dropped / denied / errored / expired",
    "sessions left",
    "cleanup p50 / p99 ms",
    "capacity recovery p99 ms",
    "scenario figures",
    "rss baseline -> final MB",
    "tasks / threads growth",
    "final probe",
)


class LatencyPercentiles(BaseModel):
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class ServerChange(BaseModel):
    """What one client process saw the server hold before its first cycle and after its last."""

    sessions_left: int | None
    rss_baseline_megabytes: float | None
    rss_final_megabytes: float | None
    rss_growth_megabytes: float | None
    task_growth: int | None
    thread_growth: int | None
    open_fd_growth: int | None


class StressSummary(BaseModel):
    scenario: Scenario
    url: str
    server: dict[str, JsonValue]
    source: dict[str, JsonValue]
    platform: str
    server_mode: ServerMode | None
    processes: int
    concurrency: int
    cycles: int
    appends_per_session: int
    close_style: CloseStyle
    drop_point: DropPoint
    drop_style: DropStyle
    idle_point: IdlePoint
    reconnects: int
    server_timeout_s: float | None
    max_connections: int | None
    has_debug_route: bool
    sessions_attempted: int
    sessions_completed: int
    sessions_dropped: int
    sessions_denied: int
    sessions_errored: int
    sessions_expired: int
    close_codes: dict[str, int]
    cleanup_latency: LatencyPercentiles | None
    cleanup_timeouts: int
    capacity_recovery: LatencyPercentiles | None
    capacity_probes_denied: int
    overflow_admitted: int
    overflow_denied: int
    expiry_overshoot: LatencyPercentiles | None
    reconnects_denied_first: int
    readmit: LatencyPercentiles | None
    cancel_latency: LatencyPercentiles | None
    cancels_not_cancelled: int
    late_outputs: int | None
    server_changes: list[ServerChange]
    rss_growth_per_cycle_megabytes: float | None
    is_final_probe_completed: bool
    client_cpu_cores_max: float
    client_peak_rss_megabytes_max: float
    unmet_targets: list[str]
    targets_not_measurable: list[str]


def latency_percentiles(latencies_ms: list[float]) -> LatencyPercentiles | None:
    if latencies_ms:
        p50_ms, p95_ms, p99_ms = np.percentile(latencies_ms, [50, 95, 99])
        return LatencyPercentiles(
            p50_ms=float(p50_ms),
            p95_ms=float(p95_ms),
            p99_ms=float(p99_ms),
            max_ms=max(latencies_ms),
        )
    else:
        return None


def count_growth(baseline: int | None, final: int | None) -> int | None:
    return None if baseline is None or final is None else final - baseline


def leave_label(
    idle_point: IdlePoint,
    drop_point: DropPoint,
    drop_style: DropStyle,
    close_style: CloseStyle,
) -> str:
    """How the sessions of a run end, as the report and the pooled directory names show it."""
    if idle_point != "none":
        return f"idle {idle_point}"
    elif drop_point != "none":
        return f"drop {drop_point} {drop_style}"
    else:
        return f"close {close_style}"


def summarize(samples: list[StressSamples]) -> StressSummary:
    assert samples, "no samples to summarize"
    first = samples[0]
    cycles = [cycle for sample in samples for cycle in sample.cycles]
    sessions = [session for cycle in cycles for session in cycle.sessions]
    outcome_counts = Counter(session.outcome for session in sessions)
    overflow_probes = [
        cycle.overflow_probe for cycle in cycles if cycle.overflow_probe is not None
    ]
    overflow_admitted = sum(probe.admitted for probe in overflow_probes)
    expiry_overshoots_ms = [
        session.expire_ms - (first.server_timeout_s or 0.0) * MS_PER_SECOND
        for session in sessions
        if session.expire_ms is not None
    ]
    cancels = [session for session in sessions if session.cancel_status is not None]
    cancels_not_cancelled = sum(
        session.cancel_status != "cancelled" for session in cancels
    )
    late_output_counts = [
        session.late_outputs for session in sessions if session.late_outputs is not None
    ]
    late_outputs = sum(late_output_counts) if late_output_counts else None
    close_codes: Counter[str] = Counter()
    for cycle in cycles:
        close_codes.update(cycle.close_codes)
    probes_denied = sum(
        cycle.capacity_probe.denied
        for cycle in cycles
        if cycle.capacity_probe is not None
    )
    server_changes: list[ServerChange] = []
    settled_rss_growth_per_cycle: list[float] = []
    for sample in samples:
        last_snapshot = sample.cycles[-1].snapshot_after if sample.cycles else None
        baseline = sample.baseline or UNMEASURED_SNAPSHOT
        final = last_snapshot or UNMEASURED_SNAPSHOT
        settled_cycles = len(sample.cycles) - RSS_WARMUP_CYCLES
        warmed = (
            sample.cycles[RSS_WARMUP_CYCLES - 1].snapshot_after
            if settled_cycles > 0
            else None
        ) or UNMEASURED_SNAPSHOT
        if warmed.rss_megabytes is not None and final.rss_megabytes is not None:
            settled_rss_growth_per_cycle.append(
                (final.rss_megabytes - warmed.rss_megabytes) / settled_cycles
            )
        else:
            pass
        server_changes.append(
            ServerChange(
                sessions_left=final.active_sessions,
                rss_baseline_megabytes=baseline.rss_megabytes,
                rss_final_megabytes=final.rss_megabytes,
                rss_growth_megabytes=(
                    None
                    if baseline.rss_megabytes is None or final.rss_megabytes is None
                    else final.rss_megabytes - baseline.rss_megabytes
                ),
                task_growth=count_growth(baseline.task_count, final.task_count),
                thread_growth=count_growth(baseline.thread_count, final.thread_count),
                open_fd_growth=count_growth(
                    baseline.open_fd_count, final.open_fd_count
                ),
            )
        )
    sessions_left = [change.sessions_left for change in server_changes]
    growth_per_cycle = (
        max(settled_rss_growth_per_cycle) if settled_rss_growth_per_cycle else None
    )
    is_final_probe_completed = all(
        sample.final_probe is not None and sample.final_probe.outcome == "completed"
        for sample in samples
    )
    targets: list[tuple[str, bool | None]] = [
        (
            "no sessions left after the last cycle",
            (
                None
                if all(left is None for left in sessions_left)
                else any(left not in (0, None) for left in sessions_left)
            ),
        ),
        (
            "no cleanup wait timed out",
            any(cycle.is_cleanup_timed_out for cycle in cycles),
        ),
        (
            "no admission denied below the connection cap",
            None if first.scenario == "drop" else outcome_counts["denied"] > 0,
        ),
        (
            "every capacity probe fully admitted",
            (
                None
                if first.max_connections is None
                else any(cycle.capacity_recovery_ms is None for cycle in cycles)
            ),
        ),
        ("final probe session completed", not is_final_probe_completed),
        (
            "server RSS growth per cycle under budget",
            (
                None
                if growth_per_cycle is None
                else growth_per_cycle > RSS_GROWTH_BUDGET_MEGABYTES_PER_CYCLE
            ),
        ),
        ("no errored sessions", outcome_counts["errored"] > 0),
    ]
    if first.scenario == "fill":
        targets.append(("every connection above the cap denied", overflow_admitted > 0))
    elif first.scenario == "idle":
        targets += [
            (
                "every idle session closed by the server's timeout",
                outcome_counts["expired"] < len(sessions),
            ),
            (
                "no session closed before the server's timeout",
                any(overshoot_ms < 0 for overshoot_ms in expiry_overshoots_ms),
            ),
        ]
    elif first.scenario == "cancel":
        targets += [
            (
                "every cancelled reply ended as cancelled",
                cancels_not_cancelled > 0 or len(cancels) < len(sessions),
            ),
            (
                "no output of a cancelled reply after its end",
                None if late_outputs is None else late_outputs > 0,
            ),
        ]
    else:
        pass
    return StressSummary(
        scenario=first.scenario,
        url=first.url,
        server=first.server,
        source=first.source,
        platform=first.platform,
        server_mode=None if first.baseline is None else first.baseline.mode,
        processes=len(samples),
        concurrency=sum(sample.concurrency for sample in samples),
        cycles=first.cycles_requested,
        appends_per_session=first.appends_per_session,
        close_style=first.close_style,
        drop_point=first.drop_point,
        drop_style=first.drop_style,
        idle_point=first.idle_point,
        reconnects=first.reconnects,
        server_timeout_s=first.server_timeout_s,
        max_connections=first.max_connections,
        has_debug_route=all(sample.has_debug_route for sample in samples),
        sessions_attempted=sum(outcome_counts.values()),
        sessions_completed=outcome_counts["completed"],
        sessions_dropped=outcome_counts["dropped"],
        sessions_denied=outcome_counts["denied"],
        sessions_errored=outcome_counts["errored"],
        sessions_expired=outcome_counts["expired"],
        close_codes=dict(close_codes),
        cleanup_latency=latency_percentiles(
            [
                cycle.cleanup_latency_ms
                for cycle in cycles
                if cycle.cleanup_latency_ms is not None
            ]
        ),
        cleanup_timeouts=sum(cycle.is_cleanup_timed_out for cycle in cycles),
        capacity_recovery=latency_percentiles(
            [
                cycle.capacity_recovery_ms
                for cycle in cycles
                if cycle.capacity_recovery_ms is not None
            ]
        ),
        capacity_probes_denied=probes_denied,
        overflow_admitted=overflow_admitted,
        overflow_denied=sum(probe.denied for probe in overflow_probes),
        expiry_overshoot=latency_percentiles(expiry_overshoots_ms),
        reconnects_denied_first=sum(
            bool(session.reconnect_denials) for session in sessions
        ),
        readmit=latency_percentiles(
            [
                session.readmit_ms
                for session in sessions
                if session.readmit_ms is not None
            ]
        ),
        cancel_latency=latency_percentiles(
            [session.cancel_ms for session in cancels if session.cancel_ms is not None]
        ),
        cancels_not_cancelled=cancels_not_cancelled,
        late_outputs=late_outputs,
        server_changes=server_changes,
        rss_growth_per_cycle_megabytes=growth_per_cycle,
        is_final_probe_completed=is_final_probe_completed,
        client_cpu_cores_max=max(
            sample.client_cpu_s / sample.client_wall_s for sample in samples
        ),
        client_peak_rss_megabytes_max=max(
            sample.client_peak_rss_megabytes for sample in samples
        ),
        unmet_targets=[label for label, is_unmet in targets if is_unmet is True],
        targets_not_measurable=[
            label for label, is_unmet in targets if is_unmet is None
        ],
    )


def render_report(summaries: list[StressSummary]) -> str:
    lines = [
        "| " + " | ".join(REPORT_COLUMNS) + " |",
        "|" + "---|" * len(REPORT_COLUMNS),
    ]
    for summary in summaries:
        cleanup = summary.cleanup_latency
        recovery = summary.capacity_recovery
        overshoot = summary.expiry_overshoot
        readmit = summary.readmit
        cancel = summary.cancel_latency
        if summary.scenario == "fill":
            figures = f"above the cap: {summary.overflow_denied} denied, {summary.overflow_admitted} admitted"
        elif summary.scenario == "idle" and overshoot is not None:
            figures = f"closed {overshoot.p50_ms:.0f} / {overshoot.p99_ms:.0f} ms after the {summary.server_timeout_s:g} s timeout"
        elif summary.scenario == "reconnect" and readmit is not None:
            figures = f"{summary.reconnects_denied_first} reconnects denied first, readmitted in {readmit.p50_ms:.0f} / {readmit.p99_ms:.0f} ms"
        elif summary.scenario == "cancel" and cancel is not None:
            late_outputs = (
                "unread" if summary.late_outputs is None else summary.late_outputs
            )
            figures = f"cancelled in {cancel.p50_ms:.0f} / {cancel.p99_ms:.0f} ms, {summary.cancels_not_cancelled} not cancelled, late outputs {late_outputs}"
        else:
            figures = "-"
        cells = (
            summary.scenario,
            summary.server_mode or "-",
            str(summary.server["revision"])[:REVISION_DISPLAY_LENGTH],
            leave_label(
                summary.idle_point,
                summary.drop_point,
                summary.drop_style,
                summary.close_style,
            ),
            f"{summary.processes} x {summary.concurrency} x {summary.cycles}",
            f"{summary.sessions_completed} / {summary.sessions_dropped} / "
            f"{summary.sessions_denied} / {summary.sessions_errored} / "
            f"{summary.sessions_expired}",
            ", ".join(
                "-" if change.sessions_left is None else str(change.sessions_left)
                for change in summary.server_changes
            ),
            (
                "-"
                if cleanup is None
                else f"{cleanup.p50_ms:.0f} / {cleanup.p99_ms:.0f}"
            ),
            "-" if recovery is None else f"{recovery.p99_ms:.0f}",
            figures,
            ", ".join(
                (
                    "-"
                    if change.rss_baseline_megabytes is None
                    or change.rss_final_megabytes is None
                    else f"{change.rss_baseline_megabytes:.0f} -> {change.rss_final_megabytes:.0f}"
                )
                for change in summary.server_changes
            ),
            ", ".join(
                (
                    "-"
                    if change.task_growth is None or change.thread_growth is None
                    else f"{change.task_growth:+d} / {change.thread_growth:+d}"
                )
                for change in summary.server_changes
            ),
            "completed" if summary.is_final_probe_completed else "failed",
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_summary(samples: list[StressSamples], output: Path) -> StressSummary:
    summary = summarize(samples)
    output.mkdir(parents=True, exist_ok=True)
    (output / SUMMARY_FILE).write_text(summary.model_dump_json(indent=2) + "\n")
    return summary
