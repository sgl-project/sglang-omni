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
    Scenario,
    ServerMode,
    ServerSnapshot,
    StressSamples,
)

# note (Yi Zhao): a steady server should not grow a full megabyte per cycle of churn.
RSS_GROWTH_BUDGET_MEGABYTES_PER_CYCLE = 1.0
REVISION_DISPLAY_LENGTH = 9
UNMEASURED_SNAPSHOT = ServerSnapshot(taken_s=0.0)
REPORT_COLUMNS = (
    "scenario",
    "mode",
    "server revision",
    "client leaves",
    "processes x concurrency x cycles",
    "completed / dropped / denied / errored",
    "sessions left",
    "cleanup p50 / p99 ms",
    "capacity recovery p99 ms",
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
    max_connections: int | None
    has_debug_route: bool
    sessions_attempted: int
    sessions_completed: int
    sessions_dropped: int
    sessions_denied: int
    sessions_errored: int
    close_codes: dict[str, int]
    cleanup_latency: LatencyPercentiles | None
    cleanup_timeouts: int
    capacity_recovery: LatencyPercentiles | None
    capacity_probes_denied: int
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


def summarize(samples: list[StressSamples]) -> StressSummary:
    assert samples, "no samples to summarize"
    first = samples[0]
    cycles = [cycle for sample in samples for cycle in sample.cycles]
    outcome_counts = Counter(
        session.outcome for cycle in cycles for session in cycle.sessions
    )
    close_codes: Counter[str] = Counter()
    for cycle in cycles:
        close_codes.update(cycle.close_codes)
    probes_denied = sum(
        cycle.capacity_probe.denied
        for cycle in cycles
        if cycle.capacity_probe is not None
    )
    server_changes: list[ServerChange] = []
    for sample in samples:
        last_snapshot = sample.cycles[-1].snapshot_after if sample.cycles else None
        baseline = sample.baseline or UNMEASURED_SNAPSHOT
        final = last_snapshot or UNMEASURED_SNAPSHOT
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
    measured_rss_growth = [
        change.rss_growth_megabytes
        for change in server_changes
        if change.rss_growth_megabytes is not None
    ]
    growth_per_cycle = (
        None
        if not measured_rss_growth
        else max(measured_rss_growth) / max(1, first.cycles_requested)
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
            outcome_counts["denied"] > 0 if first.scenario == "churn" else None,
        ),
        (
            "every capacity probe fully admitted",
            None if first.max_connections is None else probes_denied > 0,
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
        max_connections=first.max_connections,
        has_debug_route=all(sample.has_debug_route for sample in samples),
        sessions_attempted=sum(outcome_counts.values()),
        sessions_completed=outcome_counts["completed"],
        sessions_dropped=outcome_counts["dropped"],
        sessions_denied=outcome_counts["denied"],
        sessions_errored=outcome_counts["errored"],
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
        cells = (
            summary.scenario,
            summary.server_mode or "-",
            str(summary.server["revision"])[:REVISION_DISPLAY_LENGTH],
            (
                f"{summary.drop_point} {summary.drop_style}"
                if summary.scenario == "drop"
                else f"close {summary.close_style}"
            ),
            f"{summary.processes} x {summary.concurrency} x {summary.cycles}",
            f"{summary.sessions_completed} / {summary.sessions_dropped} / "
            f"{summary.sessions_denied} / {summary.sessions_errored}",
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
