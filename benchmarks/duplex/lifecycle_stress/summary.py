# SPDX-License-Identifier: Apache-2.0
"""Pool lifecycle stress samples into one summary with pass/fail targets and a report table."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import JsonValue

from benchmarks.duplex.lifecycle_stress.records import (
    REPORT_FILE,
    SUMMARY_FILE,
    CycleSample,
    StressSamples,
)
from benchmarks.duplex.pacing_probe import percentiles_ms

MS_PER_SECOND = 1000
# note (Yi Zhao): a steady server should not grow a full megabyte per cycle of churn.
RSS_GROWTH_BUDGET_MEGABYTES_PER_CYCLE = 1.0
REPORT_COLUMNS = (
    "scenario",
    "mode",
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


def count_outcomes(cycles: list[CycleSample], outcome: str) -> int:
    return sum(
        session.outcome == outcome for cycle in cycles for session in cycle.sessions
    )


def last_snapshot_field(samples: StressSamples, field: str) -> int | float | None:
    snapshot = samples.cycles[-1].snapshot_after if samples.cycles else None
    return None if snapshot is None else getattr(snapshot, field)


def growth(samples: StressSamples, field: str) -> int | float | None:
    baseline = None if samples.baseline is None else getattr(samples.baseline, field)
    final = last_snapshot_field(samples, field)
    return None if baseline is None or final is None else final - baseline


def format_ms(value: JsonValue) -> str:
    return "-" if value is None else f"{value:.0f}"


def summarize(samples: list[StressSamples]) -> dict[str, JsonValue]:
    assert samples, "no samples to summarize"
    first = samples[0]
    cycles = [cycle for sample in samples for cycle in sample.cycles]
    cleanup_s = [
        cycle.cleanup_latency_ms / MS_PER_SECOND
        for cycle in cycles
        if cycle.cleanup_latency_ms is not None
    ]
    recovery_s = [
        cycle.capacity_recovery_ms / MS_PER_SECOND
        for cycle in cycles
        if cycle.capacity_recovery_ms is not None
    ]
    close_codes: Counter[str] = Counter()
    for cycle in cycles:
        close_codes.update(cycle.close_codes)
    sessions_left = [
        last_snapshot_field(sample, "active_sessions") for sample in samples
    ]
    rss_growth = [growth(sample, "rss_megabytes") for sample in samples]
    measured_rss_growth = [value for value in rss_growth if value is not None]
    growth_per_cycle = (
        None
        if not measured_rss_growth
        else max(measured_rss_growth) / max(1, first.cycles_requested)
    )
    is_final_probe_completed = all(
        sample.final_probe is not None and sample.final_probe.outcome == "completed"
        for sample in samples
    )
    counts = {
        outcome: count_outcomes(cycles, outcome)
        for outcome in ("completed", "dropped", "denied", "errored")
    }
    probes_denied = sum(
        cycle.capacity_probe.denied
        for cycle in cycles
        if cycle.capacity_probe is not None
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
            counts["denied"] > 0 if first.scenario == "churn" else None,
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
        ("no errored sessions", counts["errored"] > 0),
    ]
    return {
        "scenario": first.scenario,
        "url": first.url,
        "server_mode": None if first.baseline is None else first.baseline.mode,
        "processes": len(samples),
        "concurrency": sum(sample.concurrency for sample in samples),
        "cycles": first.cycles_requested,
        "appends_per_session": first.appends_per_session,
        "close_style": first.close_style,
        "drop_point": first.drop_point,
        "drop_style": first.drop_style,
        "max_connections": first.max_connections,
        "has_debug_route": all(sample.has_debug_route for sample in samples),
        "sessions_attempted": sum(counts.values()),
        **{f"sessions_{outcome}": count for outcome, count in counts.items()},
        "close_codes": dict(close_codes),
        **(percentiles_ms(cleanup_s, "cleanup_latency") if cleanup_s else {}),
        "cleanup_timeouts": sum(cycle.is_cleanup_timed_out for cycle in cycles),
        **(percentiles_ms(recovery_s, "capacity_recovery") if recovery_s else {}),
        "capacity_probes_denied": probes_denied,
        "sessions_left_after_last_cycle": sessions_left,
        "rss_baseline_megabytes": [
            None if sample.baseline is None else sample.baseline.rss_megabytes
            for sample in samples
        ],
        "rss_final_megabytes": [
            last_snapshot_field(sample, "rss_megabytes") for sample in samples
        ],
        "rss_growth_megabytes": rss_growth,
        "rss_growth_per_cycle_megabytes": growth_per_cycle,
        "task_growth": [growth(sample, "task_count") for sample in samples],
        "thread_growth": [growth(sample, "thread_count") for sample in samples],
        "open_fd_growth": [growth(sample, "open_fd_count") for sample in samples],
        "final_probe_completed": is_final_probe_completed,
        "client_cpu_cores_max": max(
            sample.client_cpu_s / sample.client_wall_s for sample in samples
        ),
        "client_peak_rss_megabytes_max": max(
            sample.client_peak_rss_megabytes for sample in samples
        ),
        "python": first.python,
        "websockets": first.websockets,
        "platform": first.platform,
        "unmet_targets": [label for label, is_unmet in targets if is_unmet is True],
        "targets_not_measurable": [
            label for label, is_unmet in targets if is_unmet is None
        ],
    }


def report_row(summary: dict[str, JsonValue]) -> str:
    sessions_left = summary["sessions_left_after_last_cycle"]
    assert isinstance(sessions_left, list)
    rss_baseline = summary["rss_baseline_megabytes"]
    rss_final = summary["rss_final_megabytes"]
    assert isinstance(rss_baseline, list) and isinstance(rss_final, list)
    task_growth = summary["task_growth"]
    thread_growth = summary["thread_growth"]
    assert isinstance(task_growth, list) and isinstance(thread_growth, list)
    cells = (
        str(summary["scenario"]),
        str(summary["server_mode"] or "-"),
        (
            f"{summary['drop_point']} {summary['drop_style']}"
            if summary["scenario"] == "drop"
            else f"close {summary['close_style']}"
        ),
        f"{summary['processes']} x {summary['concurrency']} x {summary['cycles']}",
        f"{summary['sessions_completed']} / {summary['sessions_dropped']} / "
        f"{summary['sessions_denied']} / {summary['sessions_errored']}",
        ", ".join("-" if left is None else str(left) for left in sessions_left),
        f"{format_ms(summary.get('cleanup_latency_p50_ms'))} / "
        f"{format_ms(summary.get('cleanup_latency_p99_ms'))}",
        format_ms(summary.get("capacity_recovery_p99_ms")),
        ", ".join(
            "-" if before is None or after is None else f"{before:.0f} -> {after:.0f}"
            for before, after in zip(rss_baseline, rss_final)
        ),
        ", ".join(
            "-" if tasks is None or threads is None else f"{tasks:+d} / {threads:+d}"
            for tasks, threads in zip(task_growth, thread_growth)
        ),
        "completed" if summary["final_probe_completed"] else "failed",
    )
    return "| " + " | ".join(cells) + " |"


def render_report(summaries: list[dict[str, JsonValue]]) -> str:
    header = "| " + " | ".join(REPORT_COLUMNS) + " |"
    divider = "|" + "---|" * len(REPORT_COLUMNS)
    return (
        "\n".join([header, divider, *(report_row(summary) for summary in summaries)])
        + "\n"
    )


def write_summary(samples: list[StressSamples], output: Path) -> dict[str, JsonValue]:
    summary = summarize(samples)
    output.mkdir(parents=True, exist_ok=True)
    (output / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_report(summaries: list[dict[str, JsonValue]], output: Path) -> str:
    report = render_report(summaries)
    output.mkdir(parents=True, exist_ok=True)
    (output / REPORT_FILE).write_text(report)
    return report
