# SPDX-License-Identifier: Apache-2.0
"""Build the jsonl feedback report and the console summary for a stress run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.tts_serving.stream_stress.config_parser import StressConfig
from benchmarks.tts_serving.stream_stress.responder import (
    AggregateReport,
    RequestRecord,
)


def write_jsonl(
    config: StressConfig,
    records: list[RequestRecord],
    report: AggregateReport,
) -> Path:
    """Write one config row, one row per request, and a final summary row."""
    output_path = config.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: (not record.warmup, record.sequence))
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(dump_line({"record_type": "config", **config.snapshot()}))
        for record in ordered:
            handle.write(dump_line(record.to_dict(omit_chunks=config.omit_chunks)))
        handle.write(dump_line(report.to_dict()))
    return output_path


def write_summary_json(config: StressConfig, report: AggregateReport) -> Path | None:
    if config.summary_json is None:
        return None
    output_path = config.summary_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.snapshot(),
        "summary": report.to_dict(),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def print_console_report(
    config: StressConfig,
    report: AggregateReport,
    output: Path,
    summary_path: Path | None = None,
) -> None:
    metrics = report.to_dict()["metrics"]
    load = report.load
    planned = load.get("planned_requests")
    request_note = (
        f"{report.total_requests}"
        if planned == report.total_requests
        else f"{report.total_requests} (planned {planned})"
    )
    lines = [
        "",
        "TTS stream stress summary",
        f"run_id        : {report.run_id}"
        + (
            f"  labels={json.dumps(report.labels, ensure_ascii=False)}"
            if report.labels
            else ""
        ),
        f"target        : POST {config.url}  ({config.stream_format}/{config.response_format})",
        f"load          : QPM={config.qpm}, requests={request_note}"
        f" (+warmup {report.warmup_requests}), concurrency={config.max_concurrency or 'unlimited'}"
        f", arrival={config.arrival_mode}",
        f"wall clock    : {report.wall_clock_s:.3f}s   scheduled={load.get('scheduled')}"
        f"   cancelled={load.get('cancelled')}",
        f"results       : success={report.succeeded}/{report.dispatched}"
        f" ({report.success_rate * 100:.1f}%), failed={report.failed}",
        f"broken streams: {report.broken_streams}/{report.dispatched}"
        f" ({report.broken_rate * 100:.1f}%)"
        f"   reasons={json.dumps(report.broken_reasons, ensure_ascii=False)}",
        "",
        "  metric                    count      mean       p50       p95       p99       max",
        metric_line("TTFA (s)", metrics["ttfa_s"]),
        metric_line("latency (s)", metrics["latency_s"]),
        metric_line("RTF", metrics["rtf"]),
        metric_line("audio duration (s)", metrics["audio_duration_s"]),
        metric_line("chunks / request", metrics["chunk_count"]),
        metric_line("inter-chunk gap (s)", metrics["inter_chunk_s"]),
        metric_line("max underrun (s)", metrics["max_playback_underrun_s"]),
        "",
        "  continuity pass rate : "
        + "  ".join(
            f"{threshold_name}={'n/a' if value is None else f'{value:.2f}%'}"
            for threshold_name, value in metrics["continuity_pass_rate_pct"].items()
        ),
        f"  throughput           : achieved_qpm={report.achieved_qpm:.2f}"
        f"  qps={report.throughput_qps:.3f}"
        f"  audio_s/s={report.audio_throughput_s_per_s:.3f}",
        f"  load generator       : lagged={load.get('load_generator_lagged')}"
        f"  peak_inflight={load.get('peak_inflight')}"
        f"  max_lag_s={load.get('max_generator_lag_s')}"
        f"  peak_pending={load.get('peak_pending_tasks')}",
        f"  http status          : {json.dumps(report.http_status_counts)}",
        "",
        f"jsonl   : {output}",
    ]
    if summary_path is not None:
        lines.append(f"summary : {summary_path}")
    lines.append("")
    print("\n".join(lines), flush=True)


def metric_line(name: str, summary: dict[str, Any]) -> str:
    def cell(metric_name: str) -> str:
        value = summary.get(metric_name)
        return "n/a" if value is None else f"{value:.4f}"

    return (
        f"  {name:<24} {summary['count']:>6}"
        f" {cell('mean'):>10} {cell('p50'):>10} {cell('p95'):>10}"
        f" {cell('p99'):>10} {cell('max'):>10}"
    )


def dump_line(json_record: dict[str, Any]) -> str:
    return json.dumps(json_record, ensure_ascii=False, separators=(",", ":")) + "\n"
