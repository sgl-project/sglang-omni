# SPDX-License-Identifier: Apache-2.0
"""CLI for rendering request-level profiler reports.

Usage::

    python -m sglang_omni.profiler <event-dir-or-file> [--out report.json]
    python -m sglang_omni.profiler events/ --format table
"""

from __future__ import annotations

import argparse
import json
import sys

from sglang_omni.profiler.views import build_report, format_table


def _global_peak(gpu_global: dict, key: str) -> float | None:
    block = gpu_global.get(key) or {}
    return block.get("peak_mb")


def _global_mean(gpu_global: dict, key: str) -> float | None:
    block = gpu_global.get(key) or {}
    return block.get("mean_mb")


def _fmt_mb(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sglang_omni.profiler",
        description="Render request-level profiler views",
    )
    parser.add_argument(
        "source",
        help="Event JSONL file or directory of events_*.jsonl files",
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Output path; '-' writes to stdout (default)",
    )
    parser.add_argument(
        "--format",
        choices=("json", "table"),
        default="json",
        help="Report format (default: json)",
    )
    args = parser.parse_args(argv)

    report = build_report(args.source)
    if args.format == "json":
        text = json.dumps(report, indent=2)
    else:
        gpu = report.get("gpu_breakdown") or {}
        gpu_rows = gpu.get("per_request", [])
        gpu_global = gpu.get("global", {})
        gpu_section = (
            "\n## GPU & memory\n"
            + format_table(
                gpu_rows,
                [
                    "request_id",
                    "gpu_mem_allocated_peak_mb",
                    "gpu_mem_reserved_peak_mb",
                ],
            )
            + "\nGlobal GPU memory (allocated / reserved):\n"
            + f"  peak_mb : {_fmt_mb(_global_peak(gpu_global, 'gpu_mem_allocated'))} / "
            f"{_fmt_mb(_global_peak(gpu_global, 'gpu_mem_reserved'))}\n"
            + f"  mean_mb : {_fmt_mb(_global_mean(gpu_global, 'gpu_mem_allocated'))} / "
            f"{_fmt_mb(_global_mean(gpu_global, 'gpu_mem_reserved'))}\n"
        )
        text = (
            f"# Requests: {report['request_count']}\n\n"
            "## Stage breakdown\n"
            + format_table(
                report["stage_breakdown"],
                ["stage", "interval", "count", "total_ms", "avg_ms", "p95_ms"],
            )
            + "\n## Hop breakdown\n"
            + format_table(
                report["hop_breakdown"],
                ["src", "dst", "kind", "count", "total_ms", "avg_ms", "p95_ms"],
            )
            + (gpu_section if gpu_rows else "")
        )

    if args.out == "-":
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    else:
        with open(args.out, "w", encoding="utf-8") as fp:
            fp.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
