# SPDX-License-Identifier: Apache-2.0
"""Record one offload run and compare it with a baseline.

D1 of the memory-offload roadmap: a run keeps the hardware, checkpoint,
concurrency and workload it was measured under, the peak and steady memory it
reached, and the latency the client saw. Comparing two runs refuses the pair
unless every one of those matches, so a difference in the report is a
difference the change made.

Two fields stay absent until their producer exists: `restore_s` needs
`release_memory_occupation` / `resume_memory_occupation` (#2282), and
`moe_cache_miss_bytes` needs the counters a MoE offload introduces. An absent
field is reported as skipped rather than compared against nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1
# The provenance a pair must share before its metrics may be compared.
COMPARABLE_PROVENANCE = ("gpu_name", "model", "checkpoint", "concurrency", "workload")


class Provenance(BaseModel):
    """What a run was measured on, so two runs can be told apart from their numbers."""

    model_config = ConfigDict(extra="forbid", strict=True)

    git_commit: str = Field(min_length=1)
    gpu_name: str = Field(min_length=1)
    gpu_memory_mib: float = Field(gt=0)
    driver_version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    checkpoint: str = Field(min_length=1)
    concurrency: int = Field(ge=1)
    workload: str = Field(min_length=1)
    recorded_at: str = Field(min_length=1)


class MemoryRecord(BaseModel):
    """GPU memory held while the workload ran, in MiB."""

    model_config = ConfigDict(extra="forbid", strict=True)

    peak_mib: float = Field(ge=0)
    steady_mib: float = Field(ge=0)
    after_idle_mib: float = Field(ge=0)
    process_peak_mib: float | None = Field(default=None, ge=0)


class LatencyRecord(BaseModel):
    """What the client saw of the served replies, in seconds unless named otherwise."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ttfa_p50_s: float = Field(ge=0)
    ttfa_p95_s: float = Field(ge=0)
    inter_chunk_p95_s: float = Field(ge=0)
    inter_chunk_p99_s: float = Field(ge=0)
    rtf_p95: float = Field(ge=0)
    throughput_qps: float = Field(gt=0)


class OffloadRun(BaseModel):
    """One measured run: provenance, memory, latency, and what it could not measure."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = SCHEMA_VERSION
    label: str = Field(min_length=1)
    provenance: Provenance
    memory: MemoryRecord
    latency: LatencyRecord
    startup_s: float = Field(ge=0)
    restore_s: float | None = Field(default=None, ge=0)
    moe_cache_miss_bytes: int | None = Field(default=None, ge=0)


class MetricDelta(BaseModel):
    """One metric before and after, with the direction that counts as better."""

    model_config = ConfigDict(extra="forbid", strict=True)

    metric: str
    baseline: float
    candidate: float
    ratio: float
    lower_is_better: bool
    is_improvement: bool


class Comparison(BaseModel):
    """The deltas two compatible runs differ by, plus the fields neither measured."""

    model_config = ConfigDict(extra="forbid", strict=True)

    baseline: str
    candidate: str
    deltas: list[MetricDelta]
    skipped: list[str]


class CompareError(ValueError):
    """Two runs that cannot be compared without pretending a condition matched."""


LOWER_IS_BETTER = {
    "startup_s": True,
    "restore_s": True,
    "moe_cache_miss_bytes": True,
    "memory.peak_mib": True,
    "memory.steady_mib": True,
    "memory.after_idle_mib": True,
    "memory.process_peak_mib": True,
    "latency.ttfa_p50_s": True,
    "latency.ttfa_p95_s": True,
    "latency.inter_chunk_p95_s": True,
    "latency.inter_chunk_p99_s": True,
    "latency.rtf_p95": True,
    "latency.throughput_qps": False,
}


def write_run(run: OffloadRun, path: str | Path) -> Path:
    """Write one run as JSON, creating its directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(run.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_run(path: str | Path) -> OffloadRun:
    """Read one run back; an unknown or newer schema is an error rather than a guess."""
    return OffloadRun.model_validate_json(Path(path).read_text(encoding="utf-8"))


def metric_value(run: OffloadRun, metric: str) -> float | None:
    """One metric of a run by its dotted name, or None when the run did not measure it."""
    if metric == "startup_s":
        return run.startup_s
    elif metric == "restore_s":
        return run.restore_s
    elif metric == "moe_cache_miss_bytes":
        value = run.moe_cache_miss_bytes
        return None if value is None else float(value)
    elif metric.startswith("memory."):
        return float(getattr(run.memory, metric.removeprefix("memory.")))
    elif metric.startswith("latency."):
        return float(getattr(run.latency, metric.removeprefix("latency.")))
    else:
        raise KeyError(f"unknown offload metric {metric!r}")


def differing_provenance(baseline: OffloadRun, candidate: OffloadRun) -> list[str]:
    """The provenance fields a pair disagrees on, so a caller can say which one broke the comparison."""
    differences: list[str] = []
    for name in COMPARABLE_PROVENANCE:
        if getattr(baseline.provenance, name) != getattr(candidate.provenance, name):
            differences.append(
                f"{name}: {getattr(baseline.provenance, name)!r} != {getattr(candidate.provenance, name)!r}"
            )
        else:
            pass
    return differences


def compare_runs(baseline: OffloadRun, candidate: OffloadRun) -> Comparison:
    """Compare two runs that share their provenance; refuse the pair when they do not."""
    differences = differing_provenance(baseline, candidate)
    if differences:
        raise CompareError(
            "runs were measured under different conditions: " + "; ".join(differences)
        )
    else:
        pass
    deltas: list[MetricDelta] = []
    skipped: list[str] = []
    for metric, lower_is_better in LOWER_IS_BETTER.items():
        before = metric_value(baseline, metric)
        after = metric_value(candidate, metric)
        if before is None or after is None:
            skipped.append(metric)
            continue
        elif before == 0:
            # A ratio against zero says nothing; the pair of values still does.
            skipped.append(metric)
            continue
        else:
            pass
        ratio = after / before
        deltas.append(
            MetricDelta(
                metric=metric,
                baseline=before,
                candidate=after,
                ratio=ratio,
                lower_is_better=lower_is_better,
                is_improvement=ratio < 1.0 if lower_is_better else ratio > 1.0,
            )
        )
    return Comparison(
        baseline=baseline.label,
        candidate=candidate.label,
        deltas=deltas,
        skipped=skipped,
    )


FORMATTED_WHOLE = frozenset({"moe_cache_miss_bytes"})


def format_value(metric: str, value: float) -> str:
    """One metric's value: byte counters are whole numbers, timings and ratios keep three decimals."""
    if metric in FORMATTED_WHOLE:
        return f"{value:.0f}"
    else:
        return f"{value:.3f}"


def format_comparison(comparison: Comparison) -> str:
    """The comparison as the lines a maintainer reads in a PR."""
    lines = [f"{comparison.candidate} against {comparison.baseline}:"]
    for delta in comparison.deltas:
        verdict = "better" if delta.is_improvement else "worse"
        lines.append(
            f"  {delta.metric}: {format_value(delta.metric, delta.baseline)}"
            f" -> {format_value(delta.metric, delta.candidate)} "
            f"({delta.ratio:.2f}x, {verdict})"
        )
    if comparison.skipped:
        lines.append("  not compared: " + ", ".join(comparison.skipped))
    else:
        pass
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Compare two recorded runs, exiting non-zero when the pair cannot be compared."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    compare = subcommands.add_parser(
        "compare", help="report the deltas between two runs"
    )
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        comparison = compare_runs(
            load_run(arguments.baseline), load_run(arguments.candidate)
        )
    except CompareError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(format_comparison(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
