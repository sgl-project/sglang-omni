#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Persist immutable, best-effort Omni CI observation records."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
OBSERVATION_ROOT = Path("/data/omni-ci/observations")
RETENTION_DAYS = 30
_RESULT_NAMES = {"results.json", "benchmark.wall_time.json", "manifest.json"}
_CONTENTION_PREFIX = "[cpuset-contention]"
_CONTENTION_STATS_RE = re.compile(
    r"cpuset=(?P<cpuset>\S+) windows=(?P<windows>\d+) "
    r"foreign-cores mean=(?P<mean>\d+(?:\.\d+)?) "
    r"max=(?P<peak>\d+(?:\.\d+)?) errors=(?P<errors>\d+)"
)
_CONTENTION_EMPTY_RE = re.compile(
    r"cpuset=(?P<cpuset>\S+) no completed sample windows " r"\(errors=(?P<errors>\d+)\)"
)
_BENCHMARK_CONTAINER_KEYS = ("summary", "speed", "metrics", "overall")
_BENCHMARK_METRICS = {
    "throughput": (
        ("throughput", None),
        ("throughput_qps", None),
        ("throughput_samples_per_s", None),
        ("achieved_rps", None),
    ),
    "latency_mean": (
        ("latency_mean", None),
        ("latency_mean_s", None),
        ("latency_s", "mean"),
    ),
    "latency_p95": (
        ("latency_p95", None),
        ("latency_p95_s", None),
        ("latency_s", "p95"),
    ),
    "rtf": (("rtf_mean", None), ("rtf", "mean"), ("rtf", None)),
    "wer": (("wer_corpus", None), ("corpus_wer", None), ("wer", None)),
    "success": (
        ("success", None),
        ("succeeded", None),
        ("successful_request_count", None),
        ("completed_requests", None),
        ("successful_samples", None),
        ("evaluated", None),
        ("passed", None),
    ),
    "failed": (
        ("failed", None),
        ("failed_requests", None),
        ("failed_samples", None),
    ),
}


def _slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or fallback


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _duration_seconds(started_at: str, finished_at: str) -> float | None:
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, round((finish - start).total_seconds(), 3))


def _metadata(stage_label: str) -> dict[str, Any]:
    env = os.environ
    return {
        "run": {
            "repository": env.get("GITHUB_REPOSITORY"),
            "workflow": env.get("GITHUB_WORKFLOW"),
            "event_name": env.get("GITHUB_EVENT_NAME"),
            "run_id": env.get("GITHUB_RUN_ID"),
            "run_number": env.get("GITHUB_RUN_NUMBER"),
            "run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
            "server_url": env.get("GITHUB_SERVER_URL"),
            "actor": env.get("GITHUB_ACTOR"),
            "workflow_ref": env.get("GITHUB_WORKFLOW_REF"),
        },
        "job": {
            "id": env.get("GITHUB_JOB"),
            "runner_name": env.get("RUNNER_NAME"),
            "runner_os": env.get("RUNNER_OS"),
            "runner_arch": env.get("RUNNER_ARCH"),
        },
        "stage": {"label": stage_label},
        "commit": {
            "sha": env.get("GITHUB_SHA"),
            "ref": env.get("GITHUB_REF"),
            "head_ref": env.get("GITHUB_HEAD_REF"),
            "base_ref": env.get("GITHUB_BASE_REF"),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "cpu_set": env.get("OMNI_CI_CPUSET"),
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_visible_devices": env.get("NVIDIA_VISIBLE_DEVICES"),
            "python": env.get("OMNI_CI_PYTHON"),
        },
    }


def _record_directory(stage_label: str) -> Path | None:
    env = os.environ
    run_id = env.get("GITHUB_RUN_ID")
    if env.get("GITHUB_ACTIONS") != "true" or not run_id:
        return None
    run_id = _slug(run_id, "unknown-run")
    repository = _slug(env.get("GITHUB_REPOSITORY", "unknown"), "unknown")
    run_attempt = _slug(env.get("GITHUB_RUN_ATTEMPT", "1"), "1")
    job = _slug(env.get("GITHUB_JOB", "unknown-job"), "unknown-job")
    stage = _slug(stage_label, "unknown-stage")
    return (
        OBSERVATION_ROOT
        / f"v{SCHEMA_VERSION}"
        / repository
        / run_id
        / run_attempt
        / job
        / stage
    )


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _result_roots() -> list[Path]:
    roots = [Path("/tmp")]
    for name, value in os.environ.items():
        if name.endswith("_OUTPUT_ROOT") and value:
            roots.append(Path(value))
    return roots


def _is_result_file(name: str) -> bool:
    return name.endswith("_results.json") or name in _RESULT_NAMES


def _benchmark_containers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    seen: set[int] = set()

    def visit(container: dict[str, Any]) -> None:
        if id(container) in seen:
            return
        seen.add(id(container))
        containers.append(container)
        for key in _BENCHMARK_CONTAINER_KEYS:
            nested = container.get(key)
            if isinstance(nested, dict):
                visit(nested)

    visit(payload)
    return containers


def _benchmark_summary(payload: Any) -> dict[str, bool | int | float]:
    if not isinstance(payload, dict):
        return {}
    containers = _benchmark_containers(payload)
    summary: dict[str, bool | int | float] = {}
    for output_key, selectors in _BENCHMARK_METRICS.items():
        for source_key, statistic in selectors:
            found = False
            for container in containers:
                value = container.get(source_key)
                if statistic is not None and isinstance(value, dict):
                    value = value.get(statistic)
                if isinstance(value, (bool, int, float)):
                    summary[output_key] = value
                    found = True
                    break
            if found:
                break
    return summary


def _collect_results(started_epoch: float) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in _result_roots():
        if not root.is_dir():
            continue
        for directory, _, filenames in os.walk(root):
            for filename in filenames:
                if not _is_result_file(filename):
                    continue
                path = Path(directory) / filename
                try:
                    resolved = path.resolve()
                    if resolved in seen or path.stat().st_mtime < started_epoch:
                        continue
                    with path.open(encoding="utf-8") as result_file:
                        payload = json.load(result_file)
                except (OSError, ValueError):
                    continue
                seen.add(resolved)
                summary = _benchmark_summary(payload)
                if summary:
                    results.append({"path": str(path), **summary})
    return sorted(results, key=lambda item: item["path"])


def _compact_cpuset(spec: str) -> str:
    try:
        cpus = sorted({int(cpu) for cpu in spec.split(",")})
    except ValueError:
        return spec
    ranges: list[str] = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _collect_contention(log_file: str) -> dict[str, Any] | None:
    if not log_file:
        return None
    try:
        with open(log_file, encoding="utf-8", errors="replace") as log:
            lines = [
                line.split(_CONTENTION_PREFIX, 1)[1].strip()
                for line in log
                if _CONTENTION_PREFIX in line
            ]
    except OSError:
        return None

    contention: dict[str, Any] | None = None
    warnings: list[str] = []
    for line in lines:
        stats = _CONTENTION_STATS_RE.fullmatch(line)
        if stats:
            contention = {
                "cpuset": _compact_cpuset(stats.group("cpuset")),
                "windows": int(stats.group("windows")),
                "foreign_cores_mean": float(stats.group("mean")),
                "foreign_cores_peak": float(stats.group("peak")),
                "errors": int(stats.group("errors")),
            }
            continue
        empty = _CONTENTION_EMPTY_RE.fullmatch(line)
        if empty:
            contention = {
                "cpuset": _compact_cpuset(empty.group("cpuset")),
                "windows": 0,
                "foreign_cores_mean": None,
                "foreign_cores_peak": None,
                "errors": int(empty.group("errors")),
            }
            continue
        if line.startswith("WARNING: "):
            warning = line.removeprefix("WARNING: ").partition(";")[0]
            if warning not in warnings:
                warnings.append(warning)

    if contention is None and not warnings:
        return None
    if contention is None:
        contention = {
            "cpuset": None,
            "windows": None,
            "foreign_cores_mean": None,
            "foreign_cores_peak": None,
            "errors": None,
        }
    contention["warnings"] = warnings
    return contention


def _attempt(args: argparse.Namespace) -> None:
    directory = _record_directory(args.stage_label)
    if directory is None:
        return
    payload = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "stage_attempt",
        "recorded_at": _utc_now(),
        **_metadata(args.stage_label),
        "attempt": {
            "number": args.attempt,
            "max_attempts": args.max_attempts,
            "started_at": args.started_at,
            "finished_at": args.finished_at,
            "duration_seconds": _duration_seconds(args.started_at, args.finished_at),
            "exit_code": args.exit_code,
        },
        "cpu_contention": _collect_contention(args.log_file),
        "benchmark_results": _collect_results(args.started_epoch),
    }
    _write_immutable(directory / f"attempt-{args.attempt}.json", payload)


def _wrapper_terminal(args: argparse.Namespace) -> None:
    directory = _record_directory(args.stage_label)
    if directory is None:
        return
    payload = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "wrapper_terminal",
        "recorded_at": _utc_now(),
        **_metadata(args.stage_label),
        "wrapper": {
            "started_at": args.started_at,
            "finished_at": args.finished_at,
            "duration_seconds": _duration_seconds(args.started_at, args.finished_at),
            "exit_code": args.exit_code,
            "attempts_completed": args.attempts_completed,
        },
    }
    _write_immutable(directory / "wrapper-terminal.json", payload)


def _current_attempt_records(stage_label: str) -> list[dict[str, Any]]:
    directory = _record_directory(stage_label)
    if directory is None:
        return []
    records = []
    for path in directory.glob("attempt-*.json"):
        try:
            with path.open(encoding="utf-8") as record_file:
                record = json.load(record_file)
        except (OSError, json.JSONDecodeError):
            continue
        record["_path"] = str(path)
        records.append(record)
    return sorted(
        records, key=lambda record: record.get("attempt", {}).get("number", 0)
    )


def _append_summary(stage_label: str, records: list[dict[str, Any]]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [f"### Omni CI observation: {stage_label}", ""]
    if not records:
        lines.append("No completed retry attempts were observed.")
    else:
        lines.extend(
            [
                "| Attempt | Exit code | Duration | Benchmark summaries |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for record in records:
            attempt = record.get("attempt", {})
            duration = attempt.get("duration_seconds")
            duration_text = (
                f"{duration:.1f}s" if isinstance(duration, (int, float)) else "unknown"
            )
            lines.append(
                f"| {attempt.get('number', 'unknown')} | "
                f"{attempt.get('exit_code', 'unknown')} | {duration_text} | "
                f"{len(record.get('benchmark_results', []))} |"
            )
        lines.extend(
            ["", f"Records: `{records[0]['_path']}` and sibling attempt files."]
        )
    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write("\n".join(lines) + "\n")


def _garbage_collect() -> None:
    if not OBSERVATION_ROOT.is_dir():
        return
    cutoff = time.time() - RETENTION_DAYS * 24 * 60 * 60
    for path in OBSERVATION_ROOT.rglob("*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue
    directories = sorted(
        (path for path in OBSERVATION_ROOT.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            if directory.stat().st_mtime < cutoff:
                directory.rmdir()
        except OSError:
            continue


def _post_stage(args: argparse.Namespace) -> None:
    records = _current_attempt_records(args.stage_label)
    _append_summary(args.display_label, records)
    _garbage_collect()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    attempt = subparsers.add_parser("attempt")
    attempt.add_argument("--stage-label", required=True)
    attempt.add_argument("--attempt", type=int, required=True)
    attempt.add_argument("--max-attempts", type=int, required=True)
    attempt.add_argument("--started-at", required=True)
    attempt.add_argument("--started-epoch", type=float, required=True)
    attempt.add_argument("--finished-at", required=True)
    attempt.add_argument("--exit-code", type=int, required=True)
    attempt.add_argument("--log-file", default="")
    attempt.set_defaults(handler=_attempt)

    terminal = subparsers.add_parser("wrapper-terminal")
    terminal.add_argument("--stage-label", required=True)
    terminal.add_argument("--started-at", required=True)
    terminal.add_argument("--finished-at", required=True)
    terminal.add_argument("--exit-code", type=int, required=True)
    terminal.add_argument("--attempts-completed", type=int, required=True)
    terminal.set_defaults(handler=_wrapper_terminal)

    post_stage = subparsers.add_parser("post-stage")
    post_stage.add_argument("--stage-label", required=True)
    post_stage.add_argument("--display-label", required=True)
    post_stage.set_defaults(handler=_post_stage)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
