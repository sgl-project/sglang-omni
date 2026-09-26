# SPDX-License-Identifier: Apache-2.0
"""Validate and replay one recorded native duplex run without a server."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import statistics
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from benchmarks.duplex.oracle import TraceRecord, evaluate_trace
from benchmarks.duplex.profiles import ProfileName


class InputArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate: Literal[16000]


class CaseArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    scenario: Literal["continuous"]
    trace_file: str


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    profile: ProfileName
    source: dict[str, Any]
    server: dict[str, Any]
    config: dict[str, Any]
    input: InputArtifact
    cases: list[CaseArtifact] = Field(min_length=1)


HARNESS_PACKAGES = ("websockets", "numpy", "pydantic", "soundfile")
SHA_PATTERN = r"[0-9a-f]{40}"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def add_server_identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--server-revision",
        required=True,
        help="Operator-supplied full server commit SHA",
    )
    parser.add_argument(
        "--model", required=True, help="Operator-supplied served model path or ID"
    )
    parser.add_argument(
        "--model-revision", help="Operator-supplied full model commit SHA, if known"
    )
    parser.add_argument(
        "--runtime", help="Operator-supplied server runtime, e.g. container digest"
    )


def served_models(url: str) -> dict[str, Any]:
    parts = urllib.parse.urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    models_url = urllib.parse.urlunsplit((scheme, parts.netloc, "/v1/models", "", ""))
    # Note (wenyao): A failed probe is recorded; it must not block the benchmark.
    try:
        with urllib.request.urlopen(models_url, timeout=5) as response:
            body = json.load(response)
        return {"url": models_url, "ids": [card["id"] for card in body["data"]]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"url": models_url, "error": f"{type(exc).__name__}: {exc}"}


def server_identity(
    url: str,
    *,
    revision: str,
    model: str,
    model_revision: str | None = None,
    runtime: str | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(SHA_PATTERN, revision):
        raise ValueError("server revision must be a full lowercase commit SHA")
    if not model:
        raise ValueError("model must be nonempty")
    if model_revision is not None and not re.fullmatch(SHA_PATTERN, model_revision):
        raise ValueError("model revision must be a full lowercase commit SHA")
    return {
        "revision": revision,
        "revision_source": "operator_supplied",
        "model": model,
        "model_revision": model_revision,
        "runtime": runtime,
        "identity_source": "operator_supplied",
        "served_models": served_models(url),
    }


def source_fingerprint() -> dict[str, Any]:
    """Identify the actual recorder and evaluator, including uncommitted edits."""
    root = Path(__file__).resolve().parents[2]
    paths = (
        "benchmarks/duplex/client.py",
        "benchmarks/duplex/oracle.py",
        "benchmarks/duplex/profiles.py",
        "benchmarks/duplex/artifacts.py",
        "benchmarks/eval/benchmark_duplex.py",
    )
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        revision = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        revision = None
    return {
        "harness_git_head": revision,
        "files_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in paths
            if (root / path).is_file()
        },
        "missing_files": [path for path in paths if not (root / path).is_file()],
        "python": platform.python_version(),
        "packages": {name: package_version(name) for name in HARNESS_PACKAGES},
        "clock": "client time.perf_counter; seconds; one process clock domain",
    }


def artifact_path(run_dir: Path, relative: str) -> Path:
    path = (run_dir / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(run_dir):
        raise ValueError(f"artifact path escapes run directory: {relative}")
    else:
        return path


def aggregate_metrics(cases: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    values: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        for name, value in case["metrics"].items():
            if type(value) in (int, float) and math.isfinite(value):
                values[name].append(value)
    return {
        name: {
            "n": len(items),
            "mean": statistics.fmean(items),
            "min": min(items),
            "max": max(items),
        }
        for name, items in sorted(values.items())
    }


def replay_run(run_dir: Path) -> dict[str, Any]:
    """Account for every selected case and recompute verdicts from raw records."""
    run_dir = run_dir.resolve()
    manifest = RunManifest.model_validate_json(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    input_errors: list[str] = []
    try:
        pcm = artifact_path(run_dir, manifest.input.file).read_bytes()
        if hashlib.sha256(pcm).hexdigest() != manifest.input.sha256:
            input_errors.append("input PCM SHA256 mismatch")
        if not pcm:
            input_errors.append("input PCM is empty")
        if len(pcm) % 2:
            input_errors.append("input PCM contains an incomplete 16-bit sample")
    except (OSError, ValueError) as exc:
        input_errors.append(f"input artifact unavailable: {exc}")

    id_counts = Counter(case.id for case in manifest.cases)
    trace_counts = Counter(
        (run_dir / case.trace_file).resolve() for case in manifest.cases
    )
    cases = []
    for case in manifest.cases:
        errors = list(input_errors)
        if id_counts[case.id] > 1:
            errors.append(f"duplicate case id: {case.id}")
        if trace_counts[(run_dir / case.trace_file).resolve()] > 1:
            errors.append(f"trace shared by selected cases: {case.trace_file}")
        records = []
        try:
            path = artifact_path(run_dir, case.trace_file)
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        records.append(
                            TraceRecord.model_validate_json(line).model_dump()
                        )
                    except ValidationError as exc:
                        reasons = "; ".join(
                            error["msg"] for error in exc.errors(include_input=False)
                        )
                        errors.append(f"trace line {line_number} invalid: {reasons}")
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"trace artifact unavailable: {exc}")
        sent_digest = hashlib.sha256()
        for record in records:
            event = record["event"]
            if (
                record["direction"] == "send"
                and event.get("type") == "input_audio_buffer.append"
            ):
                try:
                    sent_digest.update(
                        base64.b64decode(event.get("audio"), validate=True)
                    )
                except (TypeError, ValueError):
                    errors.append("sent audio contains invalid base64")
        if sent_digest.hexdigest() != manifest.input.sha256:
            errors.append("sent audio does not match persisted input PCM")
        verdict = evaluate_trace(
            records, scenario=case.scenario, profile=manifest.profile
        )
        cases.append(
            {
                "id": case.id,
                "scenario": case.scenario,
                "status": "fail" if errors else verdict["status"],
                "violations": [*verdict["violations"], *errors],
                "coverage": verdict["coverage"],
                "metrics": verdict["metrics"],
                "artifact_errors": errors,
            }
        )
    passed = [case for case in cases if case["status"] == "pass"]
    diagnostic = [case for case in cases if case["status"] != "pass"]
    return {
        "schema_version": 1,
        "profile": manifest.profile,
        "recorded_source": manifest.source,
        "replay_source": source_fingerprint(),
        "server": manifest.server,
        "config": manifest.config,
        "input": manifest.input.model_dump(),
        "cases": cases,
        "summary": {
            "selected": len(cases),
            "passed": len(passed),
            "failed": sum(case["status"] == "fail" for case in cases),
            "not_exercised": sum(case["status"] == "not_exercised" for case in cases),
            "qualified_metrics": aggregate_metrics(passed),
            "diagnostic_metrics": aggregate_metrics(diagnostic),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="Directory containing manifest.json."
    )
    args = parser.parse_args()
    try:
        result = replay_run(args.run_dir)
    except (OSError, UnicodeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1) from exc
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        summary = result["summary"]
        raise SystemExit(0 if summary["passed"] == summary["selected"] else 1)


if __name__ == "__main__":
    main()
