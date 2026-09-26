# SPDX-License-Identifier: Apache-2.0
"""Persist realtime ASR observations and replay protocol and timing checks offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from benchmarks.realtime_asr.client import SessionTrace
from benchmarks.realtime_asr.metrics import check_invariants, latency_metrics


class TraceArtifact(BaseModel):
    """Versioned observations; input PCM is stored beside the JSON by its hash."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1]
    sample_id: str
    config: dict[str, Any]
    source: dict[str, Any]
    input_pcm_sha256: str
    trace: SessionTrace


def source_fingerprint() -> dict[str, Any]:
    """Identify the actual recorder and grader files, including uncommitted edits."""
    root = Path(__file__).resolve().parents[2]
    paths = (
        "benchmarks/realtime_asr/client.py",
        "benchmarks/realtime_asr/metrics.py",
        "benchmarks/realtime_asr/replay.py",
        "benchmarks/eval/benchmark_asr_realtime.py",
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
        },
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("websockets", "pydantic")},
        "clock": "client time.perf_counter; seconds; one process clock domain",
        "server_revision": None,
    }


def save_trace(path: Path, artifact: TraceArtifact) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(artifact.model_dump_json(indent=2))
        handle.write("\n")


def load_trace(path: Path) -> TraceArtifact:
    return TraceArtifact.model_validate_json(path.read_text(encoding="utf-8"))


def replay_trace(artifact: TraceArtifact) -> dict[str, Any]:
    """Recompute the verdict and client timing without a server, model, or WER."""
    violations = check_invariants(artifact.trace)
    return {
        "sample_id": artifact.sample_id,
        "verdict": "fail" if violations else "pass",
        "violations": violations,
        "metrics": latency_metrics(artifact.trace),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Saved session trace JSON.")
    args = parser.parse_args()
    artifact = load_trace(args.trace)
    result = {
        **replay_trace(artifact),
        "recorded_source": artifact.source,
        "replay_source": source_fingerprint(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(1 if result["violations"] else 0)


if __name__ == "__main__":
    main()
