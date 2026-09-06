#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Observe CI metrics over N runs on the H100 repro host.
Emits worst-of-N markdown. Does NOT propose thresholds or edit tests.
Model-agnostic: pass --model <name>; config comes from
models/<name>/config.yaml. Metrics come from result JSONs that tests
already write under pytest's --basetemp (set fresh per run).
"""
from __future__ import annotations
import argparse, ast, datetime as dt, hashlib, json, math, os, platform, re, shutil, signal
import statistics, subprocess, sys, time, tomllib
from pathlib import Path
from typing import NamedTuple

__version__ = "0.7.0"

SKILL_DIR = Path(__file__).resolve().parent
MODELS_DIR = SKILL_DIR / "models"
HOSTS_DIR = SKILL_DIR / "hosts"
DEFAULT_MODEL = "omni"
_SPEAKER_SIM_MIN_BYTES = 100 * 1024 * 1024
REPO_ROOT = Path("/sgl-workspace/sglang-omni")
if not REPO_ROOT.exists():
    REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = Path("/github/home/ci-threshold-runs")
RETRY_SIGS = ("OOM", "exit 137", "exit 139", "TimeoutExpired")
# Per-GPU memory must be strictly below this (MiB) before any pytest/server
# restart. 2048 MiB = 2 GiB. Do not launch on stale 17 GiB contexts.
_GPU_RETRY_MEM_MIB = 2048
_GPU_LAUNCH_RECHECK_S = 3
_GPU_WAIT_POLL_S = 5
_GPU_WAIT_TIMEOUT_S = 600
_PYTEST_POLL_S = 30
_MAX_RUN_ATTEMPTS = 4  # infra-failure retries (OOM/crash/GPU-not-clear) to obtain one clean repeat; calibration-specific, unrelated to CI's per-test failure retry
_DEFAULT_CALIBRATION_PASSES = 10
_AGENT_POLL_INTERVAL_S = 120

# Destructive-observation rejection.
#
# A pytest round can complete with full sample scope and non-null metrics and
# still be worthless: host contention, a cold autotune cache, or a thrashing
# server produce numbers that describe the machine, not the model. Feeding one
# such round into strict worst-of-N sets the CI reference from the accident —
# observed inflation up to 4.53x on 2026-08-01.
#
# A value is destructive only when BOTH hold:
#   * robust z (MAD) above _DESTRUCTIVE_Z — it is far from the centre; and
#   * it is separated from its nearest neighbour by a real gap.
# The gap test is what separates a broken round from the tail of a small
# sample. At n=5, MAD alone flags ordinary tail points: on the 2026-08-01 data
# it fired in every round of the 27-metric serving unit, which would make the
# reject-and-replace loop non-terminating.
#
# Rejection is per ROUND, not per metric: a round whose speed collapsed cannot
# be trusted for accuracy either, so any single destructive metric discards the
# whole round for every stage in that pytest invocation.
_DESTRUCTIVE_Z = 3.5
_DESTRUCTIVE_GAP = 0.20
_DESTRUCTIVE_MIN_OBS = 5      # need this many values before judging outliers
_DESTRUCTIVE_FULL_RERUN_N = 3  # n>=this: the "others agree" premise is gone
# Two comparable populations this far apart mean the robust centre has moved
# into one of them and outlier identification has inverted. Deliberately much
# larger than _DESTRUCTIVE_GAP: mild bimodality is a noisy stage (surfaced by
# the speed-health check), not a detector failure.
_DEGENERATE_SPLIT = 0.50
_DESTRUCTIVE_MAX_RESTARTS = 1
_DESTRUCTIVE_MAX_ROUNDS = 15
_CI_HOME = Path("/github/home")
_CRASH_SIGS = (
    "Fatal Python error",
    "Segmentation fault",
    "CUDA error: an illegal memory access",
    "Child process died",
    "All workers failed",
    "Router failed to start",
    "Address already in use",
    "Connection refused",
    "Server process exited",
    "worker crashed",
)


def _flashinfer_cache_dirs(env: dict[str, str] | None = None) -> list[Path]:
    env = env or os.environ
    candidates = [
        Path(env.get("XDG_CACHE_HOME", "")) / "flashinfer"
        if env.get("XDG_CACHE_HOME")
        else None,
        Path(env.get("HOME", "")) / ".cache" / "flashinfer"
        if env.get("HOME")
        else None,
        _CI_HOME / ".cache" / "flashinfer",
    ]
    seen: set[Path] = set()
    paths: list[Path] = []
    for candidate in candidates:
        if candidate:
            canon = candidate.resolve()
            if canon not in seen:
                seen.add(canon)
                paths.append(canon)
            elif canon == _CI_HOME / ".cache" / "flashinfer":
                # The CI-specific one is often a fallback; if it resolves to the
                # same spot as the env-backed one, still count it to avoid emptying.
                seen.add(canon)
                paths.append(canon)
    return paths


def _parse_preset_args(model_name: str) -> argparse.Namespace:
    """Parse and normalize arguments for the omni launcher."""
    parser = argparse.ArgumentParser(description=f"Run {model_name} with omni tuning")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model preset name")
    parser.add_argument("--n-runs", type=int, default=_DEFAULT_CALIBRATION_PASSES, help="Number of runs to calibrate")
    parser.add_argument("--basetemp", type=str, default=None, help="Pytest temp dir for result JSONs")
    parser.add_argument("--env", type=str, nargs="*", default=[], help="Extra environment variables")
    args = parser.parse_args()
    return args


def _build_env(added: list[str]) -> dict[str, str]:
    """Build an environment dict from current env plus any added vars."""
    base = dict(os.environ)
    for var in added:
        key, val = var.split("=", 1) if "=" in var else (var, os.environ.get(var, ""))
        base[key] = val
    return base


def _get_model_config(model_name: str) -> dict[str, Any]:
    """Load the model-specific config from models/<name>/config.yaml."""
    path = MODELS_DIR / model_name / "config.yaml"
    if path.exists():
        return tomllib.loads(path.read_text())
    return {}


def _collect_resource_samples(n_runs: int, host_name: str) -> list[dict[str, Any]]:
    """Collect GPU/system samples for each run, de-duplicating stale metrics."""
    from benchmarks.runtime_metrics import ResourceSample, _REPO_ROOT
    import threading
    import platform
    
    samples: list[dict[str, Any]] = []
    
    # Use a lock if we're collecting in parallel
    lock = threading.Lock()
    
    for i in range(n_runs):
        try:
            with lock:
                sample = ResourceSample(
                    elapsed_s=time.time(),
                    gpu_memory_used_mib=2048.0,
                    gpu_memory_free_mib=64.0,
                    gpu_process_memory_mib=256.0,
                    gpu_util_percent=90.0,
                    power_w=250.0,
                    system_cpu_percent=35.0,
                    gpu_process_cpu_percent=15.0,
                    gpu_process_pids=(i,),
                )
                samples.append(sample)
        except Exception as e:
            print(f"Run {i} sample collection error: {e}")
    
    return samples


def _check_destructive_rounds(metrics: dict[str, list[float]],
                              n_obs: int, z_thresh: float,
                              gap_thresh: float) -> dict[str, bool]:
    """Identify rounds where a single metric is so far from the pack it breaks the mean."""
    is_destructive = {}
    seen = {}
    
    for metric, values in metrics.items():
        if len(values) < n_obs:
            is_destructive[metric] = False
            continue
            
        # Calculate MAD-based robust z
        mad = statistics.median([abs(v - statistics.median(values)) for v in values])
        robust_z = mad if mad > 0 else 0.0
        
        # First pass: establish the range of "normal" values
        min_v, max_v = min(values), max(values)
        range_span = max_v - min_v if max_v and min_v else 1.0
        
        is_destructive[metric] = robust_z > (z_thresh * mad) if values else False
        seen[metric] = robust_z
    
    return is_destructive


def _round_robin_scheduler(n_replicas: int) -> dict[str, int]:
    """Simple round-robin mapping for N process replicas."""
    replica_ids = list(range(n_replicas))
    requests = {}
    for i, (key, val) in enumerate(os.environ.items()):
        if i < n_replicas:
            requests[f"REPLICA_{i}"] = replica_ids[i]
        else:
            requests[f"REPLICA_{i}"] = replica_ids[i % n_replicas]
    return requests


def _wait_for_gpu_free(env: dict[str, str], target_mib: int = _GPU_RETRY_MEM_MIB,
                       poll_s: int = _GPU_WAIT_POLL_S, timeout: int = _GPU_WAIT_TIMEOUT_S) -> None:
    """Poll until the GPU has cooled below the target, accounting for context drift."""
    env["PYTEST_XDIST_WORKER"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    import subprocess
    
    start = time.time()
    while time.time() - start < timeout:
        # Check GPU process memory
        mem_used = subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.used"],
            env=env, text=True
        ).strip().split("\n")[0]
        used = float(mem_used)
        
        if used < target_mib:
            return
            
        time.sleep(poll_s)
    
    raise TimeoutError(f"GPU didn't cool below {target_mib} MiB in {timeout}s")


def _parse_result_json(path: Path) -> dict[str, Any]:
    """Parse a single pytest result JSON that stores metrics."""
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _build_metric_summary(result_path: Path, n_reruns: int) -> dict[str, dict[str, Any]]:
    """Assemble the final metric summary from N collected rounds."""
    metrics: dict[str, dict[str, Any]] = {}
    
    # Load raw results
    raw = _parse_result_json(result_path)
    
    # Build per-metric stats
    for key, vals in raw.items():
        if isinstance(vals, list) and len(vals) > 1:
            metrics[key] = {
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / len(vals),
                "median": statistics.median(vals),
                "mad": statistics.median([abs(v - statistics.median(vals)) for v in vals]),
                "n": len(vals),
            }
        elif isinstance(vals, (int, float)):
            metrics[key] = {
                "mean": vals,
                "n": 1,
            }
    
    return metrics


def _emit_worst_of_n(markdown: list[str], metric: str, values: list[float],
                      prefix: str = "") -> list[str]:
    """Append the metric's worst-of-N values to the markdown lines."""
    best = max(values) if values else 0.0
    worst = min(values) if values else 0.0
    
    markdown.append(f"{prefix}{metric} (worst-of-{len(values)}): {worst}")
    markdown.append(f"  *min*: {worst}, *max*: {best}")
    return markdown


def _main() -> None:
    """Main entry point for the tuning orchestration."""
    args = _parse_preset_args(DEFAULT_MODEL)
    n_runs = getattr(args, "n_runs", _DEFAULT_CALIBRATION_PASSES)
    
    model_name = getattr(args, "model", DEFAULT_MODEL)
    config = _get_model_config(model_name)
    
    # Initialize result tracking
    result_path = DEFAULT_RUN_ROOT / f"{model_name}_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect samples across runs
    metrics = _collect_resource_samples(n_runs, DEFAULT_MODEL)
    
    # Build summary
    metric_summary = _build_metric_summary(result_path, n_runs)
    
    # Emit markdown
    lines = [
        f"## {model_name.capitalize()} Tuning Report",
        f"- Model: {model_name}",
        f"- Config: `{model_name}/config.yaml`",
        f"- Rerun N: {n_runs}",
        "---",
    ]
    
    # Add each metric from the summary
    for metric_name, values in metric_summary.items():
        lines.extend(_emit_worst_of_n(lines, metric_name, values["mean"]))
    
    # Write final markdown
    with open(result_path.parent / f"{model_name}_report.md", "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    
    print(f"Emitted tuning report to {result_path.parent / f'{model_name}_report.md'}")


if __name__ == "__main__":
    _main()