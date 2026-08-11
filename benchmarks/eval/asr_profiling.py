# SPDX-License-Identifier: Apache-2.0
"""Bottleneck-profiling helpers for the ASR concurrency benchmark (issue #1324).

Small, dependency-light building blocks used by ``benchmark_asr_seedtts``:

- request-profile control against the serve HTTP surface
  (``/start_request_profile`` / ``/stop_request_profile``);
- stage/hop breakdown assembly from profiler event JSONL via
  ``sglang_omni.profiler.views``;
- background host-CPU / GPU utilization sampling around a benchmark pass;
- environment fingerprinting so results stay attributable to an exact
  code + dependency + hardware state.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

import requests

_NO_PROXIES = {"http": None, "https": None}
_PROFILE_TIMEOUT_S = 30


def start_request_profile(base_url: str, run_id: str, event_dir: str) -> dict:
    """Start request-level (JSONL) event profiling on one serve process."""
    response = requests.post(
        f"{base_url.rstrip('/')}/start_request_profile",
        json={"run_id": run_id, "event_dir": event_dir},
        timeout=_PROFILE_TIMEOUT_S,
        proxies=_NO_PROXIES,
    )
    response.raise_for_status()
    return response.json()


def stop_request_profile(base_url: str, run_id: str | None = None) -> dict:
    """Stop request-level event profiling (run_id=None stops whatever runs)."""
    response = requests.post(
        f"{base_url.rstrip('/')}/stop_request_profile",
        json={"run_id": run_id},
        timeout=_PROFILE_TIMEOUT_S,
        proxies=_NO_PROXIES,
    )
    response.raise_for_status()
    return response.json()


def build_stage_breakdown(event_dir: str, *, include_timelines: bool = False) -> dict:
    """Summarize profiler event JSONL into stage and hop breakdowns.

    Requires the benchmark to run on the same host as the server, because
    ``event_dir`` is a server-side path. Timelines are dropped by default to
    keep result JSON small; breakdown rows carry count/total/avg/p50/p95/max.
    """
    from sglang_omni.profiler.views import build_report

    report = build_report(event_dir)
    if not include_timelines:
        report.pop("timelines", None)
    return report


async def run_profiled_pass(
    *,
    run_id: str,
    event_dir: str,
    profile_urls: Sequence[str],
    run_pass: Callable[[], Awaitable[dict[str, Any]]],
    log_prefix: str,
    error_stream: TextIO | None = None,
) -> dict[str, Any] | None:
    """Run one pass inside a scoped request-profiling lifecycle."""
    started: list[str] = []
    try:
        for url in profile_urls:
            start_request_profile(url, run_id, event_dir)
            started.append(url)
    except requests.RequestException as exc:
        for url in started:
            try:
                stop_request_profile(url, run_id)
            except requests.RequestException as stop_exc:
                print(
                    f"{log_prefix} failed to stop profiling on {url}: {stop_exc}",
                    file=error_stream,
                )
        print(
            f"{log_prefix} profiling unavailable, skipping: {exc}",
            file=error_stream,
        )
        return None

    try:
        pass_metrics = await run_pass()
    finally:
        for url in started:
            try:
                stop_request_profile(url, run_id)
            except requests.RequestException as stop_exc:
                # note (Xinyu): Preserve pass results when only profiler teardown fails.
                print(
                    f"{log_prefix} failed to stop profiling on {url}: {stop_exc}",
                    file=error_stream,
                )

    report = build_stage_breakdown(event_dir)
    return {
        "run_id": run_id,
        "event_dir": event_dir,
        "pass_metrics": pass_metrics,
        "request_count": report.get("request_count"),
        "stage_breakdown": report.get("stage_breakdown"),
        "hop_breakdown": report.get("hop_breakdown"),
    }


@dataclass
class UtilizationSummary:
    samples: int
    cpu_percent_mean: float | None
    cpu_percent_max: float | None
    load_avg_1m_max: float | None
    gpu: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "cpu_percent_mean": self.cpu_percent_mean,
            "cpu_percent_max": self.cpu_percent_max,
            "load_avg_1m_max": self.load_avg_1m_max,
            "gpu": self.gpu,
        }


def _read_proc_stat() -> tuple[int, int] | None:
    """Return (busy, total) jiffies from /proc/stat, or None off-Linux."""
    try:
        with open("/proc/stat", encoding="utf-8") as handle:
            first = handle.readline().split()
    except OSError:
        return None
    if not first or first[0] != "cpu":
        return None
    values = [int(v) for v in first[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total - idle, total


def _query_gpu_utilization(gpu_ids: list[int]) -> dict[str, dict[str, float]]:
    try:
        raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    out: dict[str, dict[str, float]] = {}
    wanted = {str(i) for i in gpu_ids} if gpu_ids else None
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        index, util, mem = parts[0], parts[1], parts[2]
        if wanted is not None and index not in wanted:
            continue
        try:
            out[index] = {"util_percent": float(util), "memory_mib": float(mem)}
        except ValueError:
            continue
    return out


@dataclass
class UtilizationSampler:
    """Background sampler of host CPU and selected-GPU utilization.

    CPU usage comes from /proc/stat deltas (Linux; None elsewhere) so the
    benchmark does not grow a psutil dependency. GPU stats come from a
    best-effort ``nvidia-smi`` query and are empty when unavailable.
    """

    gpu_ids: list[int] = field(default_factory=list)
    interval_s: float = 1.0
    _samples: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("UtilizationSampler already started")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="asr-bench-util-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> UtilizationSummary:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=max(self.interval_s * 3.0, 10.0))
            self._thread = None
        return self.summary()

    def _run(self) -> None:
        previous = _read_proc_stat()
        while not self._stop_event.wait(self.interval_s):
            sample: dict[str, Any] = {"t": time.time()}
            current = _read_proc_stat()
            if previous is not None and current is not None:
                busy = current[0] - previous[0]
                total = current[1] - previous[1]
                if total > 0:
                    sample["cpu_percent"] = 100.0 * busy / total
            previous = current
            try:
                sample["load_avg_1m"] = os.getloadavg()[0]
            except OSError:
                # note (luojiaxuan): load average is optional telemetry and
                # unavailable on some platforms (e.g. Windows); skip silently.
                pass
            gpu = _query_gpu_utilization(self.gpu_ids)
            if gpu:
                sample["gpu"] = gpu
            self._samples.append(sample)

    @property
    def samples(self) -> list[dict[str, Any]]:
        return list(self._samples)

    def summary(self) -> UtilizationSummary:
        cpu = [s["cpu_percent"] for s in self._samples if "cpu_percent" in s]
        loads = [s["load_avg_1m"] for s in self._samples if "load_avg_1m" in s]
        per_gpu: dict[str, dict[str, list[float]]] = {}
        for sample in self._samples:
            for index, stats in sample.get("gpu", {}).items():
                bucket = per_gpu.setdefault(index, {"util": [], "mem": []})
                bucket["util"].append(stats["util_percent"])
                bucket["mem"].append(stats["memory_mib"])
        gpu_summary = {
            index: {
                "util_percent_mean": sum(b["util"]) / len(b["util"]),
                "util_percent_max": max(b["util"]),
                "memory_mib_max": max(b["mem"]),
            }
            for index, b in per_gpu.items()
            if b["util"]
        }
        return UtilizationSummary(
            samples=len(self._samples),
            cpu_percent_mean=(sum(cpu) / len(cpu)) if cpu else None,
            cpu_percent_max=max(cpu) if cpu else None,
            load_avg_1m_max=max(loads) if loads else None,
            gpu=gpu_summary,
        )


def _run_command(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


_FINGERPRINT_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_ENDPOINT",
    "TORCHINDUCTOR_CACHE_DIR",
    "OMP_NUM_THREADS",
    "SGLANG_TORCH_PROFILER_DIR",
)


def collect_server_identity(base_url: str) -> dict:
    """Best-effort identity of the serving process under test.

    The client-side fingerprint describes the benchmark process; the server
    may run different code. This records what the server itself reports
    (currently its /v1/models listing) alongside the target URL.
    """
    identity: dict[str, Any] = {"url": base_url.rstrip("/")}
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/v1/models",
            timeout=10,
            proxies=_NO_PROXIES,
        )
        response.raise_for_status()
        payload = response.json()
        identity["models"] = [
            entry.get("id") for entry in payload.get("data", []) if entry.get("id")
        ]
    except (requests.RequestException, ValueError):
        identity["models"] = None
    return identity


def collect_environment_fingerprint(model_path: str | None = None) -> dict:
    """Capture code, dependency, and hardware identity of the client process.

    Every field is best-effort: a missing tool (git outside a checkout,
    nvidia-smi on CPU hosts) yields None rather than an exception, so the
    fingerprint never blocks a run. This describes the benchmark client;
    pair it with :func:`collect_server_identity` for the server side (they
    coincide only when client and server share one host and checkout).
    """
    pip_freeze = _run_command([sys.executable, "-m", "pip", "freeze"])
    fingerprint: dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git": {
            "sha": _run_command(["git", "rev-parse", "HEAD"]),
            "branch": _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_run_command(["git", "status", "--porcelain"]) or ""),
        },
        "packages": {
            name: _package_version(name)
            for name in ("torch", "sglang", "sglang-omni", "transformers")
        },
        "dependency_freeze_sha256": (
            hashlib.sha256(pip_freeze.encode("utf-8")).hexdigest()
            if pip_freeze
            else None
        ),
        "gpus": _run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "env": {key: os.environ.get(key) for key in _FINGERPRINT_ENV_KEYS},
    }
    if model_path:
        fingerprint["model_path"] = model_path
        fingerprint["model_revision"] = _cached_hf_revision(model_path)
    return fingerprint


def _cached_hf_revision(model_path: str) -> str | None:
    """Resolve the locally cached HF snapshot revision for a repo id."""
    if os.path.sep in model_path and os.path.isdir(model_path):
        return None
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    repo_dir = os.path.join(hf_home, "hub", "models--" + model_path.replace("/", "--"))
    ref_main = os.path.join(repo_dir, "refs", "main")
    try:
        with open(ref_main, encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
