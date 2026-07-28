# SPDX-License-Identifier: Apache-2.0
"""Best-effort provenance and host-resource sampling for local benchmarks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResourceSample:
    elapsed_s: float
    gpu_memory_used_mib: float
    gpu_memory_free_mib: float
    gpu_process_memory_mib: float
    gpu_util_percent: float | None
    power_w: float | None
    system_cpu_percent: float | None
    gpu_process_cpu_percent: float | None
    gpu_process_pids: tuple[int, ...]


class ResourceMonitor:
    """Sample one local GPU and its compute processes in a background thread."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.2) -> None:
        if gpu_index < 0:
            raise ValueError("gpu_index must be >= 0")
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.samples: list[ResourceSample] = []
        self.error: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._pynvml: Any = None
        self._handle: Any = None
        self._psutil: Any = None
        self._processes: dict[int, Any] = {}

    def start(self) -> "ResourceMonitor":
        try:
            import psutil
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._psutil = psutil
            self._handle = _resolve_nvml_handle(pynvml, self.gpu_index)
            psutil.cpu_percent(interval=None)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._shutdown_nvml()
            return self

        self._started_at = time.perf_counter()
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name=f"benchmark-resource-monitor-gpu{self.gpu_index}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 5))
        if self._handle is not None:
            self._sample_once()
        self._shutdown_nvml()
        return summarize_resource_samples(
            self.samples,
            interval_s=self.interval_s,
            error=self.error,
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        try:
            pynvml = self._pynvml
            memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            process_memory_bytes = 0
            process_pids: set[int] = set()
            for process in _nvml_compute_processes(pynvml, self._handle):
                process_pids.add(int(process.pid))
                used = getattr(process, "usedGpuMemory", 0)
                if isinstance(used, int) and 0 <= used < 2**63:
                    process_memory_bytes += used

            utilization = _best_effort(
                lambda: float(pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
            )
            power_w = _best_effort(
                lambda: float(pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
            )
            system_cpu = _best_effort(
                lambda: float(self._psutil.cpu_percent(interval=None))
            )
            process_cpu = self._gpu_process_cpu_percent(process_pids)
            self.samples.append(
                ResourceSample(
                    elapsed_s=time.perf_counter() - self._started_at,
                    gpu_memory_used_mib=float(memory.used) / (1024**2),
                    gpu_memory_free_mib=float(memory.free) / (1024**2),
                    gpu_process_memory_mib=float(process_memory_bytes) / (1024**2),
                    gpu_util_percent=utilization,
                    power_w=power_w,
                    system_cpu_percent=system_cpu,
                    gpu_process_cpu_percent=process_cpu,
                    gpu_process_pids=tuple(sorted(process_pids)),
                )
            )
        except Exception as exc:
            if self.error is None:
                self.error = f"{type(exc).__name__}: {exc}"

    def _gpu_process_cpu_percent(self, pids: set[int]) -> float | None:
        if self._psutil is None:
            return None
        total = 0.0
        observed = False
        for pid in pids:
            try:
                process = self._processes.get(pid)
                if process is None:
                    process = self._psutil.Process(pid)
                    self._processes[pid] = process
                total += float(process.cpu_percent(interval=None))
                observed = True
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                self._processes.pop(pid, None)
        return total if observed else None

    def _shutdown_nvml(self) -> None:
        if self._pynvml is None:
            return
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass
        self._pynvml = None
        self._handle = None


def summarize_resource_samples(
    samples: list[ResourceSample],
    *,
    interval_s: float,
    error: str | None = None,
) -> dict[str, Any]:
    if not samples:
        return {
            "available": False,
            "sample_interval_s": interval_s,
            "samples": 0,
            "error": error,
        }

    steady_count = max(1, min(len(samples), round(5.0 / interval_s)))
    steady = samples[-steady_count:]
    return {
        "available": True,
        "sample_interval_s": interval_s,
        "samples": len(samples),
        "duration_s": samples[-1].elapsed_s,
        "gpu_memory_used_mib": _series_summary(
            [sample.gpu_memory_used_mib for sample in samples],
            steady=[sample.gpu_memory_used_mib for sample in steady],
        ),
        "gpu_memory_free_mib": _series_summary(
            [sample.gpu_memory_free_mib for sample in samples],
            steady=[sample.gpu_memory_free_mib for sample in steady],
        ),
        "gpu_process_memory_mib": _series_summary(
            [sample.gpu_process_memory_mib for sample in samples],
            steady=[sample.gpu_process_memory_mib for sample in steady],
        ),
        "gpu_util_percent": _optional_series_summary(
            [sample.gpu_util_percent for sample in samples]
        ),
        "power_w": _optional_series_summary([sample.power_w for sample in samples]),
        "system_cpu_percent": _optional_series_summary(
            [sample.system_cpu_percent for sample in samples]
        ),
        "gpu_process_cpu_percent": _optional_series_summary(
            [sample.gpu_process_cpu_percent for sample in samples]
        ),
        "gpu_process_pids": sorted(
            {pid for sample in samples for pid in sample.gpu_process_pids}
        ),
        "error": error,
    }


def collect_benchmark_provenance(
    *,
    model_id: str,
    model_revision: str | None,
    dataset_id: str,
    dataset_revision: str | None,
    launch_command: str | None,
    server_config: dict[str, Any],
) -> dict[str, Any]:
    packages = {
        dist.metadata.get("Name", "unknown"): dist.version
        for dist in importlib.metadata.distributions()
    }
    package_payload = json.dumps(packages, sort_keys=True, separators=(",", ":"))

    try:
        import torch

        torch_cuda = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        torch_cuda = None
        cudnn_version = None

    return {
        "schema_version": 1,
        "repository": {
            "commit": _command("git", "rev-parse", "HEAD"),
            "branch": _command("git", "branch", "--show-current"),
            "dirty": bool(_command("git", "status", "--porcelain")),
        },
        "host": {
            "platform": platform.platform(),
            "cpu_model": _first_prefixed_line("/proc/cpuinfo", "model name"),
            "logical_cpus": os.cpu_count(),
            "memory_total_kib": _first_prefixed_line("/proc/meminfo", "MemTotal"),
        },
        "gpu": {
            "nvidia_smi_csv": _command(
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version,pstate,"
                "power.limit,clocks.sm,clocks.mem,compute_cap",
                "--format=csv,noheader,nounits",
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_build": torch_cuda,
            "cudnn_version": cudnn_version,
        },
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "sglang",
                "sglang-omni",
                "transformers",
                "flash-attn-4",
                "flashinfer-python",
                "sglang-kernel",
            )
        },
        "dependency_freeze_sha256": hashlib.sha256(
            package_payload.encode()
        ).hexdigest(),
        "artifacts": {
            "model_id": model_id,
            "model_revision": model_revision,
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
        },
        "launch_command": launch_command,
        "server_config": server_config,
        "normalization": {
            "en": "whisper.normalizers.EnglishTextNormalizer",
            "zh": "strip zhon+ASCII punctuation, remove spaces, score characters",
        },
    }


def _resolve_nvml_handle(pynvml: Any, logical_gpu_index: int):
    visible = [
        token.strip()
        for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if token.strip()
    ]
    device: int | str = logical_gpu_index
    if visible:
        if logical_gpu_index >= len(visible):
            raise ValueError(
                f"logical GPU {logical_gpu_index} is not in CUDA_VISIBLE_DEVICES"
            )
        token = visible[logical_gpu_index]
        device = int(token) if token.isdigit() else token
    if isinstance(device, int):
        return pynvml.nvmlDeviceGetHandleByIndex(device)
    return pynvml.nvmlDeviceGetHandleByUUID(device.encode())


def _nvml_compute_processes(pynvml: Any, handle: Any) -> list[Any]:
    processes: dict[int, Any] = {}
    for getter_name in (
        "nvmlDeviceGetComputeRunningProcesses",
        "nvmlDeviceGetGraphicsRunningProcesses",
    ):
        getter = getattr(pynvml, getter_name, None)
        if getter is None:
            continue
        try:
            for process in getter(handle):
                processes[int(process.pid)] = process
        except Exception:
            continue
    return list(processes.values())


def _series_summary(values: list[float], *, steady: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "end": values[-1],
        "steady_mean": sum(steady) / len(steady),
    }


def _optional_series_summary(values: list[float | None]) -> dict[str, float] | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return {
        "min": min(present),
        "max": max(present),
        "mean": sum(present) / len(present),
    }


def _best_effort(callback):
    try:
        return callback()
    except Exception:
        return None


def _command(*args: str) -> str | None:
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _first_prefixed_line(path: str, prefix: str) -> str | None:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[-1].strip()
    except OSError:
        return None
    return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


__all__ = [
    "ResourceMonitor",
    "ResourceSample",
    "collect_benchmark_provenance",
    "summarize_resource_samples",
]
