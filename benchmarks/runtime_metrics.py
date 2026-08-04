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

_NVML_SESSION_LOCK = threading.Lock()
_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ResourceSample:
    elapsed_s: float
    gpu_memory_used_mib: float
    gpu_memory_free_mib: float
    gpu_process_memory_mib: float | None
    gpu_util_percent: float | None
    power_w: float | None
    system_cpu_percent: float | None
    gpu_process_cpu_percent: float | None
    gpu_process_pids: tuple[int, ...]


class ResourceMonitor:
    """Sample one local GPU and explicitly selected processes in a background thread."""

    def __init__(
        self,
        gpu_index: int = 0,
        interval_s: float = 0.2,
        gpu_process_pids: list[int] | None = None,
    ) -> None:
        if gpu_index < 0:
            raise ValueError("gpu_index must be >= 0")
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        if any(pid <= 0 for pid in gpu_process_pids or []):
            raise ValueError("gpu_process_pids must contain only positive PIDs")
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.gpu_process_pids = frozenset(gpu_process_pids or [])
        self.samples: list[ResourceSample] = []
        self.error: str | None = None
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._pynvml: Any = None
        self._handle: Any = None
        self._psutil: Any = None
        self._processes: dict[int, Any] = {}
        self._inaccessible_process_pids: set[int] = set()

    def start(self) -> "ResourceMonitor":
        if not _NVML_SESSION_LOCK.acquire(blocking=False):
            self.error = "another NVML resource monitor is still active"
            self._ready_event.set()
            return self
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name=f"benchmark-resource-monitor-gpu{self.gpu_index}",
            daemon=True,
        )
        try:
            self._thread.start()
        except BaseException:
            _NVML_SESSION_LOCK.release()
            raise
        if not self._ready_event.wait(timeout=max(5.0, self.interval_s * 5)):
            self.error = "resource sampler initialization timed out"
        return self

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 5))
            if self._thread.is_alive() and self.error is None:
                self.error = "resource sampler did not stop before timeout"
        process_error = None
        if not self.gpu_process_pids:
            process_error = (
                "GPU process metrics require at least one explicit NVML PID; "
                "pass --gpu-process-pid"
            )
        elif missing_pids := self.gpu_process_pids - {
            pid for sample in self.samples for pid in sample.gpu_process_pids
        }:
            pids = ", ".join(map(str, sorted(missing_pids)))
            process_error = (
                f"target NVML PID(s) {pids} were not observed on GPU "
                f"{self.gpu_index}"
            )
        elif self._inaccessible_process_pids:
            pids = ", ".join(map(str, sorted(self._inaccessible_process_pids)))
            process_error = (
                f"GPU process CPU metrics unavailable for NVML PID(s) {pids}; "
                "use the host PID namespace (for Docker, --pid=host)"
            )
        return summarize_resource_samples(
            list(self.samples),
            interval_s=self.interval_s,
            error=self.error or process_error,
        )

    def _run(self) -> None:
        try:
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
                self._ready_event.set()
                return

            self._ready_event.set()
            self._sample_once()
            while not self._stop_event.wait(self.interval_s):
                self._sample_once()
            self._sample_once()
        finally:
            self._shutdown_nvml()
            _NVML_SESSION_LOCK.release()

    def _sample_once(self) -> None:
        try:
            pynvml = self._pynvml
            memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            all_nvml_processes = _nvml_compute_processes(pynvml, self._handle)
            nvml_processes = (
                [
                    process
                    for process in all_nvml_processes
                    if int(process.pid) in self.gpu_process_pids
                ]
                if all_nvml_processes is not None and self.gpu_process_pids
                else None
            )
            process_memory_bytes: int | None = 0 if nvml_processes is not None else None
            process_pids: set[int] = set()
            for process in nvml_processes or []:
                process_pids.add(int(process.pid))
                try:
                    used = process.usedGpuMemory
                except AttributeError:
                    process_memory_bytes = None
                    continue
                if not isinstance(used, int) or not 0 <= used < 2**63:
                    process_memory_bytes = None
                    continue
                if process_memory_bytes is not None:
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
            process_cpu = (
                self._gpu_process_cpu_percent(process_pids)
                if nvml_processes is not None
                else None
            )
            process_memory_mib = (
                float(process_memory_bytes) / (1024**2)
                if process_memory_bytes is not None
                else None
            )
            self.samples.append(
                ResourceSample(
                    elapsed_s=time.perf_counter() - self._started_at,
                    gpu_memory_used_mib=float(memory.used) / (1024**2),
                    gpu_memory_free_mib=float(memory.free) / (1024**2),
                    gpu_process_memory_mib=process_memory_mib,
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
                self._inaccessible_process_pids.add(pid)
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
        "gpu_process_memory_mib": _optional_series_summary(
            [sample.gpu_process_memory_mib for sample in samples]
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
    evaluation_input_sha256: str | None = None,
) -> dict[str, Any]:
    dependency_inventory = _distribution_inventory()
    package_payload = json.dumps(
        dependency_inventory,
        sort_keys=True,
        separators=(",", ":"),
    )
    repository_status = _git_command("status", "--porcelain")

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
            "commit": _git_command("rev-parse", "HEAD"),
            "branch": _git_command("branch", "--show-current"),
            "dirty": (None if repository_status is None else bool(repository_status)),
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
        "dependency_inventory": dependency_inventory,
        "dependency_freeze_sha256": hashlib.sha256(
            package_payload.encode()
        ).hexdigest(),
        "artifacts": {
            "requested_model_id": model_id,
            "declared_model_revision": model_revision,
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "evaluation_input_sha256": evaluation_input_sha256,
        },
        "launch_command": launch_command,
        "declared_server_config": server_config,
        "normalization": {
            "en": "whisper.normalizers.EnglishTextNormalizer",
            "zh": "strip zhon+ASCII punctuation, remove spaces, score characters",
        },
    }


def _resolve_nvml_handle(pynvml: Any, logical_gpu_index: int):
    visible_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_value is not None and not visible_value.strip():
        raise ValueError("CUDA_VISIBLE_DEVICES is empty")
    visible = (
        [token.strip() for token in visible_value.split(",") if token.strip()]
        if visible_value is not None
        else []
    )
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


def _nvml_compute_processes(pynvml: Any, handle: Any) -> list[Any] | None:
    getters = []
    try:
        getters.append(pynvml.nvmlDeviceGetComputeRunningProcesses)
    except AttributeError:
        pass
    try:
        getters.append(pynvml.nvmlDeviceGetGraphicsRunningProcesses)
    except AttributeError:
        pass

    available = False
    processes: dict[int, Any] = {}
    for getter in getters:
        try:
            current = getter(handle)
        except Exception:
            continue
        available = True
        for process in current:
            processes[int(process.pid)] = process
    return list(processes.values()) if available else None


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


def _git_command(*args: str) -> str | None:
    return _command("git", "-C", str(_REPO_ROOT), *args)


def _first_prefixed_line(path: str, prefix: str) -> str | None:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[-1].strip()
    except OSError:
        return None
    return None


def _distribution_inventory() -> list[dict[str, Any]]:
    inventory = []
    for distribution in importlib.metadata.distributions():
        entry: dict[str, Any] = {
            "name": distribution.metadata.get("Name", "unknown"),
            "version": distribution.version,
        }
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            try:
                entry["direct_url"] = json.loads(direct_url)
            except json.JSONDecodeError:
                entry["direct_url"] = direct_url
        inventory.append(entry)
    return sorted(
        inventory,
        key=lambda entry: (entry["name"].casefold(), entry["version"]),
    )


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
