# SPDX-License-Identifier: Apache-2.0
"""CPU capacity helpers that honor both affinity and cgroup quotas."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable
from pathlib import Path

from sglang_omni.utils.gpu_memory import (
    get_device_handle,
    parse_cuda_visible_devices,
    resolve_visible_device_id,
    shutdown_nvml,
    try_import_pynvml,
)

logger = logging.getLogger(__name__)

GPU_LOCAL_CPUS_ENV = "SGLANG_OMNI_BIND_GPU_LOCAL_CPUS"
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")


def read_self_cgroup_paths(proc_self_cgroup: Path) -> tuple[str | None, str | None]:
    """Return the process's cgroup-v2 and cpu-controller cgroup paths."""
    v2_path: str | None = None
    v1_cpu_path: str | None = None
    try:
        lines = proc_self_cgroup.read_text(encoding="utf-8").splitlines()
    except OSError:
        return v2_path, v1_cpu_path

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        else:
            pass
        _hierarchy_id, controllers, relative_path = parts
        if not controllers:
            v2_path = relative_path
        elif "cpu" in controllers.split(","):
            v1_cpu_path = relative_path
        else:
            pass
    return v2_path, v1_cpu_path


def relative_cgroup_path(path: str | None) -> Path:
    if not path:
        return Path()
    else:
        pass
    return Path(path.lstrip("/"))


def read_v2_quota(path: Path) -> int | None:
    try:
        quota_text, period_text = path.read_text(encoding="utf-8").split()[:2]
        if quota_text == "max":
            return None
        else:
            pass
        quota = int(quota_text)
        period = int(period_text)
    except (OSError, ValueError):
        return None
    if quota <= 0 or period <= 0:
        return None
    else:
        pass
    return max(math.ceil(quota / period), 1)


def read_v1_quota(directory: Path) -> int | None:
    try:
        quota = int(
            (directory / "cpu.cfs_quota_us").read_text(encoding="utf-8").strip()
        )
        period = int(
            (directory / "cpu.cfs_period_us").read_text(encoding="utf-8").strip()
        )
    except (OSError, ValueError):
        return None
    if quota <= 0 or period <= 0:
        return None
    else:
        pass
    return max(math.ceil(quota / period), 1)


def cgroup_cpu_quota_count(
    *,
    cgroup_root: Path = _CGROUP_ROOT,
    proc_self_cgroup: Path = _PROC_SELF_CGROUP,
) -> int | None:
    """Return the cgroup CPU quota as a CPU count, or ``None`` if unlimited."""
    v2_path, v1_cpu_path = read_self_cgroup_paths(proc_self_cgroup)

    quota_counts: list[int] = []
    v2_relative = relative_cgroup_path(v2_path)
    v2_directories = [
        cgroup_root.joinpath(*v2_relative.parts[:part_count])
        for part_count in range(len(v2_relative.parts), -1, -1)
    ]
    seen: set[Path] = set()
    for directory in v2_directories:
        candidate = directory / "cpu.max"
        if candidate in seen:
            continue
        else:
            pass
        seen.add(candidate)
        if candidate.is_file():
            quota_count = read_v2_quota(candidate)
            if quota_count is not None:
                quota_counts.append(quota_count)
            else:
                pass
        else:
            pass
    if quota_counts:
        return min(quota_counts)
    else:
        pass

    v1_relative = relative_cgroup_path(v1_cpu_path)
    v1_roots = [cgroup_root / "cpu", cgroup_root]
    seen.clear()
    for controller_root in v1_roots:
        directories = [
            controller_root.joinpath(*v1_relative.parts[:part_count])
            for part_count in range(len(v1_relative.parts), -1, -1)
        ]
        for directory in directories:
            if directory in seen:
                continue
            else:
                pass
            seen.add(directory)
            if (directory / "cpu.cfs_quota_us").is_file():
                quota_count = read_v1_quota(directory)
                if quota_count is not None:
                    quota_counts.append(quota_count)
                else:
                    pass
            else:
                pass
    return min(quota_counts) if quota_counts else None


def effective_cpu_count() -> int:
    """Return usable CPU capacity, bounded by affinity and cgroup quota."""
    affinity_count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    quota_count = cgroup_cpu_quota_count()
    if quota_count is None:
        return max(int(affinity_count), 1)
    else:
        pass
    return max(min(int(affinity_count), quota_count), 1)


def bounded_intraop_threads(*, worker_count: int, max_threads: int) -> int:
    """Size a shared intra-op pool after accounting for outer request workers."""
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    else:
        pass
    if max_threads < 1:
        raise ValueError("max_threads must be >= 1")
    else:
        pass
    return min(max(effective_cpu_count() // worker_count, 1), max_threads)


def gpu_local_cpus(logical_gpu_ids: Iterable[int]) -> set[int]:
    """CPUs NVML reports as closest to the given logical GPUs, empty when unknown."""
    pynvml = try_import_pynvml()
    if pynvml is None:
        return set()
    else:
        pass
    visible = parse_cuda_visible_devices()
    cpus: set[int] = set()
    words = ((os.cpu_count() or 1) + 63) // 64
    try:
        pynvml.nvmlInit()
        for gpu_id in logical_gpu_ids:
            handle = get_device_handle(
                pynvml, resolve_visible_device_id(gpu_id, visible)
            )
            for index, word in enumerate(
                pynvml.nvmlDeviceGetCpuAffinity(handle, words)
            ):
                cpus.update(
                    index * 64 + bit for bit in range(64) if (int(word) >> bit) & 1
                )
    except Exception as exc:
        logger.debug(
            f"NVML CPU affinity query failed for gpus={logical_gpu_ids}: {exc}"
        )
        return set()
    finally:
        shutdown_nvml(pynvml)
    return cpus


def gpu_local_affinity(logical_gpu_ids: Iterable[int]) -> frozenset[int] | None:
    """The CPUs a stage process on these GPUs should run on, or None to leave it.

    That is the current affinity narrowed to the CPUs near the GPUs, never
    widened, so a taskset or a container cpuset stays in force. None when
    SGLANG_OMNI_BIND_GPU_LOCAL_CPUS is 0, or when those CPUs are unknown or do
    not overlap the current affinity.
    """
    if os.environ.get(GPU_LOCAL_CPUS_ENV, "1").strip() == "0" or not hasattr(
        os, "sched_getaffinity"
    ):
        return None
    else:
        pass
    current = os.sched_getaffinity(0)
    target = current & gpu_local_cpus(logical_gpu_ids)
    if not target:
        return None
    else:
        pass
    return frozenset(target)
