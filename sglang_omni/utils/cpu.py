# SPDX-License-Identifier: Apache-2.0
"""CPU capacity helpers that honor both affinity and cgroup quotas."""

from __future__ import annotations

import math
import os
from pathlib import Path

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")


def _read_self_cgroup_paths(proc_self_cgroup: Path) -> tuple[str | None, str | None]:
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
        _hierarchy_id, controllers, relative_path = parts
        if not controllers:
            v2_path = relative_path
        elif "cpu" in controllers.split(","):
            v1_cpu_path = relative_path
    return v2_path, v1_cpu_path


def _relative_cgroup_path(path: str | None) -> Path:
    if not path:
        return Path()
    return Path(path.lstrip("/"))


def _read_v2_quota(path: Path) -> int | None:
    try:
        quota_text, period_text = path.read_text(encoding="utf-8").split()[:2]
        if quota_text == "max":
            return None
        quota = int(quota_text)
        period = int(period_text)
    except (OSError, ValueError):
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(math.ceil(quota / period), 1)


def _read_v1_quota(directory: Path) -> int | None:
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
    return max(math.ceil(quota / period), 1)


def cgroup_cpu_quota_count(
    *,
    cgroup_root: Path = _CGROUP_ROOT,
    proc_self_cgroup: Path = _PROC_SELF_CGROUP,
) -> int | None:
    """Return the cgroup CPU quota as a CPU count, or ``None`` if unlimited."""
    v2_path, v1_cpu_path = _read_self_cgroup_paths(proc_self_cgroup)

    quota_counts: list[int] = []
    v2_relative = _relative_cgroup_path(v2_path)
    v2_directories = [
        cgroup_root.joinpath(*v2_relative.parts[:part_count])
        for part_count in range(len(v2_relative.parts), -1, -1)
    ]
    seen: set[Path] = set()
    for directory in v2_directories:
        candidate = directory / "cpu.max"
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            quota_count = _read_v2_quota(candidate)
            if quota_count is not None:
                quota_counts.append(quota_count)
    if quota_counts:
        return min(quota_counts)

    v1_relative = _relative_cgroup_path(v1_cpu_path)
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
            seen.add(directory)
            if (directory / "cpu.cfs_quota_us").is_file():
                quota_count = _read_v1_quota(directory)
                if quota_count is not None:
                    quota_counts.append(quota_count)
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
    return max(min(int(affinity_count), quota_count), 1)


def bounded_intraop_threads(*, worker_count: int, max_threads: int) -> int:
    """Size a shared intra-op pool after accounting for outer request workers."""
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    if max_threads < 1:
        raise ValueError("max_threads must be >= 1")
    return min(max(effective_cpu_count() // worker_count, 1), max_threads)
