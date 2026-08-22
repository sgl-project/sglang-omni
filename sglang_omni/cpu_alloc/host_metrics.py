# SPDX-License-Identifier: Apache-2.0
"""Continuous host CPU contention monitor for serving.

Ports the CI cpuset-contention sampling idea into a long-running
serving observation: how much foreign load runs on the CPUs this server is
allowed to use. "Foreign" is total busy time on the allowed cpuset minus the
busy time of this process tree, so a p99 wobble can be attributed to (or
cleared of) core theft by reading a number instead of guessing.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


def _parse_cpu_line(line: str) -> tuple[int, int]:
    parts = line.split()
    values = [int(v) for v in parts[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def _cpuset_busy_jiffies(cpus: set[int], proc_root: Path) -> int:
    busy = 0
    with open(proc_root / "stat") as f:
        for line in f:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            cpu_id = int(line.split()[0][3:])
            if cpu_id not in cpus:
                continue
            total, idle = _parse_cpu_line(line)
            busy += total - idle
    return busy


def _tree_pids(root_pid: int, proc_root: Path) -> list[int]:
    """Collect the process tree via /proc/<pid>/task/*/children."""
    pids = [root_pid]
    frontier = [root_pid]
    while frontier:
        pid = frontier.pop()
        task_dir = proc_root / str(pid) / "task"
        try:
            tids = os.listdir(task_dir)
        except OSError:
            continue
        for tid in tids:
            try:
                text = (task_dir / tid / "children").read_text()
            except OSError:
                continue
            for child in text.split():
                child_pid = int(child)
                pids.append(child_pid)
                frontier.append(child_pid)
    return pids


def _pid_cpu_jiffies(pid: int, proc_root: Path) -> int:
    try:
        data = (
            (proc_root / str(pid) / "stat")
            .read_bytes()
            .decode("ascii", errors="replace")
        )
    except OSError:
        return 0
    fields = data.rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])


@dataclass(frozen=True)
class ContentionSample:
    monotonic: float
    foreign_cores: float
    own_cores: float


class HostCpuContentionMonitor:
    """Background sampler of foreign load on this process's allowed cpuset."""

    def __init__(
        self,
        *,
        interval_s: float = 10.0,
        window: int = 60,
        proc_root: str | os.PathLike = "/proc",
        clock=time.monotonic,
    ):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._interval = interval_s
        self._proc_root = Path(proc_root)
        self._clock = clock
        self._root_pid = os.getpid()
        self._hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        self._samples: deque[ContentionSample] = deque(maxlen=window)
        self._peak_foreign = 0.0
        self._errors = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev: tuple[float, int, int] | None = None
        self._pid_jiffies: dict[int, int] = {}
        self._retired_jiffies = 0

    @property
    def available(self) -> bool:
        return hasattr(os, "sched_getaffinity")

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="host-cpu-contention", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                self.sample_once()
            except Exception:
                with self._lock:
                    self._errors += 1

    def sample_once(self) -> None:
        """One sampling step; public so tests can drive it synchronously."""
        cpus = set(os.sched_getaffinity(0))
        now = self._clock()
        busy = _cpuset_busy_jiffies(cpus, self._proc_root)
        live = {
            pid: _pid_cpu_jiffies(pid, self._proc_root)
            for pid in _tree_pids(self._root_pid, self._proc_root)
        }
        # Note (Jiaxin Deng): a reaped child takes its time out of /proc, so
        # a sum over live pids walks backwards and reads as foreign load.
        for pid, jiffies in self._pid_jiffies.items():
            if pid not in live:
                self._retired_jiffies += jiffies
        self._pid_jiffies = live
        own = sum(live.values()) + self._retired_jiffies
        prev = self._prev
        self._prev = (now, busy, own)
        if prev is None:
            return
        prev_t, prev_busy, prev_own = prev
        dt = now - prev_t
        if dt <= 0:
            return
        busy_cores = (busy - prev_busy) / self._hz / dt
        own_cores = max(0.0, (own - prev_own) / self._hz / dt)
        foreign = max(0.0, busy_cores - own_cores)
        with self._lock:
            self._samples.append(ContentionSample(now, foreign, own_cores))
            self._peak_foreign = max(self._peak_foreign, foreign)

    def snapshot(self) -> dict:
        if not self.available:
            return {"available": False}
        cpus = sorted(os.sched_getaffinity(0))
        with self._lock:
            samples = list(self._samples)
            peak = self._peak_foreign
            errors = self._errors
        last = samples[-1] if samples else None
        window_peak = max((s.foreign_cores for s in samples), default=0.0)
        return {
            "available": True,
            "cpuset_size": len(cpus),
            "cpuset": _format_cpulist(cpus),
            "foreign_busy_cores_last": round(last.foreign_cores, 3) if last else None,
            "foreign_busy_cores_window_peak": round(window_peak, 3),
            "foreign_busy_cores_peak": round(peak, 3),
            "own_busy_cores_last": round(last.own_cores, 3) if last else None,
            "samples": len(samples),
            "sample_errors": errors,
            "interval_s": self._interval,
        }


_PROCESS_MONITOR: HostCpuContentionMonitor | None = None


def get_process_monitor(**kwargs) -> HostCpuContentionMonitor:
    """The one sampler for this process.

    Note (Jiaxin Deng): both the endpoint and the auto supervisor want the
    same reading, and a second sampler is a second thread walking /proc.
    """
    global _PROCESS_MONITOR
    if _PROCESS_MONITOR is None:
        _PROCESS_MONITOR = HostCpuContentionMonitor(**kwargs)
    return _PROCESS_MONITOR


def _format_cpulist(cpus: list[int]) -> str:
    from sglang_omni.cpu_alloc.topology import format_cpulist

    return format_cpulist(cpus)
