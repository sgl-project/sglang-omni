# SPDX-License-Identifier: Apache-2.0
"""Apply the CPU plan only while the pipeline is actually short of CPU.

Pinning pays when neighbours push the pipeline below the cores it declared,
and costs a little when they do not, so this watches the two together: how
much CPU the process tree is getting, and how much foreign work shares its
CPUs. It pins when the tree is starved next to foreign load and releases the
masks when that load goes away.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

from sglang_omni.cpu_alloc.allocator import CpuAllocationPlan
from sglang_omni.cpu_alloc.host_metrics import (
    HostCpuContentionMonitor,
    get_process_monitor,
)

logger = logging.getLogger(__name__)

# Note (Jiaxin Deng): Fun-ASR held 93% and 86% of its declared cores where
# pinning was worth nothing, and 34% where it was worth 3.5x.
STARVED_FRACTION = 0.6
RELIEVED_FOREIGN_CORES = 0.5
MIN_FOREIGN_CORES = 1.0


def set_process_affinity(pid: int, cpu_ids: set[int]) -> None:
    """Apply the mask to every task of *pid*, not just its main thread.

    Note (Jiaxin Deng): affinity is per thread; the task list is re-read until
    it stops growing so threads created mid-replay are covered too.
    """
    seen: set[int] = set()
    for _ in range(3):
        try:
            tids = [int(t) for t in os.listdir(f"/proc/{pid}/task")]
        except (OSError, ValueError):
            return
        fresh = [tid for tid in tids if tid not in seen]
        if not fresh:
            return
        for tid in fresh:
            seen.add(tid)
            try:
                os.sched_setaffinity(tid, cpu_ids)
            except OSError:
                continue


@dataclass
class _State:
    pinned: bool = False
    starved_ticks: int = 0
    relieved_ticks: int = 0


class CpuAutoPinSupervisor:
    """Pin the plan under contention, release it when the box goes quiet."""

    def __init__(
        self,
        plan: CpuAllocationPlan,
        pids: dict[str, int],
        declared_cores: int,
        *,
        monitor: HostCpuContentionMonitor | None = None,
        interval_s: float = 10.0,
        ticks_to_pin: int = 3,
        ticks_to_release: int = 3,
        set_affinity=set_process_affinity,
    ):
        if declared_cores < 1:
            raise ValueError("declared_cores must be >= 1")
        self._plan = plan
        self._pids = dict(pids)
        self._declared = declared_cores
        self._monitor = monitor or get_process_monitor(interval_s=interval_s)
        self._interval = interval_s
        self._ticks_to_pin = ticks_to_pin
        self._ticks_to_release = ticks_to_release
        self._set_affinity = set_affinity
        self._baseline = {
            name: set(os.sched_getaffinity(pid))
            for name, pid in self._pids.items()
            if hasattr(os, "sched_getaffinity")
        }
        self._state = _State()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def pinned(self) -> bool:
        return self._state.pinned

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Supervisor already started")
        self._monitor.start()
        self._thread = threading.Thread(
            target=self._run, name="cpu-auto-pin", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5)
            self._thread = None
        self._monitor.stop()
        if self._state.pinned:
            self._release()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.tick()
            except Exception:
                logger.exception("cpu_alloc auto supervisor tick failed")

    def tick(self) -> None:
        """One decision step; public so tests can drive it synchronously."""
        snap = self._monitor.snapshot()
        own = snap.get("own_busy_cores_last")
        foreign = snap.get("foreign_busy_cores_last")
        if own is None or foreign is None:
            return
        state = self._state
        if not state.pinned:
            starved = (
                own < STARVED_FRACTION * self._declared and foreign >= MIN_FOREIGN_CORES
            )
            state.starved_ticks = state.starved_ticks + 1 if starved else 0
            if state.starved_ticks >= self._ticks_to_pin:
                self._apply()
            return
        relieved = foreign < RELIEVED_FOREIGN_CORES
        state.relieved_ticks = state.relieved_ticks + 1 if relieved else 0
        if state.relieved_ticks >= self._ticks_to_release:
            self._release()

    def _apply(self) -> None:
        for name, assignment in self._plan.assignments.items():
            pid = self._pids.get(name)
            if pid is not None and assignment.cpu_ids:
                self._set_affinity(pid, set(assignment.cpu_ids))
        self._state = _State(pinned=True)
        logger.info("cpu_alloc: contention detected, applied the CPU plan")

    def _release(self) -> None:
        for name, pid in self._pids.items():
            baseline = self._baseline.get(name)
            if baseline:
                self._set_affinity(pid, baseline)
        self._state = _State(pinned=False)
        logger.info("cpu_alloc: contention gone, released the CPU plan")
