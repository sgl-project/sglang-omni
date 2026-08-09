# SPDX-License-Identifier: Apache-2.0
"""Report foreign CPU load on the pinned OMNI_CI_CPUSET during a session.

Affinity is self-restraint, not a reservation: an unpinned process can still
be scheduled onto the reserved cores and inflate what a perf gate measures.
This sampler makes that attributable: each interval it charges CPU time
consumed on the pinned cores by processes outside the session tree, and the
fixture prints a summary so a failed speed gate can be triaged from the log.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

_SAMPLE_INTERVAL_S = 30.0
_WARN_FOREIGN_CORES = 1.0
# note (Jiaxin Deng): above this, the session's speed numbers describe the
# intruder, not the model; CI fails the stage so its retry re-measures.
FAIL_FOREIGN_CORES = 2.0


@dataclass
class _Proc:
    ppid: int
    ticks: int
    overlaps: bool


def _parse_cpu_list(spec: str) -> set[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo_text, _, hi_text = part.partition("-")
        cpus.update(range(int(lo_text), int(hi_text or lo_text) + 1))
    return cpus


def _snapshot(cpuset: set[int]) -> dict[int, _Proc]:
    procs: dict[int, _Proc] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                data = f.read().decode("ascii", "replace")
            rest = data[data.rindex(")") + 2 :].split()
            ppid = int(rest[1])
            ticks = int(rest[11]) + int(rest[12])
            allowed = ""
            with open(f"/proc/{pid}/status", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("Cpus_allowed_list"):
                        allowed = line.split(":", 1)[1].strip()
                        break
            # note (Jiaxin Deng): an unreadable mask counts as overlapping;
            # a false positive beats an invisible intruder in this report.
            overlaps = bool(cpuset & _parse_cpu_list(allowed)) if allowed else True
            procs[pid] = _Proc(ppid, ticks, overlaps)
        except (OSError, ValueError):
            continue
    return procs


def _tree_pids(procs: dict[int, _Proc], root: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, proc in procs.items():
        children.setdefault(proc.ppid, []).append(pid)
    tree, queue = {root}, [root]
    while queue:
        for child in children.get(queue.pop(), ()):
            if child not in tree:
                tree.add(child)
                queue.append(child)
    return tree


def foreign_ticks(prev: dict[int, _Proc], cur: dict[int, _Proc], tree: set[int]) -> int:
    return sum(
        cur[pid].ticks - prev[pid].ticks
        for pid in cur
        if pid in prev and cur[pid].overlaps and pid not in tree
    )


class ContentionSampler:
    """Background sampler; start() before the session, summary() after."""

    def __init__(self, cpuset: set[int], interval_s: float = _SAMPLE_INTERVAL_S):
        self._cpuset = set(cpuset)
        self._interval = interval_s
        self._root = os.getpid()
        self._hz = os.sysconf("SC_CLK_TCK")
        self._samples: list[float] = []
        self._errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def peak_foreign_cores(self) -> float:
        return max(self._samples, default=0.0)

    def _loop(self) -> None:
        prev = _snapshot(self._cpuset)
        prev_t = time.monotonic()
        while not self._stop.wait(self._interval):
            try:
                cur = _snapshot(self._cpuset)
                now = time.monotonic()
                tree = _tree_pids(cur, self._root)
                cores = foreign_ticks(prev, cur, tree) / self._hz / (now - prev_t)
                self._samples.append(cores)
                prev, prev_t = cur, now
            except Exception:
                self._errors += 1

    def summary(self) -> str:
        spec = ",".join(map(str, sorted(self._cpuset)))
        if not self._samples:
            return (
                f"[cpuset-contention] cpuset={spec} no completed sample "
                f"windows (errors={self._errors})"
            )
        mean = sum(self._samples) / len(self._samples)
        peak = max(self._samples)
        lines = [
            f"[cpuset-contention] cpuset={spec} windows={len(self._samples)} "
            f"foreign-cores mean={mean:.2f} max={peak:.2f} errors={self._errors}"
        ]
        if peak > _WARN_FOREIGN_CORES:
            lines.append(
                f"[cpuset-contention] WARNING: foreign load peaked at "
                f"{peak:.2f} cores on the pinned cpuset; speed metrics in "
                f"this session may be inflated by contention"
            )
        return "\n".join(lines)
