# SPDX-License-Identifier: Apache-2.0
"""Report foreign CPU load on the pinned OMNI_CI_CPUSET during a session.

Affinity is self-restraint, not a reservation: an unpinned process can still
be scheduled onto the reserved cores and inflate what a perf gate measures.
This sampler makes that attributable. Foreign load is measured from the
kernel's per-CPU counters, not from process affinity masks: the cpuset's
busy ticks minus the session tree's ticks (the tree is pinned, so its time
is entirely inside the cpuset). Time spent by outsiders on other cores never
enters the counters, and short-lived intruders are still charged because the
per-CPU counters survive process exit. Reaped tree children are deducted
through their parent's cutime/cstime. Known blind spots, accepted: a tree
thread that widens its own affinity is over-deducted (nothing in the tree
does this; the runner-side partition check owns that invariant), and kernel
work the session induces asynchronously (softirq, kworkers) counts as
foreign by policy, since the alarm measures cpuset capacity not owned by
the tree.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

_SAMPLE_INTERVAL_S = 30.0
_WARN_FOREIGN_CORES = 1.0
# note (Jiaxin Deng): above this a round measured the intruder, not the
# model. Calibration rejects such rounds (tune.py). CI only reports:
# contention can only depress perf numbers, so a passing gate is real and
# failing a passing stage would spend its retries on nothing (PR 1379).
FAIL_FOREIGN_CORES = 2.0


@dataclass
class _Proc:
    ppid: int
    ticks: int
    child_ticks: int = 0
    start: int = 0


def _parse_cpu_list(spec: str) -> set[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo_text, _, hi_text = part.partition("-")
        cpus.update(range(int(lo_text), int(hi_text or lo_text) + 1))
    return cpus


def cpuset_busy_ticks(cpuset: set[int]) -> int:
    """Total non-idle ticks accumulated on the cpuset's cores."""
    busy = 0
    with open("/proc/stat", encoding="ascii", errors="replace") as f:
        for line in f:
            if line.startswith("cpu") and line[3:4].isdigit():
                parts = line.split()
                if int(parts[0][3:]) in cpuset:
                    # note (Jiaxin Deng): first 8 fields only; guest time is
                    # already inside user/nice and would be double-counted.
                    nums = list(map(int, parts[1:9]))
                    busy += sum(nums) - nums[3] - nums[4]
    return busy


def _snapshot() -> dict[int, _Proc]:
    procs: dict[int, _Proc] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                data = f.read().decode("ascii", "replace")
            rest = data[data.rindex(")") + 2 :].split()
            procs[pid] = _Proc(
                ppid=int(rest[1]),
                ticks=int(rest[11]) + int(rest[12]),
                child_ticks=int(rest[13]) + int(rest[14]),
                start=int(rest[19]),
            )
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


def foreign_ticks(
    busy_delta: int,
    prev: dict[int, _Proc],
    cur: dict[int, _Proc],
    tree: set[int],
) -> int:
    """Cpuset busy ticks not accounted for by the pinned session tree.

    Reaped children surface in their parent's cutime/cstime, so a server
    killed mid-window is still deducted instead of masquerading as multiple
    foreign cores. A pid is the same process only if starttime matches;
    otherwise it is treated as born in this window. Per-pid deltas clamp at
    zero so a reused pid cannot inject a negative deduction.
    """
    tree_delta = 0
    for pid in cur:
        if pid not in tree:
            continue
        proc = cur[pid]
        before = prev.get(pid)
        if before is not None and before.start == proc.start:
            tree_delta += max(0, proc.ticks - before.ticks)
            tree_delta += max(0, proc.child_ticks - before.child_ticks)
        else:
            tree_delta += proc.ticks + proc.child_ticks
    return max(0, busy_delta - tree_delta)


class ContentionSampler:
    """Background sampler; start() before the session, summary() after."""

    def __init__(
        self,
        cpuset: set[int],
        interval_s: float = _SAMPLE_INTERVAL_S,
        root_pid: int | None = None,
    ):
        self._cpuset = set(cpuset)
        self._interval = interval_s
        self._root = root_pid or os.getpid()
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

    def mean_foreign_cores(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    def window_count(self) -> int:
        return len(self._samples)

    def sustained_foreign_cores(self, windows: int, cores: float) -> bool:
        """Whether the last ``windows`` samples all exceed ``cores``.

        A caller that acts on intrusion needs to know the lane is taken, not
        that one window happened to land high: sampling is seconds and a
        gated stage is minutes, so a single crossing is thinner evidence than
        the decision it drives. Reading the tail rather than the whole series
        lets a caller poll this while the session is still running.
        """
        if windows <= 0 or len(self._samples) < windows:
            return False
        return all(sample > cores for sample in self._samples[-windows:])

    def _loop(self) -> None:
        prev_busy = cpuset_busy_ticks(self._cpuset)
        prev = _snapshot()
        prev_t = time.monotonic()
        while not self._stop.wait(self._interval):
            try:
                cur_busy = cpuset_busy_ticks(self._cpuset)
                cur = _snapshot()
                now = time.monotonic()
                tree = _tree_pids(cur, self._root)
                ticks = foreign_ticks(cur_busy - prev_busy, prev, cur, tree)
                self._samples.append(ticks / self._hz / (now - prev_t))
                prev_busy, prev, prev_t = cur_busy, cur, now
            except Exception:
                self._errors += 1

    def summary(self) -> str:
        spec = ",".join(map(str, sorted(self._cpuset)))
        if not self._samples:
            return (
                f"[cpuset-contention] cpuset={spec} no completed sample "
                f"windows (errors={self._errors})"
            )
        mean = self.mean_foreign_cores()
        peak = self.peak_foreign_cores()
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
