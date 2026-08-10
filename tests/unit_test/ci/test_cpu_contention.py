# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the cpuset contention sampler."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from tests.utils.ci_cpu_contention import (
    ContentionSampler,
    _parse_cpu_list,
    _Proc,
    _tree_pids,
    foreign_ticks,
)

_HAS_PROC = os.path.isdir("/proc")
_TWO_CPUS = (
    _HAS_PROC and hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) >= 2
)


def test_parse_cpu_list_ranges_and_singles():
    assert _parse_cpu_list("48-51,112") == {48, 49, 50, 51, 112}


def test_tree_pids_follows_descendants_only():
    procs = {
        1: _Proc(ppid=0, ticks=0),
        10: _Proc(ppid=1, ticks=0),
        20: _Proc(ppid=10, ticks=0),
        30: _Proc(ppid=20, ticks=0),
        99: _Proc(ppid=1, ticks=0),
    }
    assert _tree_pids(procs, 10) == {10, 20, 30}


def test_foreign_ticks_deducts_tree_and_clamps():
    prev = {10: _Proc(ppid=1, ticks=100, start=5)}
    # note (Jiaxin Deng): pid 20 was born inside the window, so its full
    # tick count is deducted; outsiders never appear here because foreign
    # time comes from the per-CPU busy delta, not from process scans.
    cur = {10: _Proc(ppid=1, ticks=300, start=5), 20: _Proc(ppid=1, ticks=50)}
    assert foreign_ticks(500, prev, cur, tree={10, 20}) == 250
    assert foreign_ticks(100, prev, cur, tree={10, 20}) == 0


def test_foreign_ticks_deducts_reaped_child_via_parent_cutime():
    # A pinned server burned 400 ticks then was killed and reaped; its time
    # reappears as the parent's child_ticks delta and must not count foreign.
    prev = {10: _Proc(ppid=1, ticks=100, child_ticks=50, start=5)}
    cur = {10: _Proc(ppid=1, ticks=110, child_ticks=450, start=5)}
    assert foreign_ticks(410, prev, cur, tree={10}) == 0


def test_foreign_ticks_pid_reuse_is_a_new_process():
    # Same pid, different starttime: the old incarnation's ticks must not
    # produce a negative deduction; the new one deducts its full count.
    prev = {10: _Proc(ppid=1, ticks=100000, start=5)}
    cur = {10: _Proc(ppid=1, ticks=30, start=999)}
    assert foreign_ticks(100, prev, cur, tree={10}) == 70


@pytest.mark.skipif(not _HAS_PROC, reason="requires Linux procfs")
def test_sampler_smoke_produces_summary():
    sampler = ContentionSampler({0, 1}, interval_s=0.2)
    sampler.start()
    time.sleep(0.7)
    sampler.stop()
    assert "[cpuset-contention]" in sampler.summary()


def _spin(core: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import os\nos.sched_setaffinity(0, {{{core}}})\nwhile True: pass",
        ]
    )


def _idlest_cores(candidates: list[int], count: int) -> list[int]:
    from tests.utils.ci_cpu_contention import cpuset_busy_ticks

    before = {c: cpuset_busy_ticks({c}) for c in candidates}
    time.sleep(0.3)
    return sorted(candidates, key=lambda c: cpuset_busy_ticks({c}) - before[c])[:count]


@pytest.mark.skipif(not _TWO_CPUS, reason="requires Linux procfs and two allowed CPUs")
def test_controlled_load_inside_cpuset_counts_outside_does_not():
    """A busy outsider on the pinned core is charged; one on another core is not."""
    # note (Jiaxin Deng): /proc/stat is host-global even inside a container,
    # so pick the two currently idlest allowed cores to keep the bounds tight
    # on a shared machine.
    core_in, core_out = _idlest_cores(sorted(os.sched_getaffinity(0)), 2)
    session_root = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    inside = _spin(core_in)
    outside = _spin(core_out)
    sampler = ContentionSampler({core_in}, interval_s=1.0, root_pid=session_root.pid)
    try:
        time.sleep(0.2)
        sampler.start()
        time.sleep(3.3)
        sampler.stop()
        peak = sampler.peak_foreign_cores()
        assert 0.4 <= peak <= 1.8, sampler.summary()
    finally:
        for proc in (inside, outside, session_root):
            proc.kill()
            proc.wait(timeout=5)
