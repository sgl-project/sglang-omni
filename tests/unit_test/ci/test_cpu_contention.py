# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the cpuset contention sampler."""

from __future__ import annotations

import os

import pytest

from tests.utils.ci_cpu_contention import (
    ContentionSampler,
    _parse_cpu_list,
    _Proc,
    _tree_pids,
    foreign_ticks,
)


def test_parse_cpu_list_ranges_and_singles():
    assert _parse_cpu_list("48-51,112") == {48, 49, 50, 51, 112}


def test_tree_pids_follows_descendants_only():
    procs = {
        1: _Proc(ppid=0, ticks=0, overlaps=True),
        10: _Proc(ppid=1, ticks=0, overlaps=True),
        20: _Proc(ppid=10, ticks=0, overlaps=True),
        30: _Proc(ppid=20, ticks=0, overlaps=True),
        99: _Proc(ppid=1, ticks=0, overlaps=True),
    }
    assert _tree_pids(procs, 10) == {10, 20, 30}


def test_foreign_ticks_charges_overlapping_outsiders_only():
    prev = {
        10: _Proc(ppid=1, ticks=100, overlaps=True),
        50: _Proc(ppid=1, ticks=100, overlaps=True),
        60: _Proc(ppid=1, ticks=100, overlaps=False),
    }
    cur = {
        10: _Proc(ppid=1, ticks=500, overlaps=True),
        50: _Proc(ppid=1, ticks=300, overlaps=True),
        60: _Proc(ppid=1, ticks=900, overlaps=False),
        70: _Proc(ppid=1, ticks=999, overlaps=True),
    }
    # note (Jiaxin Deng): 10 is in-tree, 60 does not overlap, 70 lacks a
    # baseline; only pid 50 may be charged.
    assert foreign_ticks(prev, cur, tree={10}) == 200


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="requires Linux procfs")
def test_sampler_smoke_produces_summary():
    sampler = ContentionSampler({0, 1}, interval_s=0.2)
    sampler.start()
    import time

    time.sleep(0.7)
    sampler.stop()
    assert "[cpuset-contention]" in sampler.summary()
