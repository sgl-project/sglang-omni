# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from sglang_omni.cpu_alloc.allocator import CpuAllocationPlan, ProcessCpuAssignment
from sglang_omni.cpu_alloc.auto_pin import CpuAutoPinSupervisor


class FakeMonitor:
    def __init__(self):
        self.own = 5.0
        self.foreign = 0.0
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def snapshot(self):
        return {
            "own_busy_cores_last": self.own,
            "foreign_busy_cores_last": self.foreign,
        }


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: set(range(16)), raising=False
    )
    plan = CpuAllocationPlan(
        assignments={"asr": ProcessCpuAssignment("asr", (0, 1, 8, 9), True, 0)},
        shared_pools={0: ()},
        events=(),
    )
    calls = []
    monitor = FakeMonitor()
    sup = CpuAutoPinSupervisor(
        plan,
        {"asr": 100},
        declared_cores=5,
        monitor=monitor,
        ticks_to_pin=3,
        ticks_to_release=3,
        set_affinity=lambda pid, cpus: calls.append((pid, set(cpus))),
    )
    return sup, monitor, calls


def drive(sup, monitor, own, foreign, ticks):
    monitor.own, monitor.foreign = own, foreign
    for _ in range(ticks):
        sup.tick()


class TestAutoPin:
    def test_quiet_box_never_pins(self, rig):
        sup, monitor, calls = rig
        drive(sup, monitor, own=4.7, foreign=0.2, ticks=10)
        assert not sup.pinned
        assert calls == []

    def test_busy_neighbour_without_starvation_never_pins(self, rig):
        # Foreign load exists but the tree still gets most of its cores, which
        # is the level where pinning measured no gain.
        sup, monitor, calls = rig
        drive(sup, monitor, own=4.3, foreign=24.0, ticks=10)
        assert not sup.pinned
        assert calls == []

    def test_starvation_pins_after_hysteresis(self, rig):
        sup, monitor, calls = rig
        drive(sup, monitor, own=1.7, foreign=30.0, ticks=2)
        assert not sup.pinned
        drive(sup, monitor, own=1.7, foreign=30.0, ticks=1)
        assert sup.pinned
        assert calls == [(100, {0, 1, 8, 9})]

    def test_interruption_resets_the_streak(self, rig):
        sup, monitor, calls = rig
        drive(sup, monitor, own=1.7, foreign=30.0, ticks=2)
        drive(sup, monitor, own=4.8, foreign=30.0, ticks=1)
        drive(sup, monitor, own=1.7, foreign=30.0, ticks=2)
        assert not sup.pinned

    def test_release_restores_the_original_mask(self, rig):
        sup, monitor, calls = rig
        drive(sup, monitor, own=1.7, foreign=30.0, ticks=3)
        assert sup.pinned
        calls.clear()
        drive(sup, monitor, own=2.0, foreign=0.1, ticks=3)
        assert not sup.pinned
        assert calls == [(100, set(range(16)))]

    def test_pinned_state_ignores_its_own_low_usage(self, rig):
        # Once pinned the tree may sit well under its grant; only foreign load
        # going away should release it.
        sup, monitor, calls = rig
        drive(sup, monitor, own=1.7, foreign=30.0, ticks=3)
        calls.clear()
        drive(sup, monitor, own=0.5, foreign=30.0, ticks=10)
        assert sup.pinned
        assert calls == []

    def test_stop_releases(self, rig):
        sup, monitor, calls = rig
        drive(sup, monitor, own=1.7, foreign=30.0, ticks=3)
        calls.clear()
        sup.stop()
        assert not sup.pinned
        assert calls == [(100, set(range(16)))]

    def test_rejects_zero_declaration(self):
        with pytest.raises(ValueError, match="declared_cores"):
            CpuAutoPinSupervisor(CpuAllocationPlan({}, {}, ()), {}, declared_cores=0)
