# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import pytest

from sglang_omni.cpu_alloc.host_metrics import HostCpuContentionMonitor


class FakeProc:
    """Writable fake /proc: per-cpu busy jiffies and a small process tree."""

    def __init__(self, root: Path, cpus: list[int]):
        self.root = root
        self.cpus = cpus
        self.busy = dict.fromkeys(cpus, 0)
        self.pid_jiffies: dict[int, int] = {}
        self.children: dict[int, list[int]] = {}

    def write(self):
        lines = ["cpu  0 0 0 0 0 0 0 0 0 0"]
        for cpu in self.cpus:
            busy = self.busy[cpu]
            lines.append(f"cpu{cpu} {busy} 0 0 1000 0 0 0 0 0 0")
        (self.root / "stat").write_text("\n".join(lines) + "\n")
        for pid, jiffies in self.pid_jiffies.items():
            pid_dir = self.root / str(pid)
            task_dir = pid_dir / "task" / str(pid)
            task_dir.mkdir(parents=True, exist_ok=True)
            stat = f"{pid} (proc) S 1 1 1 0 -1 0 0 0 0 0 {jiffies} 0 0 0"
            (pid_dir / "stat").write_text(stat)
            kids = " ".join(str(c) for c in self.children.get(pid, []))
            (task_dir / "children").write_text(kids)


@pytest.fixture
def fake_proc(tmp_path, monkeypatch):
    cpus = [0, 1, 2, 3]
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(cpus), raising=False)
    proc = FakeProc(tmp_path, cpus)
    root_pid = os.getpid()
    proc.pid_jiffies[root_pid] = 0
    proc.children[root_pid] = [777]
    proc.pid_jiffies[777] = 0
    proc.write()
    return proc


def make_monitor(proc, clock_values):
    it = iter(clock_values)
    return HostCpuContentionMonitor(
        interval_s=1.0, proc_root=proc.root, clock=lambda: next(it)
    )


class TestHostCpuContentionMonitor:
    def test_foreign_excludes_own_tree(self, fake_proc):
        hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        monitor = make_monitor(fake_proc, [0.0, 10.0])
        monitor.sample_once()
        # Over 10s: total busy 30 cores-jiffies-equiv, own tree (incl. child)
        # accounts for 1 core, so foreign = 2 cores.
        for cpu in fake_proc.cpus[:3]:
            fake_proc.busy[cpu] += 10 * hz
        fake_proc.pid_jiffies[os.getpid()] += 5 * hz
        fake_proc.pid_jiffies[777] += 5 * hz
        fake_proc.write()
        monitor.sample_once()
        snap = monitor.snapshot()
        assert snap["available"] is True
        assert snap["cpuset"] == "0-3"
        assert snap["foreign_busy_cores_last"] == pytest.approx(2.0, abs=0.05)
        assert snap["own_busy_cores_last"] == pytest.approx(1.0, abs=0.05)
        assert snap["foreign_busy_cores_peak"] == pytest.approx(2.0, abs=0.05)

    def test_first_sample_yields_no_reading(self, fake_proc):
        monitor = make_monitor(fake_proc, [0.0])
        monitor.sample_once()
        snap = monitor.snapshot()
        assert snap["samples"] == 0
        assert snap["foreign_busy_cores_last"] is None

    def test_quiet_host_reads_zero(self, fake_proc):
        monitor = make_monitor(fake_proc, [0.0, 10.0])
        monitor.sample_once()
        fake_proc.write()
        monitor.sample_once()
        snap = monitor.snapshot()
        assert snap["foreign_busy_cores_last"] == 0.0

    def test_dead_child_is_tolerated(self, fake_proc):
        monitor = make_monitor(fake_proc, [0.0, 10.0])
        monitor.sample_once()
        # Child vanishes between samples: its stat read returns 0, no crash.
        (fake_proc.root / "777" / "stat").unlink()
        fake_proc.write()
        monitor.sample_once()
        assert monitor.snapshot()["samples"] == 1

    def test_invalid_interval_rejected(self):
        with pytest.raises(ValueError, match="interval_s"):
            HostCpuContentionMonitor(interval_s=0)


class TestReapedChildren:
    def test_a_child_exiting_is_not_read_as_foreign_load(self, fake_proc):
        hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        monitor = make_monitor(fake_proc, [0.0, 10.0, 20.0])
        fake_proc.pid_jiffies[777] = 100 * hz
        fake_proc.write()
        monitor.sample_once()

        # The child burns a core for 10s, then exits and is reaped: its time
        # leaves /proc but none of it was foreign.
        fake_proc.busy[0] += 10 * hz
        fake_proc.pid_jiffies[777] += 10 * hz
        fake_proc.write()
        monitor.sample_once()
        assert monitor.snapshot()["foreign_busy_cores_last"] == pytest.approx(
            0.0, abs=0.05
        )

        fake_proc.children[os.getpid()] = []
        del fake_proc.pid_jiffies[777]
        fake_proc.write()
        monitor.sample_once()
        snap = monitor.snapshot()
        assert snap["own_busy_cores_last"] == pytest.approx(0.0, abs=0.05)
        assert snap["foreign_busy_cores_last"] == pytest.approx(0.0, abs=0.05)
        assert snap["foreign_busy_cores_peak"] == pytest.approx(0.0, abs=0.05)


class TestLifecycle:
    def test_start_after_stop_samples_again(self, fake_proc, monkeypatch):
        monitor = HostCpuContentionMonitor(interval_s=0.01, proc_root=fake_proc.root)
        monitor.start()
        monitor.stop()
        monitor.start()
        assert monitor._thread is not None
        assert not monitor._stop_event.is_set()
        monitor.stop()


class TestProcessMonitorIsShared:
    def test_the_endpoint_and_the_supervisor_get_the_same_sampler(self, monkeypatch):
        import sglang_omni.cpu_alloc.host_metrics as hm

        monkeypatch.setattr(hm, "_PROCESS_MONITOR", None)
        first = hm.get_process_monitor(interval_s=1.0)
        assert hm.get_process_monitor() is first
