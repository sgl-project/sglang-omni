# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.utils import cpu


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_cgroup_v2_cpu_quota(tmp_path: Path) -> None:
    cgroup_root = tmp_path / "cgroup"
    proc_self_cgroup = tmp_path / "self.cgroup"
    _write(proc_self_cgroup, "0::/pod/container\n")
    _write(cgroup_root / "pod/container/cpu.max", "1600000 100000\n")

    assert (
        cpu.cgroup_cpu_quota_count(
            cgroup_root=cgroup_root,
            proc_self_cgroup=proc_self_cgroup,
        )
        == 16
    )


def test_cgroup_v2_inherits_tighter_parent_quota(tmp_path: Path) -> None:
    cgroup_root = tmp_path / "cgroup"
    proc_self_cgroup = tmp_path / "self.cgroup"
    _write(proc_self_cgroup, "0::/pod/container\n")
    _write(cgroup_root / "cpu.max", "max 100000\n")
    _write(cgroup_root / "pod/cpu.max", "800000 100000\n")
    _write(cgroup_root / "pod/container/cpu.max", "max 100000\n")

    assert (
        cpu.cgroup_cpu_quota_count(
            cgroup_root=cgroup_root,
            proc_self_cgroup=proc_self_cgroup,
        )
        == 8
    )


def test_cgroup_v1_cpu_quota_rounds_partial_cpu_up(tmp_path: Path) -> None:
    cgroup_root = tmp_path / "cgroup"
    proc_self_cgroup = tmp_path / "self.cgroup"
    _write(proc_self_cgroup, "2:cpu,cpuacct:/pod/container\n")
    controller = cgroup_root / "cpu/pod/container"
    _write(controller / "cpu.cfs_quota_us", "150000\n")
    _write(controller / "cpu.cfs_period_us", "100000\n")

    assert (
        cpu.cgroup_cpu_quota_count(
            cgroup_root=cgroup_root,
            proc_self_cgroup=proc_self_cgroup,
        )
        == 2
    )


@pytest.mark.parametrize("cpu_max", ["max 100000\n", "invalid\n"])
def test_unlimited_or_invalid_cgroup_v2_quota(
    tmp_path: Path,
    cpu_max: str,
) -> None:
    cgroup_root = tmp_path / "cgroup"
    proc_self_cgroup = tmp_path / "self.cgroup"
    _write(proc_self_cgroup, "0::/\n")
    _write(cgroup_root / "cpu.max", cpu_max)

    assert (
        cpu.cgroup_cpu_quota_count(
            cgroup_root=cgroup_root,
            proc_self_cgroup=proc_self_cgroup,
        )
        is None
    )


def test_effective_cpu_count_uses_lower_affinity_or_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda _pid: set(range(224)))
    monkeypatch.setattr(cpu, "cgroup_cpu_quota_count", lambda: 16)

    assert cpu.effective_cpu_count() == 16


def test_bounded_intraop_threads_accounts_for_outer_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpu, "effective_cpu_count", lambda: 32)

    assert cpu.bounded_intraop_threads(worker_count=16, max_threads=8) == 2
