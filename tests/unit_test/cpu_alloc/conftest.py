# SPDX-License-Identifier: Apache-2.0
"""Fake sysfs builders for CPU topology tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.cpu_alloc.topology import format_cpulist


def write_sysfs(
    root: Path,
    node_cores: dict[int, list[tuple[int, ...]]],
) -> Path:
    """Write a fake sysfs tree.

    ``node_cores`` maps NUMA node id to a list of physical cores, each given
    as the tuple of its SMT sibling CPU ids.
    """
    for node, cores in node_cores.items():
        node_dir = root / "devices" / "system" / "node" / f"node{node}"
        node_dir.mkdir(parents=True, exist_ok=True)
        all_cpus = [cpu for core in cores for cpu in core]
        (node_dir / "cpulist").write_text(format_cpulist(all_cpus) + "\n")
        for core_index, core in enumerate(cores):
            for cpu in core:
                topo = root / "devices" / "system" / "cpu" / f"cpu{cpu}" / "topology"
                topo.mkdir(parents=True, exist_ok=True)
                (topo / "core_id").write_text(f"{core_index}\n")
                (topo / "physical_package_id").write_text(f"{node}\n")
    return root


@pytest.fixture
def dual_node_sysfs(tmp_path: Path) -> Path:
    """2 NUMA nodes x 4 physical cores x 2 SMT siblings (16 CPUs).

    Node 0: cores (0,8) (1,9) (2,10) (3,11); node 1: (4,12) (5,13) (6,14) (7,15).
    """
    return write_sysfs(
        tmp_path,
        {
            0: [(0, 8), (1, 9), (2, 10), (3, 11)],
            1: [(4, 12), (5, 13), (6, 14), (7, 15)],
        },
    )
