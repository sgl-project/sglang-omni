# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import pytest

from sglang_omni.cpu_alloc.topology import (
    _sysfs_pci_address,
    discover_topology,
    format_cpulist,
    gpu_numa_nodes,
    parse_cpulist,
)

# The sysfs PCI address contains ':', which Windows cannot use in file names.
pci_fs = pytest.mark.skipif(
    sys.platform == "win32", reason="PCI sysfs paths are not representable"
)


class TestCpulist:
    def test_roundtrip(self):
        assert parse_cpulist("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}
        assert format_cpulist({0, 1, 2, 3, 8, 10, 11}) == "0-3,8,10-11"

    def test_empty(self):
        assert parse_cpulist("") == set()
        assert format_cpulist([]) == ""

    def test_single(self):
        assert parse_cpulist("5") == {5}
        assert format_cpulist([5]) == "5"

    @pytest.mark.parametrize("bad", ["a", "1-", "3-1", "1,,2", "-2"])
    def test_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            parse_cpulist(bad)


class TestDiscoverTopology:
    def test_smt_and_numa_grouping(self, dual_node_sysfs: Path):
        topo = discover_topology(range(16), sysfs_root=dual_node_sysfs)
        assert topo.universe == tuple(range(16))
        assert topo.numa_nodes == (0, 1)
        node0 = topo.cores_on_node(0)
        assert [core.cpu_ids for core in node0] == [
            (0, 8),
            (1, 9),
            (2, 10),
            (3, 11),
        ]
        assert all(core.numa_node == 1 for core in topo.cores_on_node(1))

    def test_universe_subset_drops_siblings(self, dual_node_sysfs: Path):
        # CI-lane style universe: only cpus 0-3 (no SMT siblings visible).
        topo = discover_topology([0, 1, 2, 3], sysfs_root=dual_node_sysfs)
        assert all(core.cpu_ids in {(0,), (1,), (2,), (3,)} for core in topo.cores)
        assert topo.numa_nodes == (0,)

    def test_empty_universe_raises(self, dual_node_sysfs: Path):
        with pytest.raises(ValueError, match="universe is empty"):
            discover_topology([], sysfs_root=dual_node_sysfs)

    def test_missing_topology_raises(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="cpu99"):
            discover_topology([99], sysfs_root=tmp_path)

    def test_to_dict_is_json_ready(self, dual_node_sysfs: Path):
        topo = discover_topology(range(16), sysfs_root=dual_node_sysfs)
        data = topo.to_dict()
        assert data["universe"] == "0-15"
        assert data["cores"][0] == {
            "package": 0,
            "core": 0,
            "numa_node": 0,
            "cpu_ids": [0, 8],
        }


class TestGpuNumaNodes:
    def _write_pci(self, root: Path, address: str, node: int) -> None:
        device = root / "bus" / "pci" / "devices" / address
        device.mkdir(parents=True)
        (device / "numa_node").write_text(f"{node}\n")

    def test_bus_id_normalization(self):
        assert _sysfs_pci_address("00000000:0F:00.0") == "0000:0f:00.0"
        assert _sysfs_pci_address("0000:0f:00.0") == "0000:0f:00.0"

    @pci_fs
    def test_resolves_and_normalizes_bus_id(self, tmp_path: Path):
        self._write_pci(tmp_path, "0000:0f:00.0", 1)
        result = gpu_numa_nodes(
            [0],
            sysfs_root=tmp_path,
            pci_query=lambda: {0: "00000000:0F:00.0"},
        )
        assert result == {0: 1}

    @pci_fs
    def test_negative_node_maps_to_none(self, tmp_path: Path):
        self._write_pci(tmp_path, "0000:0f:00.0", -1)
        result = gpu_numa_nodes(
            [0], sysfs_root=tmp_path, pci_query=lambda: {0: "0000:0f:00.0"}
        )
        assert result == {0: None}

    def test_unknown_gpu_and_missing_sysfs(self, tmp_path: Path):
        result = gpu_numa_nodes(
            [0, 1],
            sysfs_root=tmp_path,
            pci_query=lambda: {0: "0000:0f:00.0"},
        )
        assert result == {0: None, 1: None}

    def test_query_failure_maps_all_to_none(self, tmp_path: Path):
        def boom() -> dict[int, str]:
            raise OSError("nvidia-smi missing")

        assert gpu_numa_nodes([0], sysfs_root=tmp_path, pci_query=boom) == {0: None}
