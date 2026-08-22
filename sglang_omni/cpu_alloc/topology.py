# SPDX-License-Identifier: Apache-2.0
"""Linux CPU/NUMA/SMT topology discovery.

Every input (CPU universe, sysfs root, GPU PCI query) is injectable so the
planning logic stays testable on non-Linux hosts and in CI.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

_CPULIST_RANGE = re.compile(r"^(\d+)(?:-(\d+))?$")


def parse_cpulist(spec: str) -> set[int]:
    """Parse a kernel cpulist string ("0-3,8,10-11") into a CPU id set."""
    cpus: set[int] = set()
    text = spec.strip()
    if not text:
        return cpus
    for part in text.split(","):
        part = part.strip()
        match = _CPULIST_RANGE.match(part)
        if match is None:
            raise ValueError(f"Invalid cpulist component {part!r} in {spec!r}")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if end < start:
            raise ValueError(f"Invalid cpulist range {part!r} in {spec!r}")
        cpus.update(range(start, end + 1))
    return cpus


def format_cpulist(cpus: Iterable[int]) -> str:
    """Format CPU ids as a canonical kernel cpulist string."""
    ordered = sorted(set(cpus))
    if not ordered:
        return ""
    parts: list[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = cpu
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


@dataclass(frozen=True)
class PhysicalCore:
    """One physical core, keyed by (physical_package_id, core_id); cpu_ids
    are its SMT siblings visible inside the universe."""

    key: tuple[int, int]
    numa_node: int
    cpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class CpuTopology:
    universe: tuple[int, ...]
    cores: tuple[PhysicalCore, ...]

    @property
    def numa_nodes(self) -> tuple[int, ...]:
        return tuple(sorted({core.numa_node for core in self.cores}))

    def cores_on_node(self, numa_node: int) -> tuple[PhysicalCore, ...]:
        return tuple(c for c in self.cores if c.numa_node == numa_node)

    def to_dict(self) -> dict:
        return {
            "universe": format_cpulist(self.universe),
            "cores": [
                {
                    "package": core.key[0],
                    "core": core.key[1],
                    "numa_node": core.numa_node,
                    "cpu_ids": list(core.cpu_ids),
                }
                for core in self.cores
            ],
        }


def _read_int(path: Path) -> int:
    return int(path.read_text().strip())


def _cpu_to_numa_map(root: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    node_root = root / "devices" / "system" / "node"
    if node_root.is_dir():
        for node_dir in sorted(node_root.glob("node[0-9]*")):
            node_id = int(node_dir.name[len("node") :])
            cpulist = node_dir / "cpulist"
            if not cpulist.is_file():
                continue
            for cpu in parse_cpulist(cpulist.read_text()):
                mapping[cpu] = node_id
    return mapping


def discover_topology(
    universe: Iterable[int] | None = None,
    *,
    sysfs_root: str | os.PathLike = "/sys",
) -> CpuTopology:
    """Build the topology of the CPUs this process may use.

    The default universe is ``sched_getaffinity(0)``, which already reflects
    any container cpuset or CI lane restriction.
    """
    root = Path(sysfs_root)
    if universe is None:
        universe = os.sched_getaffinity(0)
    cpus = sorted(set(int(c) for c in universe))
    if not cpus:
        raise ValueError("CPU universe is empty")

    cpu_to_numa = _cpu_to_numa_map(root)
    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for cpu in cpus:
        topo_dir = root / "devices" / "system" / "cpu" / f"cpu{cpu}" / "topology"
        try:
            core_id = _read_int(topo_dir / "core_id")
            package_id = _read_int(topo_dir / "physical_package_id")
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot read CPU topology for cpu{cpu} under {root}"
            ) from exc
        numa_node = cpu_to_numa.get(cpu, 0)
        grouped[(numa_node, package_id, core_id)].append(cpu)

    cores = tuple(
        PhysicalCore(
            key=(package_id, core_id),
            numa_node=numa_node,
            cpu_ids=tuple(sorted(cpu_ids)),
        )
        for (numa_node, package_id, core_id), cpu_ids in sorted(grouped.items())
    )
    return CpuTopology(universe=tuple(cpus), cores=cores)


def _nvidia_smi_pci_query() -> dict[int, str]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    mapping: dict[int, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        index_text, bus_id = (part.strip() for part in line.split(",", 1))
        mapping[int(index_text)] = bus_id
    return mapping


def _sysfs_pci_address(bus_id: str) -> str:
    # Note (Jiaxin Deng): nvidia-smi reports "00000000:0F:00.0"; sysfs wants "0000:0f:00.0".
    text = bus_id.strip().lower()
    if len(text.split(":")[0]) == 8:
        text = text[4:]
    return text


def gpu_numa_nodes(
    gpu_ids: Iterable[int],
    *,
    sysfs_root: str | os.PathLike = "/sys",
    pci_query: Callable[[], dict[int, str]] = _nvidia_smi_pci_query,
) -> dict[int, int | None]:
    """Resolve each GPU index to its NUMA node (None when unresolvable)."""
    wanted = sorted(set(int(g) for g in gpu_ids))
    if not wanted:
        return {}
    root = Path(sysfs_root)
    try:
        pci_map = pci_query()
    except (OSError, subprocess.SubprocessError, ValueError):
        return {gpu: None for gpu in wanted}

    result: dict[int, int | None] = {}
    for gpu in wanted:
        bus_id = pci_map.get(gpu)
        if bus_id is None:
            result[gpu] = None
            continue
        node_file = root / "bus" / "pci" / "devices" / _sysfs_pci_address(bus_id)
        node_file = node_file / "numa_node"
        try:
            node = int(node_file.read_text().strip())
        except (FileNotFoundError, ValueError, OSError):
            result[gpu] = None
            continue
        result[gpu] = node if node >= 0 else None
    return result
