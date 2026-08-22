# SPDX-License-Identifier: Apache-2.0
"""Two-pool CPU allocation over physical cores.

Exclusive demands are granted whole physical cores (all SMT siblings move
together, so no foreign thread lands on a sibling). Everything else shares
the remaining CPUs of its NUMA node. When a node cannot satisfy every
exclusive demand, the processes that do not fit move to the shared pool,
which is today's behavior; it never silently oversubscribes an exclusive
grant, and every degradation is recorded in ``CpuAllocationPlan.events``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.cpu_alloc.topology import CpuTopology, PhysicalCore, format_cpulist


@dataclass(frozen=True)
class ProcessCpuDemand:
    """Aggregated exclusive-core demand of one OS process."""

    process_name: str
    numa_node: int | None
    exclusive_cores: int = 0

    def __post_init__(self) -> None:
        if self.exclusive_cores < 0:
            raise ValueError(f"Process {self.process_name!r}: core demand must be >= 0")


@dataclass(frozen=True)
class ProcessCpuAssignment:
    process_name: str
    cpu_ids: tuple[int, ...]
    exclusive: bool
    numa_node: int | None = None


@dataclass(frozen=True)
class CpuAllocationPlan:
    assignments: dict[str, ProcessCpuAssignment]
    shared_pools: dict[int | None, tuple[int, ...]]
    events: tuple[str, ...]
    # Note (Jiaxin Deng): physical cores, not mask width; a declaration and
    # its cpu ids differ by 2x on an SMT2 host.
    exclusive_physical_cores: int = 0
    universe_physical_cores: int = 0

    def to_dict(self) -> dict:
        return {
            "assignments": {
                name: {
                    "cpus": format_cpulist(assignment.cpu_ids),
                    "exclusive": assignment.exclusive,
                }
                for name, assignment in sorted(self.assignments.items())
            },
            "shared_pools": {
                str(node): format_cpulist(cpus)
                for node, cpus in sorted(
                    self.shared_pools.items(), key=lambda item: str(item[0])
                )
            },
            "events": list(self.events),
            "exclusive_physical_cores": self.exclusive_physical_cores,
            "universe_physical_cores": self.universe_physical_cores,
        }


@dataclass
class _NodeState:
    free_cores: list[PhysicalCore]
    reserved_shared: int


def _anchor_node(
    demand: ProcessCpuDemand,
    node_states: dict[int, _NodeState],
    projected: dict[int, int],
    events: list[str],
) -> int | None:
    """Node to grant exclusive cores on, or None to stay in the shared pool.

    An explicit node is honored; without one the demand goes to the node with
    the most remaining capacity, so exclusive grants spread across sockets
    instead of piling onto the first one. An explicit node that has no usable
    cores stays in the shared pool: a guessed grant could pin the process to
    the wrong socket, which is worse than not pinning.
    """
    if demand.numa_node is not None:
        if demand.numa_node in node_states:
            return demand.numa_node
        events.append(
            f"process {demand.process_name}: NUMA node {demand.numa_node} has "
            f"no usable cores in the universe; keeping it in the shared pool"
        )
        return None
    return max(
        node_states,
        key=lambda n: (
            len(node_states[n].free_cores)
            - node_states[n].reserved_shared
            - projected[n],
            -n,
        ),
    )


def allocate(
    topology: CpuTopology,
    demands: list[ProcessCpuDemand],
    *,
    min_shared_physical_cores: int = 1,
) -> CpuAllocationPlan:
    """Allocate exclusive physical cores per process and build shared pools."""
    names = [d.process_name for d in demands]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate process names in demands: {names}")
    if min_shared_physical_cores < 1:
        raise ValueError("min_shared_physical_cores must be >= 1")

    events: list[str] = []
    capacity = {node: len(topology.cores_on_node(node)) for node in topology.numa_nodes}
    if not capacity:
        raise ValueError("topology has no NUMA nodes")

    exclusive_demands = sorted(
        (d for d in demands if d.exclusive_cores),
        key=lambda d: (-d.exclusive_cores, d.process_name),
    )
    shared_declared = any(not d.exclusive_cores for d in demands)

    def _place(
        reserve: int,
    ) -> tuple[dict[str, int], dict[int, int], list[str], list[str]]:
        # Note (Jiaxin Deng): charge a node only once the demand is known to
        # fit, so a demand that ends up shared strands nothing.
        placed: dict[str, int] = {}
        used = dict.fromkeys(capacity, 0)
        demoted: list[str] = []
        notes: list[str] = []
        room = lambda n: capacity[n] - used[n] - reserve  # noqa: E731
        for demand in exclusive_demands:
            want = demand.exclusive_cores
            node = demand.numa_node
            if node is not None and node not in capacity:
                notes.append(
                    f"process {demand.process_name}: NUMA node {node} has no "
                    f"usable cores in the universe; moved to the shared pool"
                )
                demoted.append(demand.process_name)
                continue
            if node is None:
                fits = [n for n in sorted(capacity) if room(n) >= want]
                node = max(fits, key=lambda n: (room(n), -n), default=None)
            elif room(node) < want:
                node = None
            if node is None:
                notes.append(
                    f"process {demand.process_name}: wants {want} core(s) but "
                    f"no node has room; moved to the shared pool"
                )
                demoted.append(demand.process_name)
                continue
            placed[demand.process_name] = node
            used[node] += want
        return placed, used, demoted, notes

    reserve = min_shared_physical_cores if shared_declared else 0
    placed, used, demoted, notes = _place(reserve)
    if demoted and reserve == 0:
        # Somebody will live in the shared pool after all, so redo the layout
        # with the pool reserved instead of evicting a holder that already fit.
        reserve = min_shared_physical_cores
        placed, used, demoted, notes = _place(reserve)
    events.extend(notes)

    granted = {
        d.process_name: d.exclusive_cores
        for d in exclusive_demands
        if d.process_name in placed
    }
    has_shared_tenant = bool(demoted) or shared_declared
    # Note (Jiaxin Deng): cores held for a pool nobody joins are taken from
    # the only process there is, measured as -11% on a single-process replica.
    if not has_shared_tenant:
        for node in capacity:
            local = sorted(n for n, nd in placed.items() if nd == node)
            if not local:
                continue
            spare = capacity[node] - used[node]
            for i, name in enumerate(local):
                granted[name] += spare // len(local) + (
                    1 if i < spare % len(local) else 0
                )

    node_states = {
        node: _NodeState(
            free_cores=list(topology.cores_on_node(node)),
            reserved_shared=reserve,
        )
        for node in capacity
    }
    anchored = placed
    exclusive_demands = [d for d in exclusive_demands if d.process_name in placed]

    assignments: dict[str, ProcessCpuAssignment] = {}
    for demand in exclusive_demands:
        node = anchored[demand.process_name]
        state = node_states[node]
        count = granted[demand.process_name]
        if count == 0:
            continue
        cores, state.free_cores = (
            state.free_cores[:count],
            state.free_cores[count:],
        )
        cpu_ids = tuple(sorted(c for core in cores for c in core.cpu_ids))
        assignments[demand.process_name] = ProcessCpuAssignment(
            process_name=demand.process_name,
            cpu_ids=cpu_ids,
            exclusive=True,
            numa_node=node,
        )

    shared_pools: dict[int | None, tuple[int, ...]] = {
        node: tuple(sorted(c for core in state.free_cores for c in core.cpu_ids))
        for node, state in node_states.items()
    }
    all_shared = tuple(sorted(cpu for cpus in shared_pools.values() for cpu in cpus))
    shared_pools[None] = all_shared

    for demand in demands:
        if demand.process_name in assignments:
            continue
        node = demand.numa_node if demand.numa_node in node_states else None
        cpu_ids = shared_pools[node]
        if not cpu_ids:
            cpu_ids = all_shared
        assignments[demand.process_name] = ProcessCpuAssignment(
            process_name=demand.process_name,
            cpu_ids=cpu_ids,
            exclusive=False,
            numa_node=node,
        )

    return CpuAllocationPlan(
        assignments=assignments,
        shared_pools=shared_pools,
        events=tuple(events),
        exclusive_physical_cores=sum(granted[name] for name in placed),
        universe_physical_cores=sum(capacity.values()),
    )
