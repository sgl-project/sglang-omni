# SPDX-License-Identifier: Apache-2.0
"""NUMA placement for CPU stage worker processes.

CPU has no device index to place a stage with, so placement is the thread binding SGLang derives from SGLANG_CPU_OMP_THREADS_BIND. The binding unit is the OS process: stages sharing a process share its node, and a request that would split one process across nodes is rejected.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import Counter
from collections.abc import Mapping

import sglang.srt.distributed.bootstrap as sglang_bootstrap
import torch
from sglang.srt.utils import get_cpu_ids_by_node

from sglang_omni.config.schema import parse_replica_instance_name
from sglang_omni.utils.cpu import LAUNCHER_CPUS_ENV, parse_cpu_list

try:
    from sglang.srt.utils import get_physical_cpus_by_numa
except ImportError:
    # The pinned release names it differently from newer SGLang.
    from sglang.srt.utils import (
        get_physical_cpus_by_numa_or_block as get_physical_cpus_by_numa,
    )

logger = logging.getLogger(__name__)

THREADS_BIND_ENV = "SGLANG_CPU_OMP_THREADS_BIND"
NUMA_MAP_ENV = "SGLANG_OMNI_CPU_NUMA_MAP"
WIDEN_HOOK_MARKER = "omni_widen_hooked"

node_by_process_name: dict[str, int] = {}
node_usage: Counter[int] = Counter()
tp_group_nodes: dict[str, list[int]] = {}
node_assignment_lock = threading.Lock()


class NumaPlacementConflict(ValueError):
    """Two stages in one process were asked for different NUMA nodes."""


def parse_numa_map(raw: str | None) -> dict[str, tuple[int, ...]]:
    """Parse "thinker=0+1,talker=2" into {"thinker": (0, 1), "talker": (2,)}, one node per tp rank."""
    mapping: dict[str, tuple[int, ...]] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, separator, value = item.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not name or not value:
            raise ValueError(
                f"{NUMA_MAP_ENV} entries must look like '<stage>=<node>' or '<stage>=<node>+<node>', got {item!r}"
            )
        try:
            nodes = tuple(int(part) for part in value.split("+"))
        except ValueError as exc:
            raise ValueError(
                f"{NUMA_MAP_ENV} nodes for stage {name!r} must be integers joined by '+', got {value!r}"
            ) from exc
        if min(nodes) < 0:
            raise ValueError(
                f"{NUMA_MAP_ENV} nodes for stage {name!r} must be >= 0, got {value!r}"
            )
        mapping[name] = nodes
    return mapping


def numa_nodes() -> dict[int, str]:
    """CPU lists of the NUMA nodes that have CPUs, keyed by the node id lscpu reports, which need not be contiguous."""
    return {
        node: ",".join(map(str, sorted(cpus)))
        for node, cpus in sorted(get_physical_cpus_by_numa().items())
        if cpus
    }


def read_numa_nodes(consequence: str) -> dict[int, str] | None:
    """The NUMA topology, or None with a warning naming what is skipped when lscpu cannot be read."""
    try:
        return numa_nodes()
    except (RuntimeError, OSError) as exc:
        logger.warning("Could not read the NUMA topology (%s); %s.", exc, consequence)
        return None


def reset_assignment() -> None:
    """Forget recorded placements, for tests and for a changed topology or map."""
    with node_assignment_lock:
        node_by_process_name.clear()
        node_usage.clear()
        tp_group_nodes.clear()


def assign_node(
    process_name: str,
    stage_names: list[str],
    tp_size: int,
    tp_rank: int,
    user_map: Mapping[str, tuple[int, ...]],
    topology: Mapping[int, str],
) -> int:
    """Pick and record the node of one process; the caller holds node_assignment_lock."""
    node_ids = sorted(topology)
    group = ",".join(stage_names)
    reserved = {node for nodes in user_map.values() for node in nodes}
    free_nodes = [n for n in node_ids if n not in reserved] or node_ids
    by_load = sorted(free_nodes, key=lambda n: (node_usage[n], n))
    requested: dict[str, int] = {}
    for name in stage_names:
        # A replica instance such as talker@r0 takes its own entry, else its logical stage's.
        nodes = user_map.get(name) or user_map.get(parse_replica_instance_name(name)[0])
        if nodes is None:
            continue
        if len(nodes) not in (1, tp_size):
            raise ValueError(
                f"{NUMA_MAP_ENV} lists {len(nodes)} nodes for stage {name!r}, which has tp_size {tp_size}."
            )
        requested[name] = nodes[tp_rank] if len(nodes) > 1 else nodes[0]

    if not requested and tp_size > 1:
        planned = tp_group_nodes.get(group)
        if planned is None:
            planned = sorted(by_load[rank % len(by_load)] for rank in range(tp_size))
            tp_group_nodes[group] = planned
            if any(node_usage[node] for node in planned):
                logger.warning(
                    "Tensor-parallel ranks of %s share NUMA nodes with other worker processes; spread them with %s.",
                    group,
                    NUMA_MAP_ENV,
                )
            node_usage.update(planned)
        node = planned[tp_rank]
    else:
        for name, requested_node in requested.items():
            if requested_node not in topology:
                raise ValueError(
                    f"{NUMA_MAP_ENV} asks for node {requested_node} for stage {name!r}, but "
                    f"this host has NUMA node(s) {node_ids} with CPUs."
                )
        if len(set(requested.values())) > 1:
            raise NumaPlacementConflict(
                f"{NUMA_MAP_ENV} asks to split one process across NUMA nodes: "
                + ", ".join(
                    f"{name}->node {node}" for name, node in sorted(requested.items())
                )
                + ". Stages sharing a process share its binding."
            )
        node = next(iter(requested.values())) if requested else by_load[0]
        if node_usage[node]:
            logger.warning(
                "Process %s shares NUMA node %d with another worker process; "
                "their OMP threads are pinned to the same cores. Spread them with %s.",
                process_name,
                node,
                NUMA_MAP_ENV,
            )
        node_usage[node] += 1
    node_by_process_name[process_name] = node
    return node


def claim_explicit_nodes(process_name: str, env: Mapping[str, str]) -> None:
    """Count the nodes a user-set bind list covers, so auto placement does not stack other workers on them."""
    claimed = bound_cpus(env)
    if not claimed:
        return
    topology = read_numa_nodes("explicit binds are not counted")
    if topology is None:
        return
    covered = [node for node, ids in topology.items() if parse_cpu_list(ids) & claimed]
    with node_assignment_lock:
        if covered and process_name not in node_by_process_name:
            node_usage.update(covered)
            node_by_process_name[process_name] = covered[0]


def threads_bind_env(
    *,
    process_name: str,
    stage_names: list[str],
    tp_size: int = 1,
    tp_rank: int = 0,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the SGLANG_CPU_OMP_THREADS_BIND override for one process.

    CPU engines always hand SGLang gpu_id 0, so a single entry is enough. Each tp rank is its own process and gets its own node.
    """
    source_env = env if env is not None else os.environ
    topology = read_numa_nodes(f"leaving {THREADS_BIND_ENV} unset")
    if not topology:
        return {}
    if source_env.get(THREADS_BIND_ENV):
        claim_explicit_nodes(process_name, source_env)
        return {}

    user_map = parse_numa_map(source_env.get(NUMA_MAP_ENV))
    with node_assignment_lock:
        node = node_by_process_name.get(process_name)
        if node is None:
            node = assign_node(
                process_name,
                stage_names,
                tp_size,
                tp_rank,
                user_map,
                topology,
            )
    logger.info(
        "Binding CPU process %s (stages: %s) to NUMA node %d (of %d with CPUs).",
        process_name,
        ", ".join(stage_names) or "<none>",
        node,
        len(topology),
    )
    return {THREADS_BIND_ENV: topology[node]}


def bound_cpus(env: Mapping[str, str]) -> set[int] | None:
    """CPUs SGLang binds a worker to, its first bind entry, since CPU engines always run with gpu_id 0."""
    raw = env.get(THREADS_BIND_ENV, "").strip()
    if not raw or raw == "all":
        return None
    try:
        cpus = parse_cpu_list(raw.split("|")[0])
    except ValueError:
        return None
    return cpus or None


def apply_bound_affinity() -> None:
    """Move the worker onto its bound CPUs before anything creates threads, for workers that never start SGLang's own pinning."""
    cpus = bound_cpus(os.environ)
    # Pop even without a bind so the variable never leaks to child processes.
    launcher_cpus = os.environ.pop(LAUNCHER_CPUS_ENV, "")
    if cpus is None:
        return
    spawned_cpus = os.sched_getaffinity(0)
    allowed_cpus = parse_cpu_list(launcher_cpus) if launcher_cpus else spawned_cpus
    try:
        # SGLang caches the topology intersected with the current affinity, so cache it under the launcher's mask, which the spawn narrowed away.
        if launcher_cpus:
            os.sched_setaffinity(0, allowed_cpus)
        get_cpu_ids_by_node()
    except (RuntimeError, OSError) as exc:
        logger.warning("Could not cache the NUMA topology (%s).", exc)
        if launcher_cpus:
            os.sched_setaffinity(0, spawned_cpus)
        return
    target_cpus = cpus & allowed_cpus
    if not target_cpus:
        logger.warning(
            "The bound CPUs are outside the launcher's allowed CPUs; keeping the spawned mask."
        )
        os.sched_setaffinity(0, spawned_cpus)
        return
    try:
        os.sched_setaffinity(0, target_cpus)
    except OSError as exc:
        logger.warning("Could not apply the bound CPU range (%s).", exc)
        os.sched_setaffinity(0, spawned_cpus)


def node_core_count() -> int | None:
    """Cores of the smallest NUMA node, the most one bound process or one tp rank runs OMP threads on."""
    topology = read_numa_nodes("no per-node thread limit")
    if not topology:
        return None
    return min(len(parse_cpu_list(ids)) for ids in topology.values()) or None


def install_binding_hook() -> bool:
    """Keep SGLang's thread binding inside the worker's CPUs and thread budget.

    The bind list is cut to the allowed CPUs, and to OMP_NUM_THREADS of them, before SGLang pins OMP threads to it and sets the thread count to its length. init_cpu_threads_env also pins the main thread to a single core, and every thread created afterwards inherits that mask, so the main thread is widened back to the bound range. OMP workers keep their own per-core affinity, so that does not disturb them.
    """
    original = getattr(sglang_bootstrap, "_init_cpu_threads_env", None)
    if original is None:
        logger.warning(
            "SGLang has no _init_cpu_threads_env to hook; the main thread stays pinned to one core after thread binding."
        )
        return False
    if getattr(sglang_bootstrap, WIDEN_HOOK_MARKER, False):
        return False

    def widen_after_binding(
        *, local_omp_cpuid: str | list[int] | None, **kwargs
    ) -> None:
        if not local_omp_cpuid:
            original(local_omp_cpuid=local_omp_cpuid, **kwargs)
            return
        allowed_cpus = os.sched_getaffinity(0)
        cpus = (
            parse_cpu_list(local_omp_cpuid)
            if isinstance(local_omp_cpuid, str)
            else set(local_omp_cpuid)
        ) & allowed_cpus
        if not cpus:
            raise RuntimeError(
                f"The bound CPUs {local_omp_cpuid} are outside the CPUs this worker may use ({sorted(allowed_cpus)})."
            )
        # SGLang sets the OMP and torch thread count to the length of the list it pins, so hand it only as many CPUs as the thread budget.
        threads = os.environ.get("OMP_NUM_THREADS", "")
        pinned = (
            sorted(cpus)[: max(1, int(threads))] if threads.isdigit() else sorted(cpus)
        )
        original(
            local_omp_cpuid=(
                ",".join(map(str, pinned))
                if isinstance(local_omp_cpuid, str)
                else pinned
            ),
            **kwargs,
        )
        if torch.get_num_threads() != len(pinned):
            logger.warning(
                "SGLang left %d compute threads for %d pinned CPUs.",
                torch.get_num_threads(),
                len(pinned),
            )
        try:
            os.sched_setaffinity(0, cpus)
        except OSError as exc:
            logger.warning("Could not widen the CPU mask to the bound range (%s).", exc)

    sglang_bootstrap._init_cpu_threads_env = widen_after_binding
    setattr(sglang_bootstrap, WIDEN_HOOK_MARKER, True)
    return True
