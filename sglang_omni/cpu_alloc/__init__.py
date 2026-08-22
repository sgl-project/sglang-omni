# SPDX-License-Identifier: Apache-2.0
"""Topology-aware CPU allocation for pipeline stage processes."""

from sglang_omni.cpu_alloc.allocator import (
    CpuAllocationPlan,
    ProcessCpuAssignment,
    ProcessCpuDemand,
    allocate,
)
from sglang_omni.cpu_alloc.cost import StageCpuCost, resolve_stage_cpu_costs
from sglang_omni.cpu_alloc.topology import (
    CpuTopology,
    PhysicalCore,
    discover_topology,
    format_cpulist,
    gpu_numa_nodes,
    parse_cpulist,
)

__all__ = [
    "CpuAllocationPlan",
    "CpuTopology",
    "PhysicalCore",
    "ProcessCpuAssignment",
    "ProcessCpuDemand",
    "StageCpuCost",
    "allocate",
    "discover_topology",
    "format_cpulist",
    "gpu_numa_nodes",
    "parse_cpulist",
    "resolve_stage_cpu_costs",
]
