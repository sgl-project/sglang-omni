# SPDX-License-Identifier: Apache-2.0
"""Standalone planning CLI.

``python -m sglang_omni.cpu_alloc plan --replicas N [--gpu-id G]`` prints
shell-evalable ``CORE_BLOCKS`` / ``CLIENT_CPUS`` lines for the DP launchers,
replacing their naive 3/4 split with a NUMA/SMT-aware one: server blocks are
whole physical cores on the GPU's NUMA node, never splitting SMT siblings
across replicas.

``python -m sglang_omni.cpu_alloc topology`` dumps the discovered topology
as JSON for audits.
"""

from __future__ import annotations

import argparse
import json
import sys

from sglang_omni.cpu_alloc.topology import (
    CpuTopology,
    discover_topology,
    format_cpulist,
    gpu_numa_nodes,
)


def plan_replica_blocks(
    topology: CpuTopology,
    *,
    replicas: int,
    gpu_numa_node: int | None,
    server_share: float,
) -> dict:
    """Split whole physical cores into N server blocks plus a client pool."""
    if replicas < 1:
        raise ValueError(f"replicas must be >= 1, got {replicas}")
    if not 0.0 < server_share < 1.0:
        raise ValueError(f"server_share must be in (0, 1), got {server_share}")

    if gpu_numa_node is not None and topology.cores_on_node(gpu_numa_node):
        cores = list(topology.cores_on_node(gpu_numa_node))
        node_used = gpu_numa_node
    else:
        cores = list(topology.cores)
        node_used = None

    n_server = max(replicas, int(len(cores) * server_share))
    n_server = min(n_server, len(cores))
    if n_server < replicas:
        raise ValueError(
            f"need >= {replicas} physical cores for {replicas} replicas, "
            f"have {len(cores)} on node {node_used}"
        )
    server_cores, client_cores = cores[:n_server], cores[n_server:]

    base, extra = divmod(len(server_cores), replicas)
    blocks: list[list[int]] = []
    index = 0
    for replica in range(replicas):
        size = base + (1 if replica < extra else 0)
        block = server_cores[index : index + size]
        index += size
        blocks.append(sorted(c for core in block for c in core.cpu_ids))

    client_cpus = sorted(c for core in client_cores for c in core.cpu_ids)
    return {
        "numa_node": node_used,
        "server_blocks": blocks,
        "client_cpus": client_cpus,
    }


def format_blocks(result: dict) -> str:
    """Bare space-separated CORE_BLOCKS value for shell consumers."""
    return " ".join(
        ",".join(str(cpu) for cpu in block) for block in result["server_blocks"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sglang_omni.cpu_alloc")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="Plan DP replica core blocks.")
    plan_parser.add_argument("--replicas", type=int, required=True)
    plan_parser.add_argument("--gpu-id", type=int, default=None)
    plan_parser.add_argument("--server-share", type=float, default=0.75)
    plan_parser.add_argument("--json", action="store_true", dest="as_json")
    plan_parser.add_argument(
        "--format",
        choices=["shell", "blocks"],
        default="shell",
        help="shell prints CORE_BLOCKS/CLIENT_CPUS lines; blocks prints the bare CORE_BLOCKS value.",
    )

    sub.add_parser("topology", help="Dump the discovered CPU topology as JSON.")

    args = parser.parse_args(argv)
    topology = discover_topology()

    if args.command == "topology":
        print(json.dumps(topology.to_dict(), indent=2))
        return 0

    numa_node = None
    if args.gpu_id is not None:
        numa_node = gpu_numa_nodes([args.gpu_id]).get(args.gpu_id)
        if numa_node is None:
            print(
                f"warning: cannot resolve NUMA node for GPU {args.gpu_id}; "
                f"planning over the whole universe",
                file=sys.stderr,
            )
    try:
        result = plan_replica_blocks(
            topology,
            replicas=args.replicas,
            gpu_numa_node=numa_node,
            server_share=args.server_share,
        )
    except ValueError as exc:
        # Note (Jiaxin Deng): callers fall back to a NUMA-blind split when the
        # planner is missing, so a refusal needs its own code to stay a refusal.
        print(f"error: {exc}", file=sys.stderr)
        return 3
    if args.as_json:
        print(json.dumps(result, indent=2))
        return 0
    if args.format == "blocks":
        print(format_blocks(result))
        return 0
    print(f'CORE_BLOCKS="{format_blocks(result)}"')
    print(f'CLIENT_CPUS="{format_cpulist(result["client_cpus"])}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
