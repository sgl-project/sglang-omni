# SPDX-License-Identifier: Apache-2.0
"""Capture an all-reduce in an XPU graph on every rank, replay it, and report what each rank holds.

    python -m tests.test_ci.xpu_model.captured_all_reduce <world_size> <port>

Rank 0 prints one line, RESULT followed by JSON: the oneCCL the ranks mapped, and per
rank and size the distinct values left after the replay and after an eager all-reduce.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES = {"4 KiB": 2048, "2 MiB": 1 << 20}


def distinct(tensor: torch.Tensor) -> list[float]:
    values = tensor.float()
    return sorted({values.min().item(), values.max().item()})


def oneccl_version() -> str:
    try:
        return importlib.metadata.version("oneccl")
    except importlib.metadata.PackageNotFoundError:
        return "not pip-installed"


def run_rank(rank: int, world_size: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", rank=rank, world_size=world_size)
    held = {}
    for label, numel in SIZES.items():
        captured = torch.zeros(numel, dtype=torch.bfloat16, device="xpu")
        dist.all_reduce(captured)
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            dist.all_reduce(captured)
        torch.xpu.synchronize()
        captured.fill_(float(rank + 1))
        graph.replay()
        eager = torch.full(
            (numel,), float(rank + 1), dtype=torch.bfloat16, device="xpu"
        )
        dist.all_reduce(eager)
        torch.xpu.synchronize()
        held[label] = {"replay": distinct(captured), "eager": distinct(eager)}
    libs = sorted(
        {line.split()[-1] for line in open("/proc/self/maps") if "libccl.so" in line}
    )
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, held)
    if rank == 0:
        result = {"oneccl": oneccl_version(), "libccl": libs, "ranks": gathered}
        print("RESULT " + json.dumps(result), flush=True)
    else:
        pass
    dist.destroy_process_group()


def main() -> None:
    world_size, port = int(sys.argv[1]), int(sys.argv[2])
    mp.spawn(run_rank, args=(world_size, port), nprocs=world_size)


if __name__ == "__main__":
    main()
