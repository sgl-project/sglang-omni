# SPDX-License-Identifier: Apache-2.0
"""CUDA integration test for cross-process weight aliasing."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from sglang_omni.distributed.weight_ipc import (
    WeightIpcStore,
    export_shared_weights,
    import_and_alias,
)
from sglang_omni.distributed.weight_ipc.cuda_handles import allocation_offset_bytes

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="weight IPC CUDA tests require CUDA"
)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


def _wait(event: Any, name: str) -> None:
    assert event.wait(60), f"timeout waiting for {name}"


def _leader(
    store_dir: Path,
    ready: Any,
    aliased: Any,
    mutated: Any,
    done: Any,
) -> None:
    torch.cuda.set_device(0)
    model = _Tiny().cuda()
    with torch.no_grad():
        model.fc.weight.copy_(
            torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8)
        )

    bundle = export_shared_weights(
        model,
        model_path="model",
        model_revision="revision",
    )
    WeightIpcStore(store_dir).write_bundle(bundle)
    ready.set()

    _wait(aliased, "follower alias")
    with torch.no_grad():
        model.fc.weight.fill_(17.0)
    torch.cuda.synchronize()
    mutated.set()
    _wait(done, "follower completion")


def _follower(
    store_dir: Path,
    ready: Any,
    aliased: Any,
    mutated: Any,
    done: Any,
) -> None:
    torch.cuda.set_device(0)
    _wait(ready, "leader publication")

    store = WeightIpcStore(store_dir)
    store.wait_ready(30)
    bundle = store.load_bundle()
    model = _Tiny().cuda()
    with torch.no_grad():
        model.fc.weight.zero_()
    import_and_alias(
        model,
        bundle,
        model_path="model",
        model_revision="revision",
    )

    expected_weight = torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8)
    inputs = torch.arange(16, dtype=torch.float32, device="cuda").reshape(2, 8)
    assert torch.equal(model.fc.weight, expected_weight)
    assert torch.equal(model(inputs), inputs @ expected_weight.T)
    aliased.set()

    _wait(mutated, "leader mutation")
    torch.cuda.synchronize()
    assert torch.all(model.fc.weight == 17.0).item()
    done.set()


def test_cross_process_alias_observes_leader_mutation(tmp_path: Path) -> None:
    allocation = torch.empty(256 * 1024, dtype=torch.float32, device="cuda:0")
    assert allocation_offset_bytes(allocation[4096 : 4096 + 128]) != 0

    context = mp.get_context("spawn")
    ready, aliased, mutated, done = (context.Event() for _ in range(4))
    args = (tmp_path / "weight_ipc", ready, aliased, mutated, done)
    processes = [
        context.Process(target=_leader, args=args),
        context.Process(target=_follower, args=args),
    ]

    for process in processes:
        process.start()
    for process in reversed(processes):
        process.join(120)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()

    assert [process.exitcode for process in processes] == [0, 0]
