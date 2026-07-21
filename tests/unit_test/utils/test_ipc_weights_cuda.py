# SPDX-License-Identifier: Apache-2.0
"""Cross-process CUDA IPC aliasing test for same-GPU weight sharing.

The CPU protocol tests use an identity serializer in one process; this proves
the real property: a leader's in-place write lands on a follower in a separate
process through the shared CUDA storage. A ``copy_``-instead-of-alias
regression would read the pre-mutation values and fail here.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from sglang_omni.utils import ipc_weights

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="weight-share CUDA test requires CUDA"
)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=False)


def _wait(event: Any, name: str) -> None:
    assert event.wait(60), f"timeout waiting for {name}"


def _handle(store_dir: Path) -> str:
    return str(store_dir / "_Tiny.weights-ipc")


def _leader(store_dir: Path, ready: Any, aliased: Any, mutated: Any, done: Any) -> None:
    torch.cuda.set_device(0)
    model = _Tiny().cuda()
    with torch.no_grad():
        model.fc.weight.copy_(
            torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8)
        )
    ipc_weights.export_weights(model, _handle(store_dir), validate_secure=False)
    ready.set()

    _wait(aliased, "follower alias")
    with torch.no_grad():
        model.fc.weight.fill_(17.0)
    torch.cuda.synchronize()
    mutated.set()
    _wait(done, "follower completion")


def _follower(
    store_dir: Path, ready: Any, aliased: Any, mutated: Any, done: Any
) -> None:
    torch.cuda.set_device(0)
    _wait(ready, "leader publication")

    model = _Tiny().cuda()
    with torch.no_grad():
        model.fc.weight.zero_()
    ipc_weights.attach_weights(
        model, _handle(store_dir), timeout_s=30, validate_secure=False
    )

    expected = torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8)
    inputs = torch.arange(16, dtype=torch.float32, device="cuda").reshape(2, 8)
    assert torch.equal(model.fc.weight, expected)
    assert torch.equal(model.fc(inputs), inputs @ expected.T)
    aliased.set()

    _wait(mutated, "leader mutation")
    torch.cuda.synchronize()
    assert torch.all(model.fc.weight == 17.0).item()
    done.set()


def test_cross_process_alias_observes_leader_mutation(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    ready, aliased, mutated, done = (context.Event() for _ in range(4))
    args = (tmp_path, ready, aliased, mutated, done)
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
