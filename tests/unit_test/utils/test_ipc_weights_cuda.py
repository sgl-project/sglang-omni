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

from sglang_omni.comm import stage_io
from sglang_omni.pipeline.stage_workers import (
    StageLaunchConfig,
    StageWorkerProcessSpec,
    stage_process_main,
)
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


def _direct_ipc_producer(data_queue: Any, done: Any) -> None:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    torch.cuda.set_device(0)
    monkey_patch_torch_reductions()
    tensor = torch.arange(8, dtype=torch.float32, device="cuda")
    data_queue.put(stage_io.serialize_direct_cuda_ipc_stream_chunk(tensor, None))
    _wait(done, "direct CUDA IPC consumer")


def _direct_ipc_consumer_factory(data_queue: Any, done: Any) -> None:
    torch.cuda.set_device(0)
    ref = data_queue.get(timeout=60)
    tensor, metadata = stage_io.deserialize_direct_cuda_ipc_stream_chunk(ref)
    assert torch.equal(tensor, torch.arange(8, dtype=torch.float32, device="cuda"))
    assert metadata is None
    done.set()
    raise SystemExit(0)


def test_weight_share_stage_bootstrap_supports_direct_cuda_ipc(
    monkeypatch,
) -> None:
    monkeypatch.setenv(ipc_weights.ENV_WEIGHT_SHARE, "follower:/unused")
    context = mp.get_context("spawn")
    data_queue = context.Queue()
    done = context.Event()
    ready = context.Event()
    consumer_spec = StageWorkerProcessSpec(
        process_name="direct-ipc-consumer",
        stage_specs=[
            StageLaunchConfig(
                stage_name="consumer",
                factory=f"{__name__}._direct_ipc_consumer_factory",
                factory_args={"data_queue": data_queue, "done": done},
            )
        ],
    )
    processes = [
        context.Process(target=_direct_ipc_producer, args=(data_queue, done)),
        context.Process(
            target=stage_process_main,
            args=(consumer_spec, ready),
        ),
    ]

    for process in processes:
        process.start()
    for process in reversed(processes):
        process.join(120)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
    data_queue.close()
    data_queue.join_thread()

    assert [process.exitcode for process in processes] == [0, 0]


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


class _ScratchTiny(nn.Module):
    """Backbone stand-in plus a MOSS-style per-step staging scratch."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=False)
        self.scratch = nn.Embedding(4, 8)


_SCRATCH_PRIVATE = frozenset({"scratch.weight"})


def _scratch_handle(store_dir: Path) -> str:
    return str(store_dir / "_ScratchTiny.weights-ipc")


def _private_leader(
    store_dir: Path, ready: Any, b_wrote: Any, a_wrote: Any, done: Any
) -> None:
    torch.cuda.set_device(0)
    model = _ScratchTiny().cuda()
    with torch.no_grad():
        model.fc.weight.copy_(
            torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8)
        )
        model.scratch.weight.fill_(1.0)
    ipc_weights.export_weights(
        model,
        _scratch_handle(store_dir),
        validate_secure=False,
        private_names=_SCRATCH_PRIVATE,
    )
    ready.set()

    # Forced interleaving: replica A holds request-A staging (1.0) while
    # replica B stages request B; A's data must survive B's write.
    _wait(b_wrote, "follower scratch write")
    torch.cuda.synchronize()
    assert torch.all(model.scratch.weight == 1.0).item()
    with torch.no_grad():
        model.scratch.weight.fill_(3.0)
    torch.cuda.synchronize()
    a_wrote.set()
    _wait(done, "follower completion")


def _private_follower(
    store_dir: Path, ready: Any, b_wrote: Any, a_wrote: Any, done: Any
) -> None:
    torch.cuda.set_device(0)
    _wait(ready, "leader publication")

    model = _ScratchTiny().cuda()
    with torch.no_grad():
        model.fc.weight.zero_()
        model.scratch.weight.zero_()
    own_scratch_ptr = model.scratch.weight.data_ptr()
    record = ipc_weights.attach_weights(
        model,
        _scratch_handle(store_dir),
        timeout_s=30,
        validate_secure=False,
        private_names=_SCRATCH_PRIVATE,
    )
    ipc_weights.verify_attachment(model, record)

    # Backbone is a true zero-copy alias; the scratch kept this process's own
    # storage but carries the leader's exported bytes.
    expected = torch.arange(32, dtype=torch.float32, device="cuda").reshape(4, 8)
    assert torch.equal(model.fc.weight, expected)
    assert model.scratch.weight.data_ptr() == own_scratch_ptr
    assert torch.all(model.scratch.weight == 1.0).item()

    with torch.no_grad():
        model.scratch.weight.fill_(2.0)
    torch.cuda.synchronize()
    b_wrote.set()

    _wait(a_wrote, "leader scratch write")
    torch.cuda.synchronize()
    assert torch.all(model.scratch.weight == 2.0).item()
    done.set()


def test_cross_process_private_scratch_stays_isolated(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    ready, b_wrote, a_wrote, done = (context.Event() for _ in range(4))
    args = (tmp_path, ready, b_wrote, a_wrote, done)
    processes = [
        context.Process(target=_private_leader, args=args),
        context.Process(target=_private_follower, args=args),
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


def _corrupt_leader(store_dir: Path, ready: Any, b_wrote: Any, done: Any) -> None:
    torch.cuda.set_device(0)
    model = _ScratchTiny().cuda()
    with torch.no_grad():
        model.scratch.weight.fill_(1.0)
    ipc_weights.export_weights(model, _scratch_handle(store_dir), validate_secure=False)
    ready.set()

    _wait(b_wrote, "follower scratch write")
    torch.cuda.synchronize()
    # The whole point: with the scratch IPC-shared (no policy), the other
    # replica's staging write IS visible here. This arm proves the isolation
    # test above can detect the corruption it guards against.
    assert torch.all(model.scratch.weight == 2.0).item()
    done.set()


def _corrupt_follower(store_dir: Path, ready: Any, b_wrote: Any, done: Any) -> None:
    torch.cuda.set_device(0)
    _wait(ready, "leader publication")
    model = _ScratchTiny().cuda()
    ipc_weights.attach_weights(
        model, _scratch_handle(store_dir), timeout_s=30, validate_secure=False
    )
    with torch.no_grad():
        model.scratch.weight.fill_(2.0)
    torch.cuda.synchronize()
    b_wrote.set()
    _wait(done, "leader check")


def test_cross_process_shared_scratch_shows_the_hazard(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    ready, b_wrote, done = (context.Event() for _ in range(3))
    args = (tmp_path, ready, b_wrote, done)
    processes = [
        context.Process(target=_corrupt_leader, args=args),
        context.Process(target=_corrupt_follower, args=args),
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
