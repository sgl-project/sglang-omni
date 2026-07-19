# SPDX-License-Identifier: Apache-2.0
"""GPU-direct round-trip tests for the CUDA-IPC relay.

The relay shares a CUDA buffer across processes via an IPC handle; the receiver
opens it and copies into its own buffer (a peer/NVLink copy when the GPUs
differ). These run two real processes because a process cannot open its own CUDA
IPC handle.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp

import pytest
import torch

from sglang_omni.relay.cuda_ipc import (
    CudaIpcPutOperation,
    CudaIpcRelay,
    _ContiguousSlotAllocator,
)

_N = 1024 * 1024  # 1 MiB payload


def _expected(n: int) -> torch.Tensor:
    return (torch.arange(n, dtype=torch.int64) % 251).to(torch.uint8)


def test_cuda_ipc_put_timeout_fails_relay_without_releasing_slot() -> None:
    released = False
    failed: list[BaseException] = []

    def release() -> None:
        nonlocal released
        released = True

    async def run() -> None:
        op = CudaIpcPutOperation(
            metadata={},
            ready_event=object(),  # type: ignore[arg-type]
            source_tensor=object(),  # type: ignore[arg-type]
            slot_index=0,
            request_id="r",
            size=1,
            release_cb=release,
            fail_cb=failed.append,
        )
        with pytest.raises(TimeoutError):
            await op.wait_for_completion(timeout=0.0)

    asyncio.run(run())

    assert released is False
    assert len(failed) == 1
    assert isinstance(failed[0], TimeoutError)


def test_cuda_ipc_relay_failure_wakes_blocked_slot_acquire() -> None:
    class BlockingAllocator:
        def __init__(self) -> None:
            self.released: list[tuple[int, int]] = []

        async def acquire_async(
            self, num_slots: int, *, capture_layout: bool = False
        ) -> int:
            await asyncio.Event().wait()
            return 0

        def release(self, offset: int, num_slots: int) -> None:
            self.released.append((offset, num_slots))

    async def run() -> BlockingAllocator:
        relay = CudaIpcRelay(engine_id="sender", device="cuda:0")
        allocator = BlockingAllocator()
        task = asyncio.create_task(relay._acquire_slots(allocator, 2))
        await asyncio.sleep(0)
        relay._mark_failed(TimeoutError("ack timeout"))
        with pytest.raises(RuntimeError, match="cuda_ipc relay failed"):
            _ = await task
        return allocator

    allocator = asyncio.run(run())
    assert allocator.released == []


def test_cuda_ipc_put_fails_fast_after_relay_failure() -> None:
    async def run() -> None:
        relay = CudaIpcRelay(engine_id="sender", device="cuda:0")
        relay._mark_failed(TimeoutError("ack timeout"))
        with pytest.raises(RuntimeError, match="cuda_ipc relay failed"):
            await relay.put_async(torch.zeros(1, dtype=torch.uint8))

    asyncio.run(run())


def test_cuda_ipc_default_pool_uses_small_slots() -> None:
    relay = CudaIpcRelay(engine_id="sender", device="cuda:0")
    assert relay.slot_size == 64 * 1024
    assert relay.pool_size == 1024 * 1024 * 1024
    assert relay.slot_count == 16 * 1024


def test_cuda_ipc_pool_size_and_slot_size_are_configurable() -> None:
    relay = CudaIpcRelay(
        engine_id="sender",
        device="cuda:0",
        pool_size_mb=1,
        slot_size_kb=256,
    )
    assert relay.slot_size == 256 * 1024
    assert relay.pool_size == 1024 * 1024
    assert relay.slot_count == 4


def test_contiguous_slot_allocator_waits_for_contiguous_range() -> None:
    async def run() -> None:
        allocator = _ContiguousSlotAllocator(slot_count=4, slot_size=8)
        first = (await allocator.acquire_async(1)).offset
        middle = (await allocator.acquire_async(1)).offset
        tail = (await allocator.acquire_async(1)).offset
        assert (first, middle, tail) == (0, 8, 16)

        allocator.release(middle, 1)
        blocked = asyncio.create_task(allocator.acquire_async(2, capture_layout=True))
        await asyncio.sleep(0)
        assert blocked.done() is False

        allocator.release(tail, 1)
        allocation = await asyncio.wait_for(blocked, timeout=1.0)
        assert allocation.offset == 8
        assert allocation.wait_rounds == 1
        assert allocation.last_failed_free_slots == 2
        assert allocation.last_failed_largest_free_run == 1
        allocator.release(first, 1)
        allocator.release(8, 2)

    asyncio.run(run())


def test_contiguous_slot_allocator_rejects_double_release() -> None:
    async def run() -> None:
        allocator = _ContiguousSlotAllocator(slot_count=2, slot_size=8)
        offset = (await allocator.acquire_async(2)).offset
        allocator.release(offset, 2)
        with pytest.raises(RuntimeError, match="released twice"):
            allocator.release(offset, 2)

    asyncio.run(run())


def _sender(
    src_gpu: int,
    meta_q: mp.Queue,
    ack_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    relay = None
    try:
        torch.cuda.set_device(src_gpu)
        relay = CudaIpcRelay(
            engine_id="sender",
            device=f"cuda:{src_gpu}",
            slot_size_mb=2,
        )
        buf = _expected(_N).to(f"cuda:{src_gpu}")

        async def run() -> None:
            op = await relay.put_async(
                buf,
                request_id="r",
                receiver_id="receiver",
            )
            meta_q.put(op.metadata)
            ack_status, ack_value = ack_q.get(timeout=60)
            if ack_status == "ok":
                op.mark_receiver_done()
            elif ack_status == "err":
                op.mark_receiver_failed(RuntimeError(ack_value))
            else:
                raise RuntimeError(
                    f"receiver returned invalid ACK status {ack_status!r}"
                )
            await op.wait_for_completion()

        asyncio.run(run())
        result_q.put(("sender", "ok", True))
    except Exception as exc:
        result_q.put(("sender", "err", repr(exc)))
    finally:
        if relay is not None:
            relay.close()


def _receiver(
    dst_gpu: int, meta_q: mp.Queue, ack_q: mp.Queue, result_q: mp.Queue
) -> None:
    relay = None
    try:
        torch.cuda.set_device(dst_gpu)
        relay = CudaIpcRelay(engine_id="receiver", device=f"cuda:{dst_gpu}")
        metadata = meta_q.get(timeout=60)

        async def run() -> torch.Tensor:
            size = metadata["transfer_info"]["size"]
            dest = torch.zeros(size, dtype=torch.uint8, device=f"cuda:{dst_gpu}")
            op = await relay.get_async(metadata, dest, request_id="r")
            await op.wait_for_completion()
            return dest

        dest = asyncio.run(run())
        expected = _expected(_N).to(f"cuda:{dst_gpu}")
        matches = bool(torch.equal(dest, expected))
        if not matches:
            raise AssertionError("received CUDA IPC bytes do not match the source")
        ack_q.put(("ok", None))
        result_q.put(("receiver", "ok", True))
    except Exception as exc:
        error = repr(exc)
        ack_q.put(("err", error))
        result_q.put(("receiver", "err", error))
    finally:
        if relay is not None:
            relay.close()


def _run_case(src_gpu: int, dst_gpu: int) -> None:
    ctx = mp.get_context("spawn")
    meta_q, ack_q, result_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    sender = ctx.Process(target=_sender, args=(src_gpu, meta_q, ack_q, result_q))
    receiver = ctx.Process(target=_receiver, args=(dst_gpu, meta_q, ack_q, result_q))
    sender.start()
    receiver.start()
    results = {}
    try:
        for _ in range(2):
            process, status, value = result_q.get(timeout=120)
            results[process] = (status, value)
    finally:
        sender.join(timeout=30)
        receiver.join(timeout=30)
        for proc in (sender, receiver):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=30)

    assert sender.exitcode == 0
    assert receiver.exitcode == 0
    assert results.keys() == {"sender", "receiver"}
    for process in ("sender", "receiver"):
        status, value = results[process]
        assert status == "ok", value
        assert value is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ipc_same_gpu_round_trip() -> None:
    _run_case(0, 0)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >= 2 GPUs for cross-GPU transfer"
)
def test_cuda_ipc_cross_gpu_round_trip() -> None:
    _run_case(0, 1)
