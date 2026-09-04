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
import queue
import traceback
from contextlib import suppress
from multiprocessing.connection import Connection

import pytest
import torch

from sglang_omni.comm.kv_transfer import KVBufferRegion, KVPool
from sglang_omni.relay.cuda_ipc import (
    CudaIpcPutOperation,
    CudaIpcRelay,
    _ContiguousSlotAllocator,
)

_N = 1024 * 1024  # 1 MiB payload
_PAGED_PAGE_COUNT = 9
_PAGED_SOURCE_INDICES = (7, 1, 5, 3)
_PAGED_DESTINATION_INDICES = (2, 8, 0, 6)
_PAGED_BUFFER_SPECS = (
    ("keys", torch.float16, 40, 4, 6),
    ("values", torch.float32, 72, 2, 3),
)
_PAGED_OUTER_GUARD = 252
_PAGED_DESTINATION_GUARD = 253
_PAGED_MESSAGE_TIMEOUT_S = 120
_PAGED_RESULT_TIMEOUT_S = 240


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


def _paged_pattern(
    buffer_index: int,
    bytes_per_page: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    byte_count = _PAGED_PAGE_COUNT * bytes_per_page
    offsets = torch.arange(byte_count, dtype=torch.int64, device=device)
    page_indices = offsets // bytes_per_page
    byte_offsets = offsets % bytes_per_page
    return ((byte_offsets * 17 + page_indices * 43 + buffer_index * 67) % 251).to(
        torch.uint8
    )


def _make_paged_pool(
    pool_id: str,
    *,
    device: torch.device,
    source: bool,
) -> tuple[KVPool, tuple[torch.Tensor, ...]]:
    buffers = []
    backings = []
    for buffer_index, (
        name,
        dtype,
        bytes_per_page,
        prefix_elements,
        suffix_elements,
    ) in enumerate(_PAGED_BUFFER_SPECS):
        element_size = torch.empty((), dtype=dtype).element_size()
        region_bytes = _PAGED_PAGE_COUNT * bytes_per_page
        assert region_bytes % element_size == 0
        region_elements = region_bytes // element_size
        backing = torch.empty(
            prefix_elements + region_elements + suffix_elements,
            dtype=dtype,
            device=device,
        )
        backing.view(torch.uint8).fill_(_PAGED_OUTER_GUARD)
        tensor = backing.narrow(0, prefix_elements, region_elements)
        region = KVBufferRegion(
            name=name,
            tensor=tensor,
            bytes_per_page=bytes_per_page,
        )
        assert region.tensor.storage_offset() == prefix_elements
        assert region.byte_view().storage_offset() == prefix_elements * element_size
        if source:
            region.byte_view().copy_(
                _paged_pattern(
                    buffer_index,
                    bytes_per_page,
                    device=device,
                )
            )
        else:
            region.byte_view().fill_(_PAGED_DESTINATION_GUARD)
        buffers.append(region)
        backings.append(backing)
    return (
        KVPool(
            pool_id=pool_id,
            layout_id="paged-relay-test-v1",
            page_size=16,
            buffers=tuple(buffers),
        ),
        tuple(backings),
    )


def _assert_bytes_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    context: str,
) -> None:
    if torch.equal(actual, expected):
        return
    mismatch = int(torch.nonzero(actual != expected, as_tuple=False)[0].item())
    raise AssertionError(
        f"{context} differs at byte {mismatch}: "
        f"got {int(actual[mismatch].item())}, "
        f"expected {int(expected[mismatch].item())}"
    )


def _assert_bytes_are(
    actual: torch.Tensor,
    expected: int,
    *,
    context: str,
) -> None:
    mismatches = torch.nonzero(actual != expected, as_tuple=False)
    if mismatches.numel() == 0:
        return
    mismatch = int(mismatches[0].item())
    raise AssertionError(
        f"{context} changed at byte {mismatch}: "
        f"got {int(actual[mismatch].item())}, expected {expected}"
    )


def _assert_outer_guards(
    pool: KVPool,
    backings: tuple[torch.Tensor, ...],
) -> None:
    for region, backing in zip(pool.buffers, backings, strict=True):
        backing_bytes = backing.view(torch.uint8).view(-1)
        start = region.byte_view().storage_offset()
        end = start + region.byte_view().numel()
        _assert_bytes_are(
            backing_bytes[:start],
            _PAGED_OUTER_GUARD,
            context=f"{region.name} prefix guard",
        )
        _assert_bytes_are(
            backing_bytes[end:],
            _PAGED_OUTER_GUARD,
            context=f"{region.name} suffix guard",
        )


def _assert_source_pool_unchanged(
    pool: KVPool,
    backings: tuple[torch.Tensor, ...],
) -> None:
    for buffer_index, region in enumerate(pool.buffers):
        _assert_bytes_equal(
            region.byte_view(),
            _paged_pattern(
                buffer_index,
                region.bytes_per_page,
                device=pool.device,
            ),
            context=f"source buffer {region.name}",
        )
    _assert_outer_guards(pool, backings)


def _assert_destination_mapping(
    pool: KVPool,
    backings: tuple[torch.Tensor, ...],
) -> None:
    source_for_destination = dict(
        zip(
            _PAGED_DESTINATION_INDICES,
            _PAGED_SOURCE_INDICES,
            strict=True,
        )
    )
    for buffer_index, region in enumerate(pool.buffers):
        actual_pages = region.byte_view().view(
            _PAGED_PAGE_COUNT,
            region.bytes_per_page,
        )
        source_pages = _paged_pattern(
            buffer_index,
            region.bytes_per_page,
            device=pool.device,
        ).view(_PAGED_PAGE_COUNT, region.bytes_per_page)
        selected_source_pages = source_pages[list(_PAGED_SOURCE_INDICES)]
        for index, page in enumerate(selected_source_pages):
            for other_page in selected_source_pages[index + 1 :]:
                assert not torch.equal(page, other_page)
        for destination_page in range(_PAGED_PAGE_COUNT):
            source_page = source_for_destination.get(destination_page)
            if source_page is None:
                _assert_bytes_are(
                    actual_pages[destination_page],
                    _PAGED_DESTINATION_GUARD,
                    context=(
                        f"destination buffer {region.name} unmapped page "
                        f"{destination_page}"
                    ),
                )
                continue
            _assert_bytes_equal(
                actual_pages[destination_page],
                source_pages[source_page],
                context=(
                    f"destination buffer {region.name} page {destination_page} "
                    f"mapped from source page {source_page}"
                ),
            )
    _assert_outer_guards(pool, backings)


def _recv_paged_message(connection: Connection, expected: str) -> object:
    if not connection.poll(_PAGED_MESSAGE_TIMEOUT_S):
        raise TimeoutError(f"timed out waiting for paged IPC message {expected!r}")
    kind, value = connection.recv()
    if kind == "error":
        raise RuntimeError(f"paged IPC peer failed:\n{value}")
    if kind != expected:
        raise RuntimeError(
            f"expected paged IPC message {expected!r}, received {kind!r}"
        )
    return value


def _wait_for_receiver_close(
    connection: Connection,
) -> tuple[str | None, str | None]:
    peer_error = None
    while True:
        if not connection.poll(_PAGED_MESSAGE_TIMEOUT_S):
            raise TimeoutError("receiver did not close its imported CUDA handles")
        kind, value = connection.recv()
        if kind == "receiver_closed":
            return value, peer_error
        if kind == "error":
            peer_error = str(value)
            continue
        if kind == "ack":
            continue
        raise RuntimeError(f"expected receiver cleanup message, received {kind!r}")


def _paged_sender(gpu: int, connection: Connection, result_q: mp.Queue) -> None:
    relay = None
    source_shared = False
    error: str | None = None
    try:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
        relay = CudaIpcRelay(
            engine_id="paged-sender",
            device=str(device),
            pool_size_mb=1,
        )
        pool, backings = _make_paged_pool(
            "source-pool",
            device=device,
            source=True,
        )
        relay.register_kv_pool(pool)
        destination_ref = _recv_paged_message(connection, "destination")

        async def run() -> None:
            nonlocal source_shared
            op = await relay.put_kv_pages(
                source_pool_id=pool.pool_id,
                source_page_indices=_PAGED_SOURCE_INDICES,
                destination_ref=destination_ref,
            )
            connection.send(("metadata", op.metadata))
            source_shared = True
            _recv_paged_message(connection, "ack")
            op.mark_receiver_done()
            await op.wait_for_completion(timeout=_PAGED_MESSAGE_TIMEOUT_S)

        asyncio.run(run())
        _assert_source_pool_unchanged(pool, backings)
        connection.send(("sender_done", None))
    except Exception:
        error = traceback.format_exc()
        with suppress(Exception):
            connection.send(("error", error))
    finally:
        if source_shared:
            try:
                close_error, peer_error = _wait_for_receiver_close(connection)
                if peer_error is not None and error is None:
                    error = f"receiver failed:\n{peer_error}"
                if close_error is not None and error is None:
                    error = f"receiver relay cleanup failed:\n{close_error}"
            except Exception:
                if error is None:
                    error = traceback.format_exc()
        if relay is not None:
            try:
                relay.close()
            except Exception:
                if error is None:
                    error = traceback.format_exc()
        connection.close()
        result_q.put(
            ("sender", "ok", True) if error is None else ("sender", "err", error)
        )


def _paged_receiver(gpu: int, connection: Connection, result_q: mp.Queue) -> None:
    relay = None
    error: str | None = None
    try:
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
        relay = CudaIpcRelay(
            engine_id="paged-receiver",
            device=str(device),
            pool_size_mb=1,
        )
        pool, backings = _make_paged_pool(
            "destination-pool",
            device=device,
            source=False,
        )
        relay.register_kv_pool(pool)
        connection.send(("destination", relay.prepare_kv_destination(pool.pool_id)))
        metadata = _recv_paged_message(connection, "metadata")
        expected_size = len(_PAGED_SOURCE_INDICES) * sum(
            bytes_per_page for _, _, bytes_per_page, _, _ in _PAGED_BUFFER_SPECS
        )
        assert metadata["transfer_info"]["size"] == expected_size
        storages = metadata["cuda_ipc_kv"]["storages"]
        assert len(storages) == len(_PAGED_BUFFER_SPECS)
        assert all(int(storage["tensor_offset"]) > 0 for storage in storages)

        async def run() -> None:
            op = await relay.get_kv_pages(
                metadata,
                destination_pool_id=pool.pool_id,
                source_page_indices=_PAGED_SOURCE_INDICES,
                destination_page_indices=_PAGED_DESTINATION_INDICES,
                request_id="paged-round-trip",
            )
            await op.wait_for_completion(timeout=_PAGED_MESSAGE_TIMEOUT_S)

        asyncio.run(run())
        _assert_destination_mapping(pool, backings)
        connection.send(("ack", None))
        _recv_paged_message(connection, "sender_done")
    except Exception:
        error = traceback.format_exc()
        with suppress(Exception):
            connection.send(("error", error))
    finally:
        close_error = None
        if relay is not None:
            try:
                relay.close()
            except Exception:
                close_error = traceback.format_exc()
                if error is None:
                    error = close_error
        with suppress(Exception):
            connection.send(("receiver_closed", close_error))
        connection.close()
        result_q.put(
            ("receiver", "ok", True) if error is None else ("receiver", "err", error)
        )


def _run_paged_case(gpu: int) -> None:
    assert _PAGED_SOURCE_INDICES != _PAGED_DESTINATION_INDICES
    assert len(_PAGED_SOURCE_INDICES) == len(_PAGED_DESTINATION_INDICES)
    assert len(set(_PAGED_SOURCE_INDICES)) == len(_PAGED_SOURCE_INDICES)
    assert len(set(_PAGED_DESTINATION_INDICES)) == len(_PAGED_DESTINATION_INDICES)
    assert all(0 <= index < _PAGED_PAGE_COUNT for index in _PAGED_SOURCE_INDICES)
    assert all(0 <= index < _PAGED_PAGE_COUNT for index in _PAGED_DESTINATION_INDICES)
    ctx = mp.get_context("spawn")
    sender_connection, receiver_connection = ctx.Pipe(duplex=True)
    result_q = ctx.Queue()
    sender = ctx.Process(
        target=_paged_sender,
        args=(gpu, sender_connection, result_q),
    )
    receiver = ctx.Process(
        target=_paged_receiver,
        args=(gpu, receiver_connection, result_q),
    )
    sender.start()
    receiver.start()
    sender_connection.close()
    receiver_connection.close()
    results: dict[str, tuple[str, object]] = {}
    try:
        try:
            for _ in range(2):
                process, status, value = result_q.get(timeout=_PAGED_RESULT_TIMEOUT_S)
                assert process not in results, f"duplicate result from {process}"
                results[process] = (status, value)
        except queue.Empty as exc:
            raise AssertionError(
                "timed out waiting for paged CUDA IPC workers; "
                f"sender(exit={sender.exitcode}, alive={sender.is_alive()}), "
                f"receiver(exit={receiver.exitcode}, alive={receiver.is_alive()}), "
                f"results={results}"
            ) from exc
    finally:
        sender.join(timeout=30)
        receiver.join(timeout=30)
        for process in (sender, receiver):
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)
        result_q.close()
        result_q.join_thread()

    assert sender.exitcode == 0
    assert receiver.exitcode == 0
    assert results.keys() == {"sender", "receiver"}
    for process in ("sender", "receiver"):
        status, value = results[process]
        assert status == "ok", value
        assert value is True


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ipc_same_gpu_round_trip() -> None:
    _run_case(0, 0)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ipc_paged_multi_buffer_round_trip_with_storage_offsets() -> None:
    _run_paged_case(0)


@pytest.mark.accelerator
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >= 2 GPUs for cross-GPU transfer"
)
def test_cuda_ipc_cross_gpu_round_trip() -> None:
    _run_case(0, 1)
