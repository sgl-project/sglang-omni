# SPDX-License-Identifier: Apache-2.0
"""Hardware tests for CudaIpcRelay — real GPU memory, real IPC handles.

Every test is marked ``accelerator`` and skipped when no CUDA device is
visible.  Cross-GPU tests require two devices.  Run with an explicit
``CUDA_VISIBLE_DEVICES`` and never use ``pytest -n auto``.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp

import pytest
import torch

from sglang_omni.comm.kv_transfer import KVBufferRegion, KVPool
from sglang_omni.relay.cuda_ipc import (
    CudaIpcRelay,
    _dump_cuda_storage_handle,
    _ensure_peer_access,
    _load_cuda_storage_handle,
    _slots_for_size,
)

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)
_requires_two_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs two CUDA devices",
)


def _relay(device: str = "cuda:0", pool_size_mb: int = 1) -> CudaIpcRelay:
    return CudaIpcRelay(
        engine_id="test",
        device=device,
        slot_size_kb=64,
        pool_size_mb=pool_size_mb,
    )


@pytest.fixture()
def ipc_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``Event.from_ipc_handle`` so that same-process get_async /
    get_kv_pages work without cross-process IPC.  The replacement
    synchronizes the current stream and returns a recorded event,
    making ``stream.wait_event(ready_event)`` safe.  Pool tensor caching
    for get_async is done in ``_prepare_get_bypass``."""

    @classmethod  # type: ignore[misc]
    def _fake_from_ipc(cls, device, handle):
        torch.cuda.synchronize(device)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(device))
        return ev

    monkeypatch.setattr(torch.cuda.Event, "from_ipc_handle", _fake_from_ipc)


def _prepare_get_bypass(relay: CudaIpcRelay, put_op) -> None:
    """Pre-cache the pool tensor so get_async skips IPC pool import."""
    pool_id = put_op.metadata["cuda_ipc"]["pool_id"]
    relay._remote_pools[pool_id] = relay._pool_tensor


# ---------------------------------------------------------------------------
# 1. Pool allocation on real GPU memory
# ---------------------------------------------------------------------------


@pytest.mark.accelerator
@_requires_cuda
def test_pool_allocates_real_gpu_memory() -> None:
    relay = _relay(pool_size_mb=1)
    relay._ensure_local_pool()
    pool = relay._pool_tensor
    assert pool is not None
    assert pool.is_cuda
    assert pool.numel() == relay.slot_count * relay.slot_size
    relay.close()


# ---------------------------------------------------------------------------
# 2. put_async copies bytes into the pool correctly
# ---------------------------------------------------------------------------


@pytest.mark.accelerator
@_requires_cuda
def test_put_copies_data_into_pool() -> None:
    async def _run() -> None:
        relay = _relay()
        src = torch.randint(0, 256, (1024,), dtype=torch.uint8, device="cuda:0")
        put_op = await relay.put_async(src, request_id="r1", receiver_id="peer")

        offset = put_op.metadata["transfer_info"]["offset"]
        size = put_op.metadata["transfer_info"]["size"]
        pool_slice = relay._pool_tensor[offset : offset + size]
        torch.cuda.synchronize()
        assert torch.equal(pool_slice, src.contiguous().view(torch.uint8).reshape(-1))

        put_op.mark_receiver_done()
        await put_op.wait_for_completion(timeout=5.0)
        relay.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. Multi-slot put copies data correctly
# ---------------------------------------------------------------------------


@pytest.mark.accelerator
@_requires_cuda
def test_multi_slot_put_copies_all_bytes() -> None:
    size = 200 * 1024
    assert _slots_for_size(size, 64 * 1024) > 1

    async def _run() -> None:
        relay = _relay()
        src = torch.randint(0, 256, (size,), dtype=torch.uint8, device="cuda:0")
        put_op = await relay.put_async(src, request_id="r2", receiver_id="peer")

        offset = put_op.metadata["transfer_info"]["offset"]
        pool_slice = relay._pool_tensor[offset : offset + size]
        torch.cuda.synchronize()
        assert torch.equal(pool_slice, src.contiguous().view(torch.uint8).reshape(-1))

        put_op.mark_receiver_done()
        await put_op.wait_for_completion(timeout=5.0)
        relay.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. Slots recycle after ACK
# ---------------------------------------------------------------------------


@pytest.mark.accelerator
@_requires_cuda
def test_slots_recycle_after_ack() -> None:
    async def _run() -> None:
        relay = _relay(pool_size_mb=1)
        slot_count = relay.slot_count
        slot_size = relay.slot_size

        for i in range(slot_count + 2):
            src = torch.full((slot_size,), i % 256, dtype=torch.uint8, device="cuda:0")
            put_op = await relay.put_async(src, request_id=f"r{i}", receiver_id="peer")
            offset = put_op.metadata["transfer_info"]["offset"]
            pool_slice = relay._pool_tensor[offset : offset + slot_size]
            torch.cuda.synchronize()
            assert torch.equal(pool_slice, src), f"mismatch on iteration {i}"

            put_op.mark_receiver_done()
            await put_op.wait_for_completion(timeout=5.0)
        relay.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. get_async copies from pool to destination (same process, IPC bypassed)
# ---------------------------------------------------------------------------


@pytest.mark.accelerator
@_requires_cuda
def test_get_copies_from_pool_to_destination(ipc_bypass) -> None:
    async def _run() -> None:
        relay = _relay()
        src = torch.randint(0, 256, (2048,), dtype=torch.uint8, device="cuda:0")
        dst = torch.zeros(2048, dtype=torch.uint8, device="cuda:0")

        put_op = await relay.put_async(src, request_id="r1", receiver_id="peer")
        _prepare_get_bypass(relay, put_op)

        get_op = await relay.get_async(put_op.metadata, dst, request_id="r1")
        await get_op.wait_for_completion(timeout=5.0)
        put_op.mark_receiver_done()
        await put_op.wait_for_completion(timeout=5.0)

        torch.cuda.synchronize()
        assert torch.equal(src, dst)
        relay.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. Concurrent transfers don't corrupt each other
# ---------------------------------------------------------------------------


@pytest.mark.accelerator
@_requires_cuda
def test_concurrent_transfers_no_corruption(ipc_bypass) -> None:
    n = 4

    async def _run() -> None:
        relay = _relay(pool_size_mb=2)
        pairs = []
        for i in range(n):
            src = torch.full((1024,), i + 1, dtype=torch.uint8, device="cuda:0")
            dst = torch.zeros_like(src)
            pairs.append((src, dst))

        put_ops = []
        for i, (src, _) in enumerate(pairs):
            put_op = await relay.put_async(src, request_id=f"c{i}", receiver_id="peer")
            _prepare_get_bypass(relay, put_op)
            put_ops.append(put_op)

        get_ops = []
        for i, (put_op, (_, dst)) in enumerate(zip(put_ops, pairs)):
            get_op = await relay.get_async(put_op.metadata, dst, request_id=f"c{i}")
            get_ops.append(get_op)

        await asyncio.gather(*(op.wait_for_completion(timeout=5.0) for op in get_ops))
        for put_op in put_ops:
            put_op.mark_receiver_done()
        await asyncio.gather(*(op.wait_for_completion(timeout=5.0) for op in put_ops))

        for i, (src, dst) in enumerate(pairs):
            assert torch.equal(src, dst), f"transfer {i} corrupted"
        relay.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 7. IPC handle export + import across processes
# ---------------------------------------------------------------------------


def _child_import_ipc(handle_dict: dict, result_queue: mp.Queue) -> None:
    try:
        torch.cuda.init()
        device = torch.device("cuda:0")
        tensor = _load_cuda_storage_handle(handle_dict, device=device)
        data = tensor[: handle_dict["numel"]].cpu().tolist()
        result_queue.put(("ok", data))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


@pytest.mark.accelerator
@_requires_cuda
def test_ipc_handle_roundtrip_cross_process() -> None:
    src = torch.arange(64, dtype=torch.uint8, device="cuda:0")
    handle = _dump_cuda_storage_handle(src)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child_import_ipc, args=(handle, q))
    p.start()
    p.join(timeout=30)
    assert not p.is_alive(), "child process hung"
    assert not q.empty(), "child produced no result"
    status, payload = q.get()
    assert status == "ok", f"child failed: {payload}"
    assert payload == list(range(64))


# ---------------------------------------------------------------------------
# 8. Full put/get across two processes
# ---------------------------------------------------------------------------


def _child_get(metadata: dict, size: int, result_queue: mp.Queue) -> None:
    try:
        torch.cuda.init()
        relay = CudaIpcRelay(engine_id="child", device="cuda:0", pool_size_mb=1)
        dst = torch.zeros(size, dtype=torch.uint8, device="cuda:0")

        async def _run():
            get_op = await relay.get_async(metadata, dst, request_id="xproc")
            await get_op.wait_for_completion(timeout=10.0)

        asyncio.run(_run())
        result_queue.put(("ok", dst.cpu().tolist()))
        relay.close()
    except Exception as exc:
        result_queue.put(("error", str(exc)))


@pytest.mark.accelerator
@_requires_cuda
def test_cross_process_put_get_roundtrip() -> None:
    async def _run() -> tuple:
        relay = _relay()
        src = torch.arange(128, dtype=torch.uint8, device="cuda:0")
        put_op = await relay.put_async(src, request_id="xproc", receiver_id="child")
        return relay, put_op, src

    relay, put_op, src = asyncio.run(_run())

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_child_get,
        args=(put_op.metadata, src.numel(), q),
    )
    p.start()
    p.join(timeout=30)

    async def _ack():
        put_op.mark_receiver_done()
        await put_op.wait_for_completion(timeout=5.0)

    asyncio.run(_ack())

    assert not p.is_alive(), "child process hung"
    assert not q.empty(), "child produced no result"
    status, payload = q.get()
    assert status == "ok", f"child failed: {payload}"
    assert payload == list(range(128))
    relay.close()


# ---------------------------------------------------------------------------
# 9. Peer access detection
# ---------------------------------------------------------------------------


@pytest.mark.accelerator
@_requires_cuda
def test_peer_access_same_device_always_true() -> None:
    assert _ensure_peer_access(0, 0) is True


@pytest.mark.accelerator
@_requires_two_gpus
def test_peer_access_returns_consistent_result() -> None:
    first = _ensure_peer_access(0, 1)
    second = _ensure_peer_access(0, 1)
    assert first == second
    assert isinstance(first, bool)


# ---------------------------------------------------------------------------
# 10. Cross-GPU transfer (same process, IPC bypassed)
# ---------------------------------------------------------------------------


@pytest.mark.accelerator
@_requires_two_gpus
def test_cross_gpu_put_and_read(ipc_bypass) -> None:
    if not torch.cuda.can_device_access_peer(1, 0):
        pytest.skip("GPU 1 cannot peer-access GPU 0")

    async def _run() -> None:
        sender = _relay(device="cuda:0")
        receiver = _relay(device="cuda:1")
        src = torch.randint(0, 256, (4096,), dtype=torch.uint8, device="cuda:0")
        put_op = await sender.put_async(src, request_id="xgpu", receiver_id="peer")

        pool_id = put_op.metadata["cuda_ipc"]["pool_id"]
        receiver._remote_pools[pool_id] = sender._pool_tensor

        dst = torch.zeros(4096, dtype=torch.uint8, device="cuda:1")
        get_op = await receiver.get_async(put_op.metadata, dst, request_id="xgpu")
        await get_op.wait_for_completion(timeout=5.0)
        put_op.mark_receiver_done()
        await put_op.wait_for_completion(timeout=5.0)
        assert torch.equal(src.cpu(), dst.cpu())
        sender.close()
        receiver.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 11. KV page scatter/gather with transfer_kv_per_layer_mla
# ---------------------------------------------------------------------------


@pytest.mark.accelerator
@_requires_cuda
def test_kv_page_scatter_gather_kernel() -> None:
    from sgl_kernel import transfer_kv_per_layer_mla

    num_pages = 8
    bytes_per_page = 64
    total = num_pages * bytes_per_page
    device = torch.device("cuda:0")

    src_buf = torch.arange(total, dtype=torch.uint8, device=device)
    dst_buf = torch.zeros(total, dtype=torch.uint8, device=device)

    src_indices = torch.tensor([0, 2, 5], dtype=torch.int64, device=device)
    dst_indices = torch.tensor([7, 3, 1], dtype=torch.int64, device=device)

    transfer_kv_per_layer_mla(
        src=src_buf,
        dst=dst_buf,
        src_indices=src_indices,
        dst_indices=dst_indices,
        item_size=bytes_per_page,
    )
    torch.cuda.synchronize()

    for si, di in zip([0, 2, 5], [7, 3, 1]):
        expected = src_buf[si * bytes_per_page : (si + 1) * bytes_per_page]
        actual = dst_buf[di * bytes_per_page : (di + 1) * bytes_per_page]
        assert torch.equal(expected, actual), f"page {si}->{di} mismatch"


# ---------------------------------------------------------------------------
# 12. Full paged KV path through CudaIpcRelay
# ---------------------------------------------------------------------------


def _gpu_kv_pool(
    pool_id: str,
    device: str = "cuda:0",
    num_pages: int = 8,
    bytes_per_page: int = 64,
) -> KVPool:
    total = num_pages * bytes_per_page
    tensor = torch.zeros(total, dtype=torch.uint8, device=device)
    return KVPool(
        pool_id=pool_id,
        layout_id="NHD",
        page_size=1,
        buffers=(KVBufferRegion("layer.0.kv", tensor, bytes_per_page=bytes_per_page),),
    )


@pytest.mark.accelerator
@_requires_cuda
def test_kv_pages_put_get_preserves_data(ipc_bypass) -> None:
    """Paged KV through real CudaIpcRelay on one GPU.  The relay is shared
    in-process so the IPC storage handles stay local, but the kernel copy
    and event synchronization are real."""

    async def _run() -> None:
        relay = _relay()
        src_pool = _gpu_kv_pool("src", num_pages=8, bytes_per_page=64)
        dst_pool = _gpu_kv_pool("dst", num_pages=8, bytes_per_page=64)

        src_data = torch.randint(0, 256, (8 * 64,), dtype=torch.uint8, device="cuda:0")
        src_pool.buffers[0].tensor.copy_(src_data)

        relay.register_kv_pool(src_pool)
        relay.register_kv_pool(dst_pool)

        dst_ref = relay.prepare_kv_destination("dst")
        put_op = await relay.put_kv_pages(
            source_pool_id="src",
            source_page_indices=(0, 1, 2, 3),
            destination_ref=dst_ref,
            transfer_id="kv-test",
        )

        kv_meta = put_op.metadata["cuda_ipc_kv"]
        reg_id = kv_meta["registration_id"]
        engine_id = put_op.metadata["engine_id"]
        relay._remote_kv_pools[(engine_id, reg_id)] = tuple(
            buf.byte_view() for buf in src_pool.buffers
        )

        get_op = await relay.get_kv_pages(
            put_op.metadata,
            destination_pool_id="dst",
            source_page_indices=(0, 1, 2, 3),
            destination_page_indices=(4, 5, 6, 7),
            request_id="r-kv",
            transfer_id="kv-test",
        )
        await get_op.wait_for_completion(timeout=5.0)
        put_op.mark_receiver_done()
        await put_op.wait_for_completion(timeout=5.0)

        torch.cuda.synchronize()
        for src_idx, dst_idx in zip([0, 1, 2, 3], [4, 5, 6, 7]):
            src_page = src_pool.buffers[0].byte_view()[
                src_idx * 64 : (src_idx + 1) * 64
            ]
            dst_page = dst_pool.buffers[0].byte_view()[
                dst_idx * 64 : (dst_idx + 1) * 64
            ]
            assert torch.equal(
                src_page, dst_page
            ), f"page {src_idx}->{dst_idx} mismatch"
        relay.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 13. Trace events fire on real transfers
# ---------------------------------------------------------------------------


@pytest.mark.accelerator
@_requires_cuda
def test_real_transfer_emits_trace_events(
    ipc_bypass, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit_test.fixtures.trace_capture import capture_comm_trace

    with capture_comm_trace(monkeypatch) as events:

        async def _run() -> None:
            relay = _relay()
            src = torch.randint(0, 256, (1024,), dtype=torch.uint8, device="cuda:0")
            dst = torch.zeros_like(src)
            put_op = await relay.put_async(src, request_id="tr", receiver_id="peer")
            _prepare_get_bypass(relay, put_op)

            get_op = await relay.get_async(put_op.metadata, dst, request_id="tr")
            await get_op.wait_for_completion(timeout=5.0)
            put_op.mark_receiver_done()
            await put_op.wait_for_completion(timeout=5.0)
            relay.close()

        asyncio.run(_run())

    event_names = [e["event"] for e in events]
    assert "cuda_ipc_pool_alloc" in event_names
    assert "cuda_ipc_put_async" in event_names
    assert "cuda_ipc_get_async" in event_names
    assert "cuda_ipc_get_wait_copy" in event_names
    assert "cuda_ipc_put_wait_ack" in event_names
