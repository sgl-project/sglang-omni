# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.comm import KVPool
from sglang_omni.proto import KVTransferPrepareMessage
from sglang_omni.scheduling.pd_kv_adapter import (
    AllocatorKVReceiver,
    build_kv_pool,
    resolve_page_indices,
)


class _FakeKVPool:
    def __init__(self, *, page_size: int, item_bytes: int, pages: int, buffers: int):
        self.page_size = page_size
        self._tensors = [
            torch.zeros((pages, item_bytes), dtype=torch.uint8) for _ in range(buffers)
        ]
        self._item_bytes = item_bytes

    def _pd_registerable_tensors(self):
        return tuple(self._tensors)

    def get_contiguous_buf_infos(self):
        return (
            [tensor.data_ptr() for tensor in self._tensors],
            [tensor.numel() for tensor in self._tensors],
            [self._item_bytes for _ in self._tensors],
        )


class _FakeAllocator:
    def __init__(self, capacity: int):
        self._free = list(range(capacity))

    def available_size(self) -> int:
        return len(self._free)

    def alloc(self, count: int):
        if len(self._free) < count:
            return None
        slots = self._free[:count]
        self._free = self._free[count:]
        return torch.tensor(slots, dtype=torch.int64)

    def free(self, slots: torch.Tensor) -> None:
        self._free.extend(int(slot) for slot in slots.tolist())


def _prepare(request_id: str, pages: int, pool_id: str = "decode"):
    return KVTransferPrepareMessage(
        request_id=request_id,
        transfer_id=f"{request_id}:transfer",
        from_stage="prefill",
        to_stage="decode",
        source_pool_id="prefill",
        target_pool_id=pool_id,
        source_page_indices=tuple(range(pages)),
        source_layout=None,
    )


def test_build_kv_pool_and_resolve_pages() -> None:
    raw = _FakeKVPool(page_size=1, item_bytes=8, pages=16, buffers=4)
    pool = build_kv_pool(raw, pool_id="prefill", layout_id="layout")
    assert isinstance(pool, KVPool)
    assert pool.page_size == 1
    assert len(pool.buffers) == 4

    mapping = torch.zeros((2, 8), dtype=torch.int32)
    mapping[1, :5] = torch.tensor([7, 8, 9, 3, 4])
    req_pool = type("ReqPool", (), {"req_to_token": mapping})()
    assert resolve_page_indices(req_pool, req_pool_idx=1, seq_len=5, page_size=1) == (
        7,
        8,
        9,
        3,
        4,
    )


def test_resolve_paged_indices_preserves_first_seen_order() -> None:
    mapping = torch.tensor([[0, 1, 2, 4, 5, 8]], dtype=torch.int32)
    req_pool = type("ReqPool", (), {"req_to_token": mapping})()
    assert resolve_page_indices(req_pool, req_pool_idx=0, seq_len=6, page_size=4) == (
        0,
        1,
        2,
    )


def test_receiver_keeps_commit_until_scheduler_takes_it() -> None:
    allocator = _FakeAllocator(8)
    receiver = AllocatorKVReceiver(
        pool_id="decode",
        token_to_kv_pool_allocator=allocator,
        page_size=1,
    )
    request = _prepare("r", 3)
    destination = receiver.reserve(request)
    receiver.commit(request, destination)
    assert allocator.available_size() == 5

    allocation = receiver.take_committed("r")
    assert allocation.page_indices == destination.page_indices
    assert receiver.release("r") is False


def test_receiver_abort_and_failed_callback_release_once() -> None:
    allocator = _FakeAllocator(8)
    receiver = AllocatorKVReceiver(
        pool_id="decode",
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        on_commit=lambda *_: (_ for _ in ()).throw(RuntimeError("admission")),
    )
    request = _prepare("r", 3)
    destination = receiver.reserve(request)
    with pytest.raises(RuntimeError, match="admission"):
        receiver.commit(request, destination)
    assert allocator.available_size() == 8
    receiver.abort(request, destination, RuntimeError("late"))
    assert allocator.available_size() == 8


def test_receiver_rejects_unsupported_and_invalid_reservations() -> None:
    allocator = _FakeAllocator(2)
    receiver = AllocatorKVReceiver(
        pool_id="decode", token_to_kv_pool_allocator=allocator, page_size=1
    )
    with pytest.raises(RuntimeError, match="exhausted"):
        receiver.reserve(_prepare("large", 3))
    with pytest.raises(ValueError):
        receiver.reserve(_prepare("wrong", 1, "other"))
    with pytest.raises(NotImplementedError):
        AllocatorKVReceiver(
            pool_id="decode", token_to_kv_pool_allocator=allocator, page_size=2
        )
