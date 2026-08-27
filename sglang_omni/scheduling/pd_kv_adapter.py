# SPDX-License-Identifier: Apache-2.0
"""Model-neutral SGLang adapters for Omni's paged KV transfer API.

The transfer engine owns destination allocations through ``reserve`` and
``commit``.  A committed allocation remains owned by this adapter until the
decode scheduler atomically takes it for PREBUILT admission.  This extra seam
is important: a local KV commit is not, by itself, scheduler admission.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang_omni.comm import KVBufferRegion, KVPageDestination, KVPool
from sglang_omni.proto import KVTransferPrepareMessage

logger = logging.getLogger(__name__)


def build_kv_pool(
    token_to_kv_pool: Any,
    *,
    pool_id: str,
    layout_id: str,
) -> KVPool:
    """Expose an SGLang token-to-KV pool through the comm KVPool contract."""

    registerable = _pd_registerable_tensors(token_to_kv_pool)
    _, _, item_lens = token_to_kv_pool.get_contiguous_buf_infos()
    if len(registerable) != len(item_lens):
        raise ValueError(
            "SGLang KV pool exposed "
            f"{len(registerable)} tensors but {len(item_lens)} descriptors"
        )

    buffers = tuple(
        KVBufferRegion(
            name=f"kv_buffer.{index}",
            tensor=tensor,
            bytes_per_page=int(item_len),
        )
        for index, (tensor, item_len) in enumerate(zip(registerable, item_lens))
    )
    if not buffers:
        raise ValueError(f"SGLang KV pool {pool_id!r} exposed no buffers")

    return KVPool(
        pool_id=pool_id,
        layout_id=layout_id,
        page_size=int(token_to_kv_pool.page_size),
        buffers=buffers,
    )


def _pd_registerable_tensors(token_to_kv_pool: Any) -> tuple[Any, ...]:
    getter = getattr(token_to_kv_pool, "_pd_registerable_tensors", None)
    if callable(getter):
        return tuple(getter())

    tensors: list[Any] = []
    for layer_id in range(int(token_to_kv_pool.layer_num)):
        tensors.append(token_to_kv_pool.get_key_buffer(layer_id))
        tensors.append(token_to_kv_pool.get_value_buffer(layer_id))
    return tuple(tensors)


def resolve_page_indices(
    req_to_token_pool: Any,
    *,
    req_pool_idx: int,
    seq_len: int,
    page_size: int,
) -> tuple[int, ...]:
    """Return request KV pages in logical token order."""

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    slots = req_to_token_pool.req_to_token[req_pool_idx, :seq_len].tolist()
    if page_size == 1:
        return tuple(int(slot) for slot in slots)

    pages: list[int] = []
    seen: set[int] = set()
    for slot in slots:
        page = int(slot) // page_size
        if page not in seen:
            seen.add(page)
            pages.append(page)
    return tuple(pages)


@dataclass(frozen=True)
class ReservedAllocation:
    slots: torch.Tensor
    page_indices: tuple[int, ...]
    seq_len: int


class AllocatorKVReceiver:
    """KVReceiver backed by an SGLang token-to-KV allocator.

    PR3 deliberately supports only the baseline page-size-one path.  The
    allocation is retained after ``commit`` until ``take_committed`` transfers
    ownership to the decode request lifecycle.
    """

    def __init__(
        self,
        *,
        pool_id: str,
        token_to_kv_pool_allocator: Any,
        page_size: int,
        on_commit: Callable[[str, ReservedAllocation], None] | None = None,
    ) -> None:
        if page_size != 1:
            raise NotImplementedError(
                "PD runtime currently requires page_size == 1; " f"got {page_size}"
            )
        self.pool_id = pool_id
        self._allocator = token_to_kv_pool_allocator
        self._on_commit = on_commit
        self._lock = threading.RLock()
        self._reservations: dict[str, ReservedAllocation] = {}
        self._committed: dict[str, ReservedAllocation] = {}

    def reserve(self, request: KVTransferPrepareMessage) -> KVPageDestination:
        if request.target_pool_id != self.pool_id:
            raise ValueError(
                f"receiver for {self.pool_id!r} got {request.target_pool_id!r}"
            )
        count = len(request.source_page_indices)
        if count <= 0:
            raise ValueError("KV transfer requested zero pages")

        with self._lock:
            if (
                request.request_id in self._reservations
                or request.request_id in self._committed
            ):
                raise RuntimeError(
                    f"duplicate KV reservation for {request.request_id!r}"
                )
            available = int(self._allocator.available_size())
            if available < count:
                raise RuntimeError(
                    f"decode KV pool exhausted: need {count}, have {available}"
                )
            slots = self._allocator.alloc(count)
            if slots is None:
                raise RuntimeError(f"decode KV allocator failed to allocate {count}")
            allocation = ReservedAllocation(
                slots=slots,
                page_indices=tuple(int(slot) for slot in slots.tolist()),
                seq_len=count,
            )
            self._reservations[request.request_id] = allocation
            return KVPageDestination(self.pool_id, allocation.page_indices)

    def commit(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination,
    ) -> None:
        with self._lock:
            allocation = self._reservations.pop(request.request_id, None)
            if allocation is None:
                raise RuntimeError(
                    f"commit for unknown reservation {request.request_id!r}"
                )
            if tuple(destination.page_indices) != allocation.page_indices:
                self._allocator.free(allocation.slots)
                raise RuntimeError(
                    f"commit pages differ for request {request.request_id!r}"
                )
            self._committed[request.request_id] = allocation

        try:
            if self._on_commit is not None:
                self._on_commit(request.request_id, allocation)
        except Exception:
            self.release(request.request_id)
            raise

    def take_committed(self, request_id: str) -> ReservedAllocation:
        """Atomically transfer committed pages to the scheduler lifecycle."""

        with self._lock:
            allocation = self._committed.pop(request_id, None)
            if allocation is None:
                raise RuntimeError(
                    f"no committed KV allocation for request {request_id!r}"
                )
            return allocation

    def release(self, request_id: str) -> bool:
        """Free an adapter-owned reservation/commit exactly once."""

        with self._lock:
            allocation = self._reservations.pop(request_id, None)
            if allocation is None:
                allocation = self._committed.pop(request_id, None)
        if allocation is None:
            return False
        self._allocator.free(allocation.slots)
        return True

    def abort(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination | None,
        error: BaseException,
    ) -> None:
        del destination
        released = self.release(request.request_id)
        logger.warning(
            "PD KV receive aborted request=%s released=%s error=%s",
            request.request_id,
            released,
            error,
        )


class SGLangKVPageLease:
    """Keep Prefill KV alive until the receiver ACK releases the source."""

    def __init__(self, req: Any, tree_cache: Any) -> None:
        self._req = req
        self._tree_cache = tree_cache
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            req = self._req
            if req is None:
                return
            self._req = None
        from sglang.srt.mem_cache.common import release_kv_cache

        release_kv_cache(req, self._tree_cache)
