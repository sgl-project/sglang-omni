# SPDX-License-Identifier: Apache-2.0
"""Serialize KV allocation between the two threads a Decode half runs.

``TokenToKVPoolAllocator.alloc`` reads ``free_pages``, slices the leading
entries, and writes the remainder back. Nothing in
``sglang/srt/mem_cache/allocator`` holds a lock, which is correct while one
thread owns the allocator.

A PD Decode half has two. The scheduler thread allocates for decode steps.
The comm event loop allocates inside ``prepare_kv_receive`` ->
``DecodeKVReceiver.reserve``, which is handed the same allocator instance.
Two callers that interleave between the read and the write receive the same
slots.

Measured on two H200s with both calls recorded: one 3,000-request run logged
21,309 slots handed to one thread while the other still held them, and the
requests that received them returned empty completions.

This wraps the allocator so both callers take one lock. Everything else is
delegated untouched, so the allocator's own behaviour is unchanged.
"""

from __future__ import annotations

import threading
from typing import Any


class LockedKVAllocator:
    """Delegate to *inner*, holding a lock across ``alloc`` and ``free``.

    Only the two methods that mutate the free list are guarded. The lock is
    uncontended on the scheduler thread except when a transfer is reserving,
    so the cost on the decode path is one uncontended acquire per step.
    """

    def __init__(self, inner: Any) -> None:
        # Note (Audrey Zheng): set through __dict__ because __setattr__ below
        # forwards to the wrapped allocator.
        self.__dict__["_inner"] = inner
        self.__dict__["_alloc_lock"] = threading.Lock()

    def alloc(self, need_size: int) -> Any:
        with self.__dict__["_alloc_lock"]:
            return self.__dict__["_inner"].alloc(need_size)

    def free(self, free_index: Any) -> Any:
        with self.__dict__["_alloc_lock"]:
            return self.__dict__["_inner"].free(free_index)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_inner"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.__dict__["_inner"], name, value)

    def __repr__(self) -> str:
        return f"LockedKVAllocator({self.__dict__['_inner']!r})"
