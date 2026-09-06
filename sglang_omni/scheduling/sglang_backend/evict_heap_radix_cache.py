# SPDX-License-Identifier: Apache-2.0
"""A RadixCache whose eviction heap persists across evict() calls.

Order-equivalent to RadixCache for lru only; the factory falls back to
RadixCache for other policies.
"""

from __future__ import annotations

import heapq
import time

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode


class EvictHeapRadixCache(RadixCache):
    def __init__(self, params):
        self._evict_heap: list = []
        self._evict_heap_seq = 0
        super().__init__(params)

    def reset(self):
        self._evict_heap.clear()
        super().reset()

    def _evict_heap_push(self, node: TreeNode) -> None:
        self._evict_heap_seq += 1
        heapq.heappush(
            self._evict_heap,
            (self.eviction_strategy.get_priority(node), self._evict_heap_seq, node),
        )
        if len(self._evict_heap) > max(1024, 4 * len(self.evictable_leaves)):
            self._evict_heap_rebuild()

    def _evict_heap_rebuild(self) -> None:
        self._evict_heap = [
            (self.eviction_strategy.get_priority(n), i, n)
            for i, n in enumerate(self.evictable_leaves)
        ]
        self._evict_heap_seq = len(self._evict_heap)
        heapq.heapify(self._evict_heap)

    def _update_leaf_status(self, node: TreeNode) -> None:
        was_evictable = node in self.evictable_leaves
        super()._update_leaf_status(node)
        if not was_evictable and node in self.evictable_leaves:
            self._evict_heap_push(node)

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens

        num_evicted = 0
        while num_evicted < num_tokens and self._evict_heap:
            priority, _seq, x = heapq.heappop(self._evict_heap)

            if x not in self.evictable_leaves:
                continue
            current_priority = self.eviction_strategy.get_priority(x)
            if current_priority != priority:
                self._evict_heap_push(x)
                continue

            # Tree values are page-aligned copies of a kv row: page-exact segment.
            self.token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)
            num_evicted += len(x.value)
            # note (Junnan Li): _delete_leaf relands the parent via _update_leaf_status.
            self._delete_leaf(x)
            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)
