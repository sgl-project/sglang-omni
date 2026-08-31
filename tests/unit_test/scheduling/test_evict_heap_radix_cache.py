# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random

import torch
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

from sglang_omni.scheduling.sglang_backend.evict_heap_radix_cache import (
    EvictHeapRadixCache,
)


class _MockAllocator:
    device = "cpu"

    def free(self, value):
        pass

    def free_segment(self, value, start_pos=0):
        pass

    def available_size(self):
        return 1 << 30


def _make(cache_cls, eviction_policy="lru"):
    """Simulated-cache builder; create_simulated hardcodes RadixCache."""
    return cache_cls(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=_MockAllocator(),
            page_size=1,
            enable_kv_cache_events=False,
            eviction_policy=eviction_policy,
        )
    )


def _run_trace(cache, seed: int, steps: int = 4000, drain: bool = True) -> list:
    """Drive an identical insert/lock/unlock/evict trace; return eviction order."""
    order = []
    orig_delete = cache._delete_leaf
    cache._delete_leaf = lambda node: (
        order.append((node.key.extra_key, tuple(node.key.token_ids))),
        orig_delete(node),
    )[1]

    rng = random.Random(seed)
    tok = 0
    locked = []
    for _ in range(steps):
        op = rng.random()
        if op < 0.62:
            length = rng.randint(1, 12)
            key = RadixKey(
                token_ids=[rng.randint(0, 30) for _ in range(length)],
                extra_key=str(rng.randint(0, 40)),
            )
            value = torch.arange(tok, tok + length)
            tok += length
            cache.insert(InsertParams(key=key, value=value))
        elif op < 0.72 and cache.evictable_leaves:
            node = rng.choice(sorted(cache.evictable_leaves, key=lambda x: x.id))
            cache.inc_lock_ref(node)
            locked.append(node)
        elif op < 0.82 and locked:
            cache.dec_lock_ref(locked.pop(rng.randrange(len(locked))))
        else:
            cache.evict(EvictParams(num_tokens=rng.randint(1, 40)))
    if drain:
        cache.evict(EvictParams(num_tokens=1 << 20))
    return order


def test_eviction_trace_matches_stock():
    for seed in (1234, 99, 2026):
        stock = _make(RadixCache)
        patched = _make(EvictHeapRadixCache)
        stock_order = _run_trace(stock, seed)
        patched_order = _run_trace(patched, seed)
        assert patched_order == stock_order
        assert len(patched.evictable_leaves) == len(stock.evictable_leaves)


def test_factory_selects_evict_heap_only_for_lru():
    from types import SimpleNamespace

    from sglang_omni.scheduling.sglang_backend.cache import create_tree_cache

    def build(policy):
        args = SimpleNamespace(
            disable_radix_cache=False,
            chunked_prefill_size=None,
            radix_eviction_policy=policy,
        )
        return create_tree_cache(args, None, _MockAllocator(), 1)

    assert type(build("lru")) is EvictHeapRadixCache
    for policy in ("mru", "priority", "lfu", "fifo", "filo"):
        assert type(build(policy)) is RadixCache, policy


def test_heap_stays_bounded_and_recovers():
    cache = _make(EvictHeapRadixCache)
    _run_trace(cache, seed=7, steps=2000, drain=False)
    assert cache.evictable_leaves
    assert len(cache._evict_heap) <= max(1024, 4 * len(cache.evictable_leaves))
    cache.evict(EvictParams(num_tokens=1 << 20))
    # The cache keeps working after a full drain.
    key = RadixKey(token_ids=[1, 2, 3], extra_key="post")
    cache.insert(InsertParams(key=key, value=torch.arange(3)))
    result = cache.evict(EvictParams(num_tokens=1 << 20))
    assert result.num_tokens_evicted >= 3
    assert not cache.evictable_leaves


def test_reset_then_reuse():
    cache = _make(EvictHeapRadixCache)
    cache.insert(
        InsertParams(
            key=RadixKey(token_ids=[5, 6, 7], extra_key="r"), value=torch.arange(3)
        )
    )
    cache.reset()
    cache.insert(
        InsertParams(
            key=RadixKey(token_ids=[8, 9], extra_key="r2"), value=torch.arange(2)
        )
    )
    result = cache.evict(EvictParams(num_tokens=1 << 20))
    assert result.num_tokens_evicted == 2
    assert not cache.evictable_leaves
