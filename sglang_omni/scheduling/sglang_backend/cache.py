"""Tree cache factory using upstream SGLang CacheInitParams."""

from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.runtime_context import get_memory, get_schedule

from sglang_omni.scheduling.sglang_backend.evict_heap_radix_cache import (
    EvictHeapRadixCache,
)


def create_tree_cache(
    req_to_token_pool,
    token_to_kv_pool_allocator,
    page_size: int,
):
    """Create a tree cache from the published config.

    When radix cache is disabled we always return ChunkCache so the scheduler
    keeps plain KV-cache semantics without any prefix matching. Non-lru
    eviction policies fall back to the upstream RadixCache.
    """
    params = CacheInitParams(
        disable=get_memory().disable_radix_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=page_size,
        chunked_prefill_size=get_schedule().chunked_prefill_size,
        eviction_policy=get_memory().radix_eviction_policy,
    )

    if get_memory().disable_radix_cache:
        from sglang.srt.mem_cache.chunk_cache import ChunkCache

        return ChunkCache(params)

    if params.eviction_policy.lower() == "lru":
        return EvictHeapRadixCache(params)

    from sglang.srt.mem_cache.radix_cache import RadixCache

    return RadixCache(params)
