"""Tree cache factory using upstream SGLang CacheInitParams."""

from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.runtime_context import get_memory, get_schedule, get_serving

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

        cache = ChunkCache(params)
    elif params.eviction_policy.lower() == "lru":
        cache = EvictHeapRadixCache(params)
    else:
        from sglang.srt.mem_cache.radix_cache import RadixCache

        cache = RadixCache(params)

    if (
        get_serving().enable_streaming_session
        and not cache.supports_streaming_session()
    ):
        from sglang.srt.session.streaming_session import StreamingSession

        cache = StreamingSession(cache)
    return cache
