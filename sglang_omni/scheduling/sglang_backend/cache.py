"""Tree cache factory using upstream SGLang CacheInitParams."""

from __future__ import annotations

import dataclasses

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import get_memory, get_schedule, get_serving
from sglang.srt.session.streaming_session import StreamingSession

from sglang_omni.scheduling.sglang_backend.evict_heap_radix_cache import (
    EvictHeapRadixCache,
)


def create_tree_cache(
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    page_size: int,
) -> BasePrefixCache:
    """Select a base cache and wrap it when streaming sessions require it.

    Disabling radix selects ChunkCache; streaming may wrap that base cache.
    """
    cache_init_arguments = {
        "disable": get_memory().disable_radix_cache,
        "req_to_token_pool": req_to_token_pool,
        "token_to_kv_pool_allocator": token_to_kv_pool_allocator,
        "page_size": page_size,
        "chunked_prefill_size": get_schedule().chunked_prefill_size,
        "eviction_policy": get_memory().radix_eviction_policy,
    }
    if "eviction_policy_config" in {
        field.name for field in dataclasses.fields(CacheInitParams)
    }:
        cache_init_arguments["eviction_policy_config"] = (
            get_memory().radix_eviction_policy_config
        )
    else:
        pass
    params = CacheInitParams(**cache_init_arguments)

    if get_memory().disable_radix_cache:
        cache: BasePrefixCache = ChunkCache(params)
    elif params.eviction_policy.lower() == "lru":
        cache = EvictHeapRadixCache(params)
    else:
        cache = RadixCache(params)

    if (
        get_serving().enable_streaming_session
        and not cache.supports_streaming_session()
    ):
        return StreamingSession(cache)
    else:
        return cache
