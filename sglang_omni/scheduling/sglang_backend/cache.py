"""Tree cache factory using upstream SGLang CacheInitParams."""

from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams

from sglang_omni.scheduling.sglang_backend.evict_heap_radix_cache import (
    EvictHeapRadixCache,
)


def create_tree_cache(
    server_args,
    req_to_token_pool,
    token_to_kv_pool_allocator,
    page_size: int,
):
    """Create a tree cache based on server_args.

    When radix cache is disabled we always return ChunkCache so the scheduler
    keeps plain KV-cache semantics without any prefix matching. Non-lru
    eviction policies fall back to the upstream RadixCache.
    """
    params = CacheInitParams(
        disable=server_args.disable_radix_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=page_size,
        chunked_prefill_size=server_args.chunked_prefill_size,
        eviction_policy=getattr(server_args, "radix_eviction_policy", "lru"),
    )

    if server_args.disable_radix_cache:
        from sglang.srt.mem_cache.chunk_cache import ChunkCache

        tree_cache = ChunkCache(params)
    elif params.eviction_policy.lower() == "lru":
        tree_cache = EvictHeapRadixCache(params)
    else:
        from sglang.srt.mem_cache.radix_cache import RadixCache

        tree_cache = RadixCache(params)

    if (
        getattr(server_args, "enable_streaming_session", False)
        and not tree_cache.supports_streaming_session()
    ):
        from sglang.srt.session.streaming_session import StreamingSession

        tree_cache = StreamingSession(tree_cache)

    return tree_cache
