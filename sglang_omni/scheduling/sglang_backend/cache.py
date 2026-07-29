"""Tree cache factory using upstream SGLang CacheInitParams."""

from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.session.streaming_session import StreamingSession


def create_tree_cache(
    server_args,
    req_to_token_pool,
    token_to_kv_pool_allocator,
    page_size: int,
):
    """Create a tree cache based on server_args.

    When radix cache is disabled we always return ChunkCache so the scheduler
    keeps plain KV-cache semantics without any prefix matching.
    """
    params = CacheInitParams(
        disable=server_args.disable_radix_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=page_size,
        chunked_prefill_size=server_args.chunked_prefill_size,
    )

    if server_args.disable_radix_cache:
        from sglang.srt.mem_cache.chunk_cache import ChunkCache

        tree_cache = ChunkCache(params)
    else:
        tree_cache = RadixCache(params)

    # Match the cache composition used by SGLang's Scheduler.  The wrapper
    # keeps a session's request/KV-cache slot alive between append-only turns
    # while delegating ordinary requests to the underlying prefix cache.
    if (
        getattr(server_args, "enable_streaming_session", False)
        and not tree_cache.supports_streaming_session()
    ):
        tree_cache = StreamingSession(tree_cache)

    return tree_cache
