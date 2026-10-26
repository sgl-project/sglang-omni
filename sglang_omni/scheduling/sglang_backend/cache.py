"""Tree cache factory using upstream SGLang CacheInitParams."""

from __future__ import annotations

from sglang.srt.managers import schedule_batch
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_exec, get_memory, get_schedule, get_serving

from sglang_omni.scheduling.sglang_backend.evict_heap_radix_cache import (
    EvictHeapRadixCache,
)
from sglang_omni.vendor.sglang.core import set_mamba_track_indices


def create_tree_cache(
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    page_size: int,
    sliding_window_size: int | None,
) -> BasePrefixCache:
    """Select the pools' cache: upstream's for radix-disabled and session stages."""
    if not get_memory().disable_radix_cache and isinstance(
        req_to_token_pool, HybridReqToTokenPool
    ):
        schedule_batch.set_mamba_track_indices_from_reqs = set_mamba_track_indices
    else:
        pass
    params = CacheInitParams(
        disable=get_memory().disable_radix_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=page_size,
        sliding_window_size=sliding_window_size,
        chunked_prefill_size=get_schedule().chunked_prefill_size,
        eviction_policy=get_memory().radix_eviction_policy,
        eviction_policy_config=get_memory().radix_eviction_policy_config,
        enable_mamba_extra_buffer=get_exec().mamba.enable_mamba_extra_buffer,
        enable_mamba_extra_buffer_lazy=get_exec().mamba.enable_mamba_extra_buffer_lazy,
    )

    if (
        get_memory().disable_radix_cache
        and not get_serving().enable_streaming_session
        and isinstance(token_to_kv_pool_allocator, PureSWATokenToKVPoolAllocator)
    ):
        return PureSWARadixCache(params)
    elif (
        get_memory().disable_radix_cache
        or get_serving().enable_streaming_session
        or isinstance(req_to_token_pool, HybridReqToTokenPool)
    ):
        # note (ratish): streaming sessions live inside the unified cache, so a
        # session stage takes it whether or not its radix tree is enabled.
        tree_components = [ComponentType.FULL]
        if isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
            tree_components.append(ComponentType.SWA)
        else:
            pass
        if isinstance(req_to_token_pool, HybridReqToTokenPool):
            tree_components.append(ComponentType.MAMBA)
        else:
            pass
        params.tree_components = tuple(tree_components)
        return UnifiedRadixCache(params)
    elif params.eviction_policy.lower() == "lru":
        return EvictHeapRadixCache(params)
    else:
        return RadixCache(params)
