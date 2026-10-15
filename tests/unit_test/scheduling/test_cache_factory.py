# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from array import array
from collections.abc import Iterator
from contextlib import AbstractContextManager
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import PureSWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, EvictParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.scheduling.sglang_backend.cache import create_tree_cache
from sglang_omni.scheduling.sglang_backend.evict_heap_radix_cache import (
    EvictHeapRadixCache,
)


def test_factory_selects_evict_heap_only_for_lru() -> None:
    def build(policy: str) -> BasePrefixCache:
        with get_context().override_server_args(
            disable_radix_cache=False,
            chunked_prefill_size=None,
            radix_eviction_policy=policy,
        ):
            return create_tree_cache(
                None, Mock(spec=BaseTokenToKVPoolAllocator, device="cpu"), 1, None
            )

    assert type(build("lru")) is EvictHeapRadixCache
    for policy in ("mru", "priority", "lfu", "fifo", "filo"):
        assert type(build(policy)) is RadixCache, policy


def test_factory_passes_the_eviction_policy_config_to_the_strategy() -> None:
    if "eviction_policy_config" not in {
        field.name for field in dataclasses.fields(CacheInitParams)
    }:
        pytest.skip("CacheInitParams has no eviction_policy_config")
    else:
        pass

    with get_context().override_server_args(
        disable_radix_cache=False,
        chunked_prefill_size=None,
        radix_eviction_policy="slru",
        radix_eviction_policy_config={"protected_threshold": 4},
    ):
        cache = create_tree_cache(
            None, Mock(spec=BaseTokenToKVPoolAllocator, device="cpu"), 1, None
        )

    assert cache.eviction_strategy.protected_threshold == 4


@pytest.mark.parametrize("eviction_policy", ["lru", "fifo"])
def test_factory_refuses_radix_for_hybrid_state_pools(eviction_policy: str) -> None:
    """A radix hit restores attention KV but never the linear-attention state."""
    hybrid_pool = object.__new__(HybridReqToTokenPool)
    with get_context().override_server_args(
        disable_radix_cache=False,
        chunked_prefill_size=None,
        radix_eviction_policy=eviction_policy,
    ):
        with pytest.raises(ValueError, match="disable_radix_cache"):
            create_tree_cache(
                hybrid_pool,
                Mock(spec=BaseTokenToKVPoolAllocator, device="cpu"),
                1,
                None,
            )


def test_factory_keeps_chunk_cache_for_hybrid_state_pools() -> None:
    hybrid_pool = object.__new__(HybridReqToTokenPool)
    with get_context().override_server_args(
        disable_radix_cache=True,
        chunked_prefill_size=100,
    ):
        cache = create_tree_cache(
            hybrid_pool, Mock(spec=BaseTokenToKVPoolAllocator, device="cpu"), 1, None
        )

    assert type(cache) is ChunkCache


def make_window_cache(
    position_capacity: int, window_positions: int, *, radix: bool = False
) -> tuple[ReqToTokenPool, PureSWATokenToKVPoolAllocator, SWAKVPool, BasePrefixCache]:
    request_pool = ReqToTokenPool(
        1, position_capacity, "cpu", enable_memory_saver=False
    )
    kv_pool = SWAKVPool(
        size=0,
        size_swa=position_capacity,
        page_size=1,
        dtype=torch.float32,
        head_num=1,
        head_dim=8,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[],
        device="cpu",
    )
    allocator = PureSWATokenToKVPoolAllocator(
        size_swa=position_capacity,
        page_size=1,
        dtype=torch.float32,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    with get_context().override_server_args(
        disable_radix_cache=not radix,
        chunked_prefill_size=None,
        enable_streaming_session=False,
    ):
        cache = create_tree_cache(request_pool, allocator, 1, window_positions)
    return request_pool, allocator, kv_pool, cache


def test_factory_preserves_window_eviction_capability() -> None:
    _, _, _, cache = make_window_cache(32, 8)
    assert cache.supports_swa()
    assert cache.sliding_window_size == 8
    with get_context().override_server_args(
        disable_radix_cache=True,
        chunked_prefill_size=None,
        enable_streaming_session=False,
    ):
        ordinary_cache = create_tree_cache(
            None, Mock(spec=BaseTokenToKVPoolAllocator, device="cpu"), 1, None
        )
    assert isinstance(ordinary_cache, ChunkCache)
    assert not ordinary_cache.supports_swa()


def test_factory_selects_window_radix_cache_for_window_pools() -> None:
    _, _, _, cache = make_window_cache(32, 8, radix=True)
    assert type(cache) is PureSWARadixCache
    assert cache.supports_swa()
    assert cache.sliding_window_size == 8


def window_runtime(*, disable_radix_cache: bool) -> AbstractContextManager[None]:
    return get_context().override_server_args(
        attention_backend="torch_native",
        prefill_attention_backend="torch_native",
        decode_attention_backend="torch_native",
        disable_radix_cache=disable_radix_cache,
        chunked_prefill_size=None,
        enable_streaming_session=False,
        disaggregation_mode=None,
        speculative_algorithm=None,
        strip_thinking_cache=False,
    )


@pytest.fixture
def window_cache_runtime() -> Iterator[None]:
    with window_runtime(disable_radix_cache=True):
        yield


@pytest.fixture
def window_radix_runtime() -> Iterator[None]:
    with window_runtime(disable_radix_cache=False):
        yield


@pytest.mark.usefixtures("window_cache_runtime")
@pytest.mark.parametrize("cache_on_release", [False, True])
def test_window_slots_survive_reuse_and_release(
    monkeypatch: pytest.MonkeyPatch, cache_on_release: bool
) -> None:
    monkeypatch.setenv("SGLANG_SWA_EVICTION_INTERVAL", "4")
    position_capacity, position_count, window_positions = 32, 24, 8
    request_pool, allocator, kv_pool, cache = make_window_cache(
        position_capacity, window_positions
    )
    request = Req(
        rid="window-request",
        origin_input_text="",
        origin_input_ids=array("q", range(position_count)),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    assert request_pool.alloc([request]) is not None
    slots = allocator.alloc(position_count)
    assert slots is not None
    request_pool.req_to_token[request.kv.req_pool_idx, :position_count] = slots.to(
        torch.int32
    )
    request.kv.kv_committed_len = position_count
    request.kv.kv_allocated_len = position_count
    request.decode_batch_idx = 1
    expected_keys = torch.arange(position_count * 8, dtype=torch.float32).reshape(
        position_count, 1, 8
    )
    kv_pool.get_key_buffer(0)[slots] = expected_keys
    batch = ScheduleBatch(
        reqs=[request],
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=cache,
        forward_mode=ForwardMode.DECODE,
        device="cpu",
    )
    batch.maybe_evict_swa()
    retained_positions = window_positions + 1
    assert allocator.available_size() == position_capacity - retained_positions
    batch.maybe_evict_swa()
    assert allocator.available_size() == position_capacity - retained_positions
    reused_slots = allocator.alloc(position_capacity - retained_positions)
    assert reused_slots is not None
    kv_pool.get_key_buffer(0)[reused_slots] = -1
    assert torch.equal(
        kv_pool.get_key_buffer(0)[slots[-retained_positions:]],
        expected_keys[-retained_positions:],
    )
    release_kv_cache(request, cache, is_insert=cache_on_release)
    assert request_pool.available_size() == 1
    assert allocator.available_size() == retained_positions
    allocator.free(reused_slots)
    all_slots = allocator.alloc(position_capacity)
    assert all_slots is not None
    assert all_slots.unique().numel() == position_capacity
    allocator.free(all_slots)
    assert allocator.available_size() == position_capacity


def prompt_request(rid: str, prompt: list[int], extra_key: str) -> Req:
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", prompt),
        sampling_params=SamplingParams(max_new_tokens=1),
        extra_key=extra_key,
    )


def schedule_prefill(
    request: Req,
    request_pool: ReqToTokenPool,
    allocator: PureSWATokenToKVPoolAllocator,
    cache: BasePrefixCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match, lock, allocate and insert one prompt as the scheduler does.

    Returns the matched slots and the slots of every prompt position.
    """
    request.init_next_round_input(cache)
    request.lock_receipt = cache.inc_lock_ref(request.last_node).to_dec_params()
    assert request_pool.alloc([request]) is not None
    matched_slots = request.prefix_indices.to(torch.int64).clone()
    num_cached = len(matched_slots)
    num_positions = len(request.origin_input_ids)
    request.set_extend_range(num_cached, num_positions)
    new_slots = allocator.alloc(num_positions - num_cached)
    assert new_slots is not None
    row = request_pool.req_to_token[request.kv.req_pool_idx]
    row[:num_cached] = request.prefix_indices.to(torch.int32)
    row[num_cached:num_positions] = new_slots.to(torch.int32)
    request.kv.kv_committed_len = num_positions
    request.kv.kv_allocated_len = num_positions
    cache.cache_unfinished_req(request)
    return matched_slots, row[:num_positions].to(torch.int64).clone()


@pytest.mark.usefixtures("window_radix_runtime")
def test_window_radix_keeps_a_shared_prompt_until_the_tree_evicts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prompt-only request leaves its prompt to the next request with the same key."""
    monkeypatch.setenv("SGLANG_SWA_EVICTION_INTERVAL", "4")
    position_capacity, window_positions = 40, 8
    prompt, num_decoded = [101, 102, 103, 104, 105, 106], 18
    request_pool, allocator, kv_pool, cache = make_window_cache(
        position_capacity, window_positions, radix=True
    )

    first = prompt_request("first", prompt, "voice")
    first_matched, prompt_slots = schedule_prefill(
        first, request_pool, allocator, cache
    )
    assert len(first_matched) == 0
    assert first.kv.cache_protected_len == len(prompt)
    assert cache.protected_size() == len(prompt)

    # Note (Jinjie Guo): Decode positions never enter the tree, as with
    # _omni_prompt_only_radix once the first token is out.
    first.skip_radix_cache_insert = True
    first.output_ids = list(range(num_decoded))
    decode_slots = allocator.alloc(num_decoded)
    assert decode_slots is not None
    total = len(prompt) + num_decoded
    request_pool.req_to_token[first.kv.req_pool_idx, len(prompt) : total] = (
        decode_slots.to(torch.int32)
    )
    first.kv.kv_committed_len = total
    first.kv.kv_allocated_len = total
    first.decode_batch_idx = 1
    prompt_keys = torch.arange(len(prompt) * 8, dtype=torch.float32).reshape(
        len(prompt), 1, 8
    )
    kv_pool.get_key_buffer(0)[prompt_slots] = prompt_keys
    batch = ScheduleBatch(
        reqs=[first],
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=cache,
        forward_mode=ForwardMode.DECODE,
        device="cpu",
    )
    batch.maybe_evict_swa()
    window_evicted = total - 1 - window_positions - len(prompt)
    assert window_evicted > 0
    assert allocator.available_size() == position_capacity - total + window_evicted
    reused_slots = allocator.alloc(allocator.available_size())
    assert reused_slots is not None
    kv_pool.get_key_buffer(0)[reused_slots] = -1
    assert torch.equal(kv_pool.get_key_buffer(0)[prompt_slots], prompt_keys)
    allocator.free(reused_slots)

    release_kv_cache(first, cache)
    assert allocator.available_size() == position_capacity - len(prompt)
    assert cache.evictable_size() == len(prompt)
    assert cache.protected_size() == 0

    other_voice = prompt_request("other-voice", prompt, "other")
    other_voice.init_next_round_input(cache)
    assert len(other_voice.prefix_indices) == 0

    second = prompt_request("second", prompt, "voice")
    second_matched, second_slots = schedule_prefill(
        second, request_pool, allocator, cache
    )
    assert torch.equal(second_matched, prompt_slots[:-1])
    assert torch.equal(second_slots, prompt_slots)
    assert allocator.available_size() == position_capacity - len(prompt)
    assert torch.equal(kv_pool.get_key_buffer(0)[second_slots], prompt_keys)

    second.skip_radix_cache_insert = True
    release_kv_cache(second, cache)
    assert allocator.available_size() == position_capacity - len(prompt)
    cache.evict(EvictParams(num_tokens=len(prompt)))
    assert cache.evictable_size() == 0
    assert allocator.available_size() == position_capacity
