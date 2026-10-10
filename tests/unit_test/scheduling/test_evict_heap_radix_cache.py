# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import random
from array import array
from collections.abc import Iterator

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, release_req
from sglang.srt.mem_cache.allocation import alloc_for_extend
from sglang.srt.mem_cache.allocator.swa import PureSWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    InsertParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.scheduling.sglang_backend.cache import create_tree_cache
from sglang_omni.scheduling.sglang_backend.evict_heap_radix_cache import (
    EvictHeapRadixCache,
)


class MockAllocator:
    device = "cpu"

    def free(self, value):
        pass

    def free_segment(self, value, start_pos=0):
        pass

    def available_size(self):
        return 1 << 30


def make(cache_cls, eviction_policy="lru"):
    """Simulated-cache builder; create_simulated hardcodes RadixCache."""
    return cache_cls(
        CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=MockAllocator(),
            page_size=1,
            enable_kv_cache_events=False,
            eviction_policy=eviction_policy,
        )
    )


def run_trace(cache, seed: int, steps: int = 4000, drain: bool = True) -> list:
    """Drive an identical insert/lock/unlock/evict trace; return eviction order."""
    order = []
    orig_delete = cache._delete_leaf  # noqa: leading-underscore  # upstream name
    cache._delete_leaf = lambda node: (  # noqa: leading-underscore  # upstream name
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
        stock = make(RadixCache)
        patched = make(EvictHeapRadixCache)
        stock_order = run_trace(stock, seed)
        patched_order = run_trace(patched, seed)
        assert patched_order == stock_order
        assert len(patched.evictable_leaves) == len(stock.evictable_leaves)


def test_factory_selects_evict_heap_only_for_lru():
    from sglang.srt.runtime_context import get_context

    from sglang_omni.scheduling.sglang_backend.cache import create_tree_cache

    def build(policy):
        with get_context().override_server_args(
            disable_radix_cache=False,
            chunked_prefill_size=None,
            radix_eviction_policy=policy,
        ):
            return create_tree_cache(None, MockAllocator(), 1, None)

    assert type(build("lru")) is EvictHeapRadixCache
    for policy in ("mru", "priority", "lfu", "fifo", "filo"):
        assert type(build(policy)) is RadixCache, policy


def test_factory_passes_the_eviction_policy_config_to_the_strategy():
    from sglang.srt.runtime_context import get_context

    from sglang_omni.scheduling.sglang_backend.cache import create_tree_cache

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
        cache = create_tree_cache(None, MockAllocator(), 1, None)

    assert cache.eviction_strategy.protected_threshold == 4


def test_heap_stays_bounded_and_recovers():
    cache = make(EvictHeapRadixCache)
    run_trace(cache, seed=7, steps=2000, drain=False)
    assert cache.evictable_leaves
    assert len(cache.evict_heap) <= max(1024, 4 * len(cache.evictable_leaves))
    cache.evict(EvictParams(num_tokens=1 << 20))
    # The cache keeps working after a full drain.
    key = RadixKey(token_ids=[1, 2, 3], extra_key="post")
    cache.insert(InsertParams(key=key, value=torch.arange(3)))
    result = cache.evict(EvictParams(num_tokens=1 << 20))
    assert result.num_tokens_evicted >= 3
    assert not cache.evictable_leaves


def test_reset_then_reuse():
    cache = make(EvictHeapRadixCache)
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


def make_window_cache(
    position_capacity: int, window_positions: int
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
        disable_radix_cache=True,
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
        ordinary_cache = create_tree_cache(None, MockAllocator(), 1, None)
    assert isinstance(ordinary_cache, ChunkCache)
    assert not ordinary_cache.supports_swa()


@pytest.fixture
def window_cache_runtime() -> Iterator[None]:
    with get_context().override_server_args(
        attention_backend="torch_native",
        prefill_attention_backend="torch_native",
        decode_attention_backend="torch_native",
        disable_radix_cache=True,
        chunked_prefill_size=None,
        enable_streaming_session=False,
        disaggregation_mode=None,
        speculative_algorithm=None,
        strip_thinking_cache=False,
    ):
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
    assert not request.kv.holds_kv
    assert request.kv.is_kv_released
    assert request_pool.available_size() == 1
    assert allocator.available_size() == retained_positions
    allocator.free(reused_slots)
    all_slots = allocator.alloc(position_capacity)
    assert all_slots is not None
    assert all_slots.unique().numel() == position_capacity
    allocator.free(all_slots)
    assert allocator.available_size() == position_capacity


@pytest.mark.usefixtures("window_cache_runtime")
def test_window_eviction_resumes_after_retract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_SWA_EVICTION_INTERVAL", "4")
    position_capacity, position_count, window_positions = 32, 24, 8
    request_pool, allocator, kv_pool, cache = make_window_cache(
        position_capacity, window_positions
    )
    request = Req(
        rid="retracted-window-request",
        origin_input_text="",
        origin_input_ids=array("q", range(16)),
        sampling_params=SamplingParams(max_new_tokens=16),
    )
    request.output_ids.extend(range(16, position_count))
    assert request_pool.alloc([request]) is not None
    initial_slots = allocator.alloc(position_count)
    assert initial_slots is not None
    request_pool.req_to_token[request.kv.req_pool_idx, :position_count] = (
        initial_slots.to(torch.int32)
    )
    request.kv.kv_committed_len = position_count
    request.kv.kv_allocated_len = position_count
    request.decode_batch_idx = 1
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
    release_req(
        req=request,
        remaing_req_count=0,
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=cache,
        hisparse_coordinator=None,
        offload_kv=False,
    )
    assert request.is_retracted
    assert request.retraction_count == 1
    assert request.kv.is_kv_released
    assert allocator.available_size() == position_capacity
    assert request_pool.available_size() == 1
    request.init_next_round_input(cache)
    assert request.full_untruncated_fill_ids == array("q", range(position_count))
    request.set_extend_range(0, position_count)
    batch.forward_mode = ForwardMode.EXTEND
    batch.prefix_lens, batch.extend_lens = [0], [position_count]
    batch.seq_lens = batch.seq_lens_cpu = torch.tensor([position_count])
    batch.extend_num_tokens = position_count
    replay_slots, _, _ = alloc_for_extend(batch)
    expected_keys = torch.arange(position_count * 8, dtype=torch.float32).reshape(
        position_count, 1, 8
    )
    kv_pool.get_key_buffer(0)[replay_slots] = expected_keys
    request.is_retracted = False
    batch.forward_mode = ForwardMode.DECODE
    batch.maybe_evict_swa()
    assert allocator.available_size() == position_capacity - position_count
    request.decode_batch_idx = 1
    batch.maybe_evict_swa()
    assert allocator.available_size() == position_capacity - retained_positions
    continued_positions = 4
    continued_slots = allocator.alloc(continued_positions)
    assert continued_slots is not None
    request_pool.req_to_token[
        request.kv.req_pool_idx, position_count : position_count + continued_positions
    ] = continued_slots.to(torch.int32)
    request.output_ids.extend(
        range(position_count, position_count + continued_positions)
    )
    request.kv.kv_committed_len += continued_positions
    request.kv.kv_allocated_len += continued_positions
    kv_pool.get_key_buffer(0)[continued_slots] = -2
    batch.maybe_evict_swa()
    assert allocator.available_size() == position_capacity - retained_positions
    reused_slots = allocator.alloc(position_capacity - retained_positions)
    assert reused_slots is not None
    kv_pool.get_key_buffer(0)[reused_slots] = -1
    assert torch.equal(
        kv_pool.get_key_buffer(0)[
            replay_slots[-(retained_positions - continued_positions) :]
        ],
        expected_keys[-(retained_positions - continued_positions) :],
    )
    assert torch.all(kv_pool.get_key_buffer(0)[continued_slots] == -2)
    release_kv_cache(request, cache)
    assert not request.kv.holds_kv
    assert request.kv.is_kv_released
    assert request_pool.available_size() == 1
    assert allocator.available_size() == retained_positions
    allocator.free(reused_slots)
    all_slots = allocator.alloc(position_capacity)
    assert all_slots is not None
    assert all_slots.unique().numel() == position_capacity
    allocator.free(all_slots)
    assert allocator.available_size() == position_capacity
