# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import random
from array import array
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.scheduling import omni_scheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.cache import prompt_cache_key
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
            return create_tree_cache(None, MockAllocator(), 1)

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
        cache = create_tree_cache(None, MockAllocator(), 1)

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


@pytest.mark.parametrize("page_size", [1, 4])
def test_shared_prompt_switches_to_private_chunked_replay(
    page_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    allocator = PagedTokenToKVPoolAllocator(
        size=64,
        page_size=page_size,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    request_pool = ReqToTokenPool(2, 32, "cpu", enable_memory_saver=False)
    cache = EvictHeapRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=request_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
        )
    )
    scheduler = OmniScheduler.__new__(OmniScheduler)
    queued = []
    monkeypatch.setattr(
        omni_scheduler._Upstream,  # noqa: leading-underscore  # Existing request or scheduler interface.
        "_add_request_to_queue",
        lambda scheduler, request, is_retracted=False: queued.append(request),
    )
    monkeypatch.setattr(
        omni_scheduler._Upstream,  # noqa: leading-underscore  # Existing request or scheduler interface.
        "process_batch_result",
        lambda scheduler, batch, result: [
            maybe_cache_unfinished_req(request, cache) for request in batch.reqs
        ],
    )
    prompt = array("q", [1, 2, 3, 4, 5])
    initial_available = allocator.available_size()
    requests = []
    for index in range(2):
        request = Req(
            rid=str(index),
            origin_input_text="",
            origin_input_ids=prompt,
            sampling_params=SamplingParams(max_new_tokens=8),
            extra_key="same-reference",
        )
        request._omni_prompt_only_radix = (
            True  # noqa: leading-underscore  # Existing request or scheduler interface.
        )
        request.use_private_radix_on_retract = True
        request.omni_data = SimpleNamespace(decode_input_embeds=[])
        request.full_untruncated_fill_ids = prompt[:]
        request.set_extend_range(0, len(prompt))
        request.kv.req_pool_idx = index + 1
        request.last_node = cache.root_node
        allocated = allocator.alloc((9 + page_size - 1) // page_size * page_size)
        request_pool.write((index + 1, slice(0, 9)), allocated[:9])
        request.output_ids.append(6)
        scheduler.process_batch_result(SimpleNamespace(reqs=[request]), None)
        assert request.skip_radix_cache_insert
        request.output_ids.extend([7, 8, 9, 10])
        request.full_untruncated_fill_ids = prompt + request.output_ids
        request.set_extend_range(len(prompt), 9)
        maybe_cache_unfinished_req(request, cache)
        requests.append(request)
    shared_length = len(prompt) // page_size * page_size
    assert cache.total_size() == shared_length
    assert torch.equal(
        requests[0].prefix_indices[:shared_length],
        requests[1].prefix_indices[:shared_length],
    )
    private_keys = []
    for index, request in enumerate(requests):
        cache.cache_finished_req(request, is_insert=False, kv_len_to_handle=9)
        request.kv.req_pool_idx = None
        request.reset_for_retract()
        scheduler._add_request_to_queue(
            request, is_retracted=True
        )  # noqa: leading-underscore  # Existing request or scheduler interface.
        private_key = request.extra_key
        assert private_key != "same-reference"
        assert not request.skip_radix_cache_insert
        assert (
            not request._omni_prompt_only_radix
        )  # noqa: leading-underscore  # Existing request or scheduler interface.
        assert not request.use_private_radix_on_retract
        private_keys.append(private_key)
        request.init_next_round_input(cache)
        assert len(request.prefix_indices) == 0
        request.kv.req_pool_idx = index + 1
        request.is_retracted = False
        allocated = allocator.alloc((10 + page_size - 1) // page_size * page_size)
        request_pool.write((index + 1, slice(0, 10)), allocated[:10])
        request.set_extend_range(0, 8)
        maybe_cache_unfinished_req(request, cache, chunked=True)
        scheduler.process_batch_result(SimpleNamespace(reqs=[request]), None)
        request.init_next_round_input()
        assert len(request.prefix_indices) == 8
        assert not request.skip_radix_cache_insert
        request.set_extend_range(8, 10)
        maybe_cache_unfinished_req(request, cache, chunked=True)
        assert len(request.prefix_indices) == 10
        cache.cache_finished_req(request, is_insert=False, kv_len_to_handle=10)
        request.kv.req_pool_idx = None
        request.reset_for_retract()
        scheduler._add_request_to_queue(
            request, is_retracted=True
        )  # noqa: leading-underscore  # Existing request or scheduler interface.
        assert request.extra_key == private_key
        assert not request.skip_radix_cache_insert
        request.init_next_round_input(cache)
        assert len(request.prefix_indices) == 9 // page_size * page_size
    assert private_keys[0] != private_keys[1]
    assert len(queued) == 4
    cache.evict(EvictParams(num_tokens=64))
    assert allocator.available_size() == initial_available
    free_pages = allocator.get_all_free_pages()
    assert len(free_pages.unique()) == len(free_pages)


def test_private_retract_policy_does_not_change_other_prompt_cache_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = Req(
        rid="legacy",
        origin_input_text="",
        origin_input_ids=array("q", [1]),
        sampling_params=SamplingParams(max_new_tokens=8),
        extra_key="legacy-prompt",
    )
    request._omni_prompt_only_radix = (
        True  # noqa: leading-underscore  # Existing request or scheduler interface.
    )
    request.skip_radix_cache_insert = True
    request.omni_data = SimpleNamespace(decode_input_embeds=[])
    request.reset_for_retract()
    monkeypatch.setattr(
        omni_scheduler._Upstream,  # noqa: leading-underscore  # Existing request or scheduler interface.
        "_add_request_to_queue",
        lambda scheduler, request, is_retracted=False: None,
    )
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._add_request_to_queue(
        request, is_retracted=True
    )  # noqa: leading-underscore  # Existing request or scheduler interface.
    assert request.extra_key == "legacy-prompt"
    assert request.skip_radix_cache_insert
    assert (
        request._omni_prompt_only_radix
    )  # noqa: leading-underscore  # Existing request or scheduler interface.


def test_prompt_fingerprint_covers_shape_dtype_and_all_codebooks() -> None:
    codes = torch.tensor([[1, 2], [3, 4]])
    key = prompt_cache_key("audio", codes)
    assert key == prompt_cache_key("audio", codes.clone())
    assert key != prompt_cache_key("audio", codes.reshape(1, 4))
    assert key != prompt_cache_key("audio", codes.float())
    changed = codes.clone()
    changed[0, 1] = 5
    assert key != prompt_cache_key("audio", changed)
    assert key != prompt_cache_key("audio", codes, torch.ones(2))
