# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from array import array
from collections.abc import Iterator

import pytest
import torch

pytest.importorskip("sglang")

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.qwen3_omni.prompt_hidden import PromptHiddenCapture
from sglang_omni.models.qwen3_omni.prompt_hidden_cache import QwenPromptHiddenRadixCache
from sglang_omni.models.qwen3_omni.request_builders import QwenThinkerRequestData
from sglang_omni.scheduling.sglang_backend.cache import create_tree_cache

TOKEN_CAPACITY = 128


@pytest.fixture(autouse=True)
def cache_runtime() -> Iterator[None]:
    with get_context().override_server_args(
        attention_backend="torch_native",
        prefill_attention_backend="torch_native",
        decode_attention_backend="torch_native",
        disable_radix_cache=False,
        chunked_prefill_size=None,
        enable_streaming_session=False,
        disaggregation_mode=None,
        speculative_algorithm=None,
        strip_thinking_cache=False,
    ):
        yield


def make_cache(
    *, page_size: int = 1, eviction_policy: str = "lru"
) -> QwenPromptHiddenRadixCache:
    return QwenPromptHiddenRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=ReqToTokenPool(4, TOKEN_CAPACITY, "cpu", False),
            token_to_kv_pool_allocator=TokenToKVPoolAllocator(
                TOKEN_CAPACITY, torch.float32, "cpu", None, False
            ),
            page_size=page_size,
            eviction_policy=eviction_policy,
        )
    )


def make_request(
    cache: QwenPromptHiddenRadixCache,
    tokens: list[int],
    positions: list[int],
    *,
    cacheable_prefix_length: int | None = None,
    extra_key: str | None = None,
    cache_salt: str | None = None,
) -> Req:
    request = Req(
        rid="prompt-cache-request",
        origin_input_text="",
        origin_input_ids=array("q", tokens),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    request.extra_key = extra_key
    request.cache_salt = cache_salt
    request.omni_data = QwenThinkerRequestData(
        prompt_hidden_capture=(
            PromptHiddenCapture(
                positions=torch.tensor(positions),
                cacheable_prefix_length=cacheable_prefix_length,
            )
            if positions
            else None
        )
    )
    request.init_next_round_input(cache)
    return request


def allocate_extend(
    cache: QwenPromptHiddenRadixCache, request: Req, end_position: int
) -> None:
    if request.kv.holds_kv:
        start_position = request.kv.kv_allocated_len
    else:
        assert cache.req_to_token_pool.alloc([request]) is not None
        cache.inc_lock_ref(request.last_node)
        start_position = len(request.prefix_indices)
        cache.req_to_token_pool.write(
            (request.kv.req_pool_idx, slice(0, start_position)), request.prefix_indices
        )
    new_indices = cache.token_to_kv_pool_allocator.alloc(end_position - start_position)
    assert new_indices is not None
    cache.req_to_token_pool.write(
        (request.kv.req_pool_idx, slice(start_position, end_position)), new_indices
    )
    request.set_extend_range(start_position, end_position)
    request.kv.kv_committed_len = end_position
    request.kv.kv_allocated_len = end_position


def test_finished_prefill_reuses_rows_after_stream_handoff() -> None:
    cache = make_cache()
    tokens = list(range(8))
    rows = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    first = make_request(cache, tokens, [2, 5])
    allocate_extend(cache, first, len(tokens))
    capture = first.omni_data.prompt_hidden_capture
    capture.add_extend(prefix_length=0, hidden_states=rows)
    emitted = capture.take_complete()
    release_kv_cache(first, cache)
    emitted.hidden_states.fill_(-1)
    capture.hidden_states.fill_(-2)

    second = make_request(cache, tokens, [2, 5])
    assert len(second.prefix_indices) == len(tokens) - 1
    restored = second.omni_data.prompt_hidden_capture.take_complete()
    torch.testing.assert_close(restored.hidden_states, rows[[2, 5]])
    restored.hidden_states.fill_(-3)
    third = make_request(cache, tokens, [2, 5])
    torch.testing.assert_close(
        third.omni_data.prompt_hidden_capture.take_complete().hidden_states,
        rows[[2, 5]],
    )


@pytest.mark.parametrize("page_size", [1, 2])
def test_kv_only_prompt_recomputes_from_first_missing_required_row(
    page_size: int,
) -> None:
    cache = make_cache(page_size=page_size)
    tokens = list(range(8))
    first = make_request(cache, tokens, [])
    allocate_extend(cache, first, len(tokens))
    release_kv_cache(first, cache)

    speech = make_request(cache, tokens, [3, 5])
    prefix_length = 3 // page_size * page_size
    assert len(speech.prefix_indices) == prefix_length
    assert len(speech.last_node.value) == prefix_length
    assert not speech.omni_data.prompt_hidden_capture.received.any()
    allocate_extend(cache, speech, len(tokens))
    rows = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    speech.omni_data.prompt_hidden_capture.add_extend(
        prefix_length=prefix_length, hidden_states=rows[prefix_length:]
    )
    cache.cache_unfinished_req(speech)
    assert speech.kv.cache_protected_len == len(tokens)
    assert cache.token_to_kv_pool_allocator.available_size() == TOKEN_CAPACITY - len(
        tokens
    )
    release_kv_cache(speech, cache)
    repeated = make_request(cache, tokens, [3, 5])
    torch.testing.assert_close(
        repeated.omni_data.prompt_hidden_capture.take_complete().hidden_states,
        rows[[3, 5]],
    )


def test_fallback_restores_no_rows_after_the_shortened_match() -> None:
    cache = make_cache()
    tokens = list(range(8))
    first = make_request(cache, tokens, [1, 5])
    allocate_extend(cache, first, len(tokens))
    first.omni_data.prompt_hidden_capture.add_extend(
        prefix_length=0, hidden_states=torch.ones((8, 3))
    )
    release_kv_cache(first, cache)

    second = make_request(cache, tokens, [1, 3, 5])
    assert len(second.prefix_indices) == 3
    assert second.omni_data.prompt_hidden_capture.received.tolist() == [
        True,
        False,
        False,
    ]


def test_partial_page_rows_stay_request_owned() -> None:
    cache = make_cache(page_size=2)
    tokens = list(range(7))
    rows = torch.arange(21, dtype=torch.float32).reshape(7, 3)
    first = make_request(cache, tokens, [1, 6])
    allocate_extend(cache, first, len(tokens))
    first.omni_data.prompt_hidden_capture.add_extend(
        prefix_length=0, hidden_states=rows
    )
    release_kv_cache(first, cache)
    assert cache.total_size() == 6
    assert (
        sum(states.positions.numel() for states in cache.prompt_hidden_states.values())
        == 1
    )
    repeated = make_request(cache, tokens, [1, 6])
    assert len(repeated.prefix_indices) == 6
    capture = repeated.omni_data.prompt_hidden_capture
    assert capture.received.tolist() == [True, False]
    capture.add_extend(prefix_length=6, hidden_states=rows[6:])
    torch.testing.assert_close(capture.take_complete().hidden_states, rows[[1, 6]])


@pytest.mark.parametrize("page_size", [1, 2])
def test_partial_prefix_split_compacts_states_and_evicts_only_its_suffix(
    page_size: int,
) -> None:
    cache = make_cache(page_size=page_size)
    first = make_request(cache, list(range(8)), [1, 5])
    allocate_extend(cache, first, 8)
    rows = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    first.omni_data.prompt_hidden_capture.add_extend(
        prefix_length=0, hidden_states=rows
    )
    release_kv_cache(first, cache)
    partial = make_request(cache, [0, 1, 2, 99, 4, 5, 6, 7], [1, 5])
    prefix_length = 3 // page_size * page_size
    assert len(partial.prefix_indices) == prefix_length
    capture = partial.omni_data.prompt_hidden_capture
    assert capture.received.tolist() == [True, False]
    torch.testing.assert_close(capture.hidden_states[0], rows[1])
    suffix_node = next(iter(partial.last_node.children.values()))
    for stored in cache.prompt_hidden_states.values():
        assert stored.hidden_states.untyped_storage().nbytes() == (
            stored.hidden_states.numel() * stored.hidden_states.element_size()
        )
        assert stored.positions.untyped_storage().nbytes() == (
            stored.positions.numel() * stored.positions.element_size()
        )
    cache.evict(EvictParams(num_tokens=8 - prefix_length))
    assert suffix_node not in cache.prompt_hidden_states
    assert (
        sum(states.positions.numel() for states in cache.prompt_hidden_states.values())
        == 1
    )
    assert cache.total_size() == prefix_length
    cache.reset()
    assert not cache.prompt_hidden_states
    assert not cache.evict_heap
    assert not make_request(cache, list(range(8)), [1, 5]).prefix_indices.numel()


@pytest.mark.parametrize(
    ("tokens", "extra_key", "cache_salt", "expected_prefix_length"),
    [
        ([9, 100, 2, 3], None, None, 0),
        ([0, 101, 2, 3], None, None, 1),
        ([0, 100, 2, 3], "other-adapter", None, 0),
        ([0, 100, 2, 3], None, "other-salt", 0),
    ],
)
def test_rows_never_cross_media_text_or_cache_namespaces(
    tokens: list[int],
    extra_key: str | None,
    cache_salt: str | None,
    expected_prefix_length: int,
) -> None:
    cache = make_cache()
    first = make_request(cache, [0, 100, 2, 3], [1])
    allocate_extend(cache, first, 4)
    first.omni_data.prompt_hidden_capture.add_extend(
        prefix_length=0, hidden_states=torch.ones((4, 3))
    )
    release_kv_cache(first, cache)
    second = make_request(
        cache, tokens, [1], extra_key=extra_key, cache_salt=cache_salt
    )
    assert len(second.prefix_indices) == expected_prefix_length
    assert second.omni_data.prompt_hidden_capture.needs_capture


@pytest.mark.parametrize("page_size", [1, 2])
@pytest.mark.parametrize("insert_on_release", [False, True])
def test_unsafe_media_suffix_remains_private_until_release(
    page_size: int,
    insert_on_release: bool,
) -> None:
    cache = make_cache(page_size=page_size)
    tokens = [0, 100, 2, 200, 4, 5]
    request = make_request(cache, tokens, [1, 3], cacheable_prefix_length=3)
    allocate_extend(cache, request, len(tokens))
    capture = request.omni_data.prompt_hidden_capture
    capture.add_extend(prefix_length=0, hidden_states=torch.ones((6, 3)))
    capture.take_complete()
    original_indices = cache.req_to_token_pool.req_to_token[
        request.kv.req_pool_idx, : len(tokens)
    ].clone()
    cache.cache_unfinished_req(request, chunked=True)
    prefix_length = 3 // page_size * page_size
    assert request.kv.cache_protected_len == prefix_length
    assert torch.equal(request.prefix_indices, original_indices)
    assert cache.total_size() == prefix_length
    second = make_request(cache, tokens, [1, 3], cacheable_prefix_length=3)
    assert len(second.prefix_indices) == prefix_length
    assert second.omni_data.prompt_hidden_capture.received.tolist() == [True, False]
    release_kv_cache(request, cache, is_insert=insert_on_release)
    assert cache.req_to_token_pool.available_size() == 4
    assert (
        cache.token_to_kv_pool_allocator.available_size()
        == TOKEN_CAPACITY - prefix_length
    )
    assert capture.take_complete() is None
    cache.evict(EvictParams(num_tokens=TOKEN_CAPACITY))
    all_indices = cache.token_to_kv_pool_allocator.alloc(TOKEN_CAPACITY)
    assert all_indices is not None
    assert all_indices.unique().numel() == TOKEN_CAPACITY


def test_chunked_capture_survives_retraction_after_handoff() -> None:
    cache = make_cache()
    tokens = list(range(8))
    rows = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    request = make_request(cache, tokens, [2, 6])
    capture = request.omni_data.prompt_hidden_capture
    allocate_extend(cache, request, 4)
    capture.add_extend(prefix_length=0, hidden_states=rows[:4])
    cache.cache_unfinished_req(request, chunked=True)
    assert (
        sum(states.positions.numel() for states in cache.prompt_hidden_states.values())
        == 1
    )
    allocate_extend(cache, request, 8)
    capture.add_extend(prefix_length=4, hidden_states=rows[4:])
    emitted = capture.take_complete()
    cache.cache_unfinished_req(request)
    assert (
        sum(states.positions.numel() for states in cache.prompt_hidden_states.values())
        == 2
    )
    release_kv_cache(request, cache, is_insert=False)
    request.reset_for_retract()
    cache.evict(EvictParams(num_tokens=TOKEN_CAPACITY))
    assert not cache.prompt_hidden_states
    request.init_next_round_input(cache)
    assert not request.prefix_indices.numel()
    assert not capture.needs_capture
    assert capture.take_complete() is None
    allocate_extend(cache, request, 8)
    cache.cache_unfinished_req(request)
    release_kv_cache(request, cache)
    repeated = make_request(cache, tokens, [2, 6])
    torch.testing.assert_close(
        repeated.omni_data.prompt_hidden_capture.take_complete().hidden_states,
        emitted.hidden_states,
    )


@pytest.mark.parametrize(
    ("eviction_policy", "evicted_tokens"),
    [
        ("lru", [4, 5, 6, 7]),
        ("mru", [0, 1, 2, 3]),
        ("fifo", [0, 1, 2, 3]),
        ("filo", [4, 5, 6, 7]),
    ],
)
def test_hidden_rows_follow_configured_eviction_policy(
    eviction_policy: str,
    evicted_tokens: list[int],
) -> None:
    cache = make_cache(eviction_policy=eviction_policy)
    for tokens in ([0, 1, 2, 3], [4, 5, 6, 7]):
        request = make_request(cache, tokens, [1])
        allocate_extend(cache, request, 4)
        request.omni_data.prompt_hidden_capture.add_extend(
            prefix_length=0, hidden_states=torch.ones((4, 3))
        )
        release_kv_cache(request, cache)
    cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [0, 1, 2, 3]))))
    result = cache.evict(EvictParams(num_tokens=4))
    assert result.num_tokens_evicted == 4
    assert len(cache.prompt_hidden_states) == 1
    assert not cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", evicted_tokens)))
    ).device_indices.numel()


@pytest.mark.parametrize("disable_radix_cache", [False, True])
def test_qwen_cache_selection_preserves_disabled_cache(
    disable_radix_cache: bool,
) -> None:
    with get_context().override_server_args(disable_radix_cache=disable_radix_cache):
        cache = create_tree_cache(
            ReqToTokenPool(4, TOKEN_CAPACITY, "cpu", False),
            TokenToKVPoolAllocator(TOKEN_CAPACITY, torch.float32, "cpu", None, False),
            1,
            None,
            radix_cache_type=QwenPromptHiddenRadixCache,
        )
    assert type(cache) is (
        ChunkCache if disable_radix_cache else QwenPromptHiddenRadixCache
    )
