# SPDX-License-Identifier: Apache-2.0
"""Music3 CFG admission across input and full-song KV limits."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.minimax_music3.scheduler import MiniMaxMusic3Scheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


@pytest.fixture(params=[16, 128])
def scheduler(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MiniMaxMusic3Scheduler]:
    page_size = request.param
    with get_context().override_server_args(page_size=page_size):
        cache_pool = MHATokenToKVPool(
            size=29056,
            page_size=page_size,
            dtype=torch.float32,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        allocator = PagedTokenToKVPoolAllocator(
            size=29056,
            page_size=page_size,
            dtype=torch.float32,
            device="cpu",
            kvcache=cache_pool,
            need_sort=False,
        )
        request_pool = ReqToTokenPool(
            size=32,
            max_context_len=11000,
            device="cpu",
            enable_memory_saver=False,
        )
        model_scheduler = MiniMaxMusic3Scheduler.__new__(MiniMaxMusic3Scheduler)
        model_scheduler.page_size = page_size
        model_scheduler.max_prefill_tokens = 16384
        model_scheduler.max_req_len = 11000
        model_scheduler.draft_worker = None
        model_scheduler.token_to_kv_pool_allocator = allocator
        model_scheduler.tree_cache = RadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=request_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
            )
        )
        monkeypatch.setattr(
            model_scheduler, "get_num_allocatable_reqs", lambda count: 32
        )
        yield model_scheduler


def paired_requests(prompt_lengths: list[int], max_new_tokens: int) -> list[Req]:
    return [
        Req(
            rid=f"request-{index}{suffix}",
            origin_input_text="",
            origin_input_ids=[1] * prompt_tokens,
            sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        )
        for index, prompt_tokens in enumerate(prompt_lengths)
        for suffix in ("", "-cfg")
    ]


def admit_prefill(
    scheduler: MiniMaxMusic3Scheduler, running_batch: ScheduleBatch
) -> NextBatchPlan:
    """Run the real adder while excluding model execution and accelerator allocation."""
    adder = PrefillAdder(
        scheduler.page_size,
        scheduler.tree_cache,
        scheduler.token_to_kv_pool_allocator,
        running_batch,
        0.1,
        scheduler.max_prefill_tokens,
        None,
    )
    for request in scheduler.waiting_queue:
        scheduler.normalize_req_token_arrays(request)
        request.init_next_round_input(scheduler.tree_cache)
        verdict = adder.add_one_req(
            request, has_chunked_req=False, truncation_align_size=None
        )
        if verdict != AddReqResult.CONTINUE:
            break
        else:
            pass
    scheduler.waiting_queue = [
        request
        for request in scheduler.waiting_queue
        if request not in adder.can_run_list
    ]
    return NextBatchPlan(
        batch_to_run=ScheduleBatch(reqs=adder.can_run_list, batch_is_full=False),
        running_batch=running_batch,
    )


def test_long_song_admission_keeps_cfg_pairs_and_full_decode_room(
    scheduler: MiniMaxMusic3Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three long-song pairs formerly admitted five rows and then exhausted KV."""
    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", admit_prefill)
    requests = paired_requests([697, 697, 1544], 9001)
    scheduler.waiting_queue = requests.copy()
    plan = scheduler.get_new_batch_prefill(ScheduleBatch(reqs=[], batch_is_full=False))

    assert plan.batch_to_run.reqs == requests[:2]
    assert scheduler.waiting_queue == requests[2:]
    # Worst-case completion of both admitted rows must fit without retraction.
    for request in plan.batch_to_run.reqs:
        total_tokens = (
            len(request.origin_input_ids) + request.sampling_params.max_new_tokens
        )
        allocated = scheduler.token_to_kv_pool_allocator.alloc(
            -(-total_tokens // scheduler.page_size) * scheduler.page_size
        )
        assert allocated is not None


def test_short_requests_still_share_one_prefill_batch(
    scheduler: MiniMaxMusic3Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pair safety must not impose serial generation when eight requests fit."""
    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", admit_prefill)
    requests = paired_requests([128] * 8, 751)
    scheduler.waiting_queue = requests.copy()
    plan = scheduler.get_new_batch_prefill(ScheduleBatch(reqs=[], batch_is_full=False))

    assert plan.batch_to_run.reqs == requests
    assert scheduler.waiting_queue == []


def test_running_long_song_reservation_blocks_then_releases_next_pair(
    scheduler: MiniMaxMusic3Scheduler,
) -> None:
    """Clipping running-row growth to 4096 let later requests overcommit the pool."""
    running_requests = paired_requests([697], 9001)
    allocated_pages = []
    for request in running_requests:
        request.output_ids = [1] * 4500
        tokens = len(request.origin_input_ids) + len(request.output_ids) - 1
        allocated_pages.append(
            scheduler.token_to_kv_pool_allocator.alloc(
                -(-tokens // scheduler.page_size) * scheduler.page_size
            )
        )
    running_batch = ScheduleBatch(reqs=running_requests, batch_is_full=False)
    waiting_requests = paired_requests([847], 9001)
    assert scheduler.pair_admission_limit(waiting_requests, running_batch) == 0

    for pages in allocated_pages:
        scheduler.token_to_kv_pool_allocator.free(pages)
    running_batch.reqs = []
    assert scheduler.pair_admission_limit(waiting_requests, running_batch) == 2


@pytest.mark.parametrize("input_budget", [100, 512])
def test_input_budget_never_splits_a_pair_and_is_restored(
    scheduler: MiniMaxMusic3Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    input_budget: int,
) -> None:
    """The first oversize pair and subsequent page-aligned cap must remain atomic."""
    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", admit_prefill)
    scheduler.max_prefill_tokens = input_budget
    requests = paired_requests([127, 127], 10)
    scheduler.waiting_queue = requests.copy()
    plan = scheduler.get_new_batch_prefill(ScheduleBatch(reqs=[], batch_is_full=False))

    assert plan.batch_to_run.reqs == requests[:2]
    assert scheduler.waiting_queue == requests[2:]
    assert scheduler.max_prefill_tokens == input_budget


def test_impossible_pair_rejected_but_busy_pool_only_defers(
    scheduler: MiniMaxMusic3Scheduler,
) -> None:
    """A pair larger than the pool must fail admission instead of waiting forever."""
    scheduler.token_to_kv_pool_allocator.size = 16000
    request = paired_requests([697], 9001)[0]
    assert "full CFG pair" in scheduler.request_kv_capacity_error(request)
    scheduler.token_to_kv_pool_allocator.size = 29056
    scheduler.token_to_kv_pool_allocator.alloc(
        20000 // scheduler.page_size * scheduler.page_size
    )
    assert scheduler.request_kv_capacity_error(request) is None
    assert (
        scheduler.pair_admission_limit(
            paired_requests([697], 9001), ScheduleBatch(reqs=[], batch_is_full=False)
        )
        == 0
    )


def test_capacity_callback_accepts_keyword_with_legacy_parent(
    scheduler: MiniMaxMusic3Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit False keyword must also work with the older parent signature."""

    def legacy_capacity_error(self: OmniScheduler, req: Req) -> str:
        return f"Parent rejected {req.rid}"

    monkeypatch.setattr(
        OmniScheduler, "request_kv_capacity_error", legacy_capacity_error
    )
    request = paired_requests([697], 9001)[0]
    assert (
        scheduler.request_kv_capacity_error(request, session_capacity_checked=False)
        == f"Parent rejected {request.rid}"
    )


def test_capacity_callback_forwards_session_check_to_parent(
    scheduler: MiniMaxMusic3Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new parent callback's session capacity decision must be preserved."""
    received: list[bool] = []

    def session_capacity_error(
        self: OmniScheduler, req: Req, *, session_capacity_checked: bool = False
    ) -> str:
        received.append(session_capacity_checked)
        return f"Parent rejected {req.rid}"

    monkeypatch.setattr(
        OmniScheduler, "request_kv_capacity_error", session_capacity_error
    )
    request = paired_requests([697], 9001)[0]
    assert (
        scheduler.request_kv_capacity_error(request, session_capacity_checked=True)
        == f"Parent rejected {request.rid}"
    )
    assert received == [True]
