# SPDX-License-Identifier: Apache-2.0
"""CPU allocator fixtures and prefill driver shared by Music3 scheduler tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from queue import Queue
from threading import RLock, get_ident

import pytest
import torch
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import get_context, get_schedule
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.minimax_music3.scheduler import MiniMaxMusic3Scheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


@pytest.fixture
def pool_size() -> int:
    return 29056


@pytest.fixture(params=[16, 128])
def scheduler(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, pool_size: int
) -> Iterator[MiniMaxMusic3Scheduler]:
    page_size = request.param
    with get_context().override_server_args(page_size=page_size):
        cache_pool = MHATokenToKVPool(
            size=pool_size,
            page_size=page_size,
            dtype=torch.float32,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        allocator = PagedTokenToKVPoolAllocator(
            size=pool_size,
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
        model_scheduler.schedule_policy = "fcfs"
        model_scheduler.policy = SchedulePolicy(
            "fcfs", model_scheduler.tree_cache, False, False, False
        )
        monkeypatch.setattr(OmniScheduler, "__init__", lambda self, **kwargs: None)
        MiniMaxMusic3Scheduler.__init__(model_scheduler)
        model_scheduler.session_bridge = None
        model_scheduler.scheduler_thread_id = get_ident()
        model_scheduler.enable_overlap = False
        model_scheduler.tp_size = 1
        model_scheduler.is_entry_rank = True
        model_scheduler.inbox = Queue()
        model_scheduler.outbox = Queue()
        model_scheduler.idle_wait_message = None
        model_scheduler.request_admission_lock = RLock()
        model_scheduler.aborted_request_ids = set()
        model_scheduler.aborted_request_id_order = deque()
        model_scheduler.completed_request_ids = {}
        model_scheduler.pending_request_builds = {}
        model_scheduler.pending_request_admissions = {}
        model_scheduler.backlogged_request_build_payloads = deque()
        model_scheduler.deferred_request_payloads = {}
        model_scheduler.dirty_deferred_request_ids = set()
        model_scheduler.pending_stream_ingress = {}
        model_scheduler.first_emit_done = set()
        model_scheduler.prefill_start_done = set()
        model_scheduler.prefill_end_done = set()
        model_scheduler.abort_callback = None
        model_scheduler.running_batch = None
        model_scheduler.cur_batch = None
        model_scheduler.last_batch = None
        model_scheduler.decode_batch_states = {}
        model_scheduler.previous_pending_decode = None
        model_scheduler.async_pending = None
        model_scheduler.result_queue = deque()
        model_scheduler.enable_priority_scheduling = False
        model_scheduler.max_queued_requests = None
        model_scheduler.waiting_queue = []
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
    """Run real ordering and admission without model execution or device batches."""
    scheduler.policy.calc_priority(scheduler.waiting_queue, running_batch)
    adder = PrefillAdder(
        scheduler.page_size,
        scheduler.tree_cache,
        scheduler.token_to_kv_pool_allocator,
        running_batch,
        0.1,
        scheduler.max_prefill_tokens,
        None,
        prefill_max_requests=get_schedule().prefill_max_requests,
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
