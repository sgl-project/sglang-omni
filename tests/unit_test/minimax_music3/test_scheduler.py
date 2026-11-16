# SPDX-License-Identifier: Apache-2.0
"""Music3 CFG admission across input and full-song KV limits."""

from __future__ import annotations

from collections.abc import Iterator
from queue import Queue
from threading import RLock
from typing import Literal
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    CacheAgnosticPolicy,
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
from sglang_omni.models.minimax_music3.serial_offload import SerialOffloadCoordinator
from sglang_omni.models.minimax_music3.sglang_request_builder import (
    MiniMaxMusic3SGLangRequestData,
)
from sglang_omni.proto import OmniRequest, StagePayload
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
        model_scheduler.schedule_policy = "fcfs"
        model_scheduler.policy = SchedulePolicy(
            "fcfs", model_scheduler.tree_cache, False, False, False
        )
        monkeypatch.setattr(OmniScheduler, "__init__", lambda self, **kwargs: None)
        MiniMaxMusic3Scheduler.__init__(model_scheduler)
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


@pytest.mark.parametrize(
    ("prefill_max_requests", "batch_rows"),
    [(None, 6), (2, 2), (3, 2), (5, 4)],
)
def test_prefill_request_cap_keeps_pairs_across_batches(
    scheduler: MiniMaxMusic3Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    prefill_max_requests: int | None,
    batch_rows: int,
) -> None:
    """An odd prefill request cap split a CFG pair inside the real adder."""
    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", admit_prefill)
    requests = paired_requests([128] * 3, 10)
    scheduler.waiting_queue = requests.copy()
    with get_context().override_server_args(prefill_max_requests=prefill_max_requests):
        for offset in range(0, len(requests), batch_rows):
            plan = scheduler.get_new_batch_prefill(
                ScheduleBatch(reqs=[], batch_is_full=False)
            )
            assert plan.batch_to_run.reqs == requests[offset : offset + batch_rows]
            assert scheduler.waiting_queue == requests[offset + batch_rows :]
            assert get_schedule().prefill_max_requests == prefill_max_requests
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


def test_priority_orders_all_waiting_pairs_before_capacity_admission(
    scheduler: MiniMaxMusic3Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clipping before sorting hid a higher-priority pair behind the queue head."""
    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", admit_prefill)
    scheduler.policy.enable_priority_scheduling = True
    requests = paired_requests([128, 128], 10)
    for index, request in enumerate(requests):
        request.priority = index // 2
        request.time_stats.wait_queue_entry_time = float(index // 2 + 1)
    scheduler.waiting_queue = requests.copy()
    with get_context().override_server_args(prefill_max_requests=2):
        plan = scheduler.get_new_batch_prefill(
            ScheduleBatch(reqs=[], batch_is_full=False)
        )
    assert plan.batch_to_run.reqs == requests[2:]
    assert scheduler.waiting_queue == requests[:2]


def test_serial_offload_owns_the_pair_selected_after_sorting(
    scheduler: MiniMaxMusic3Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offload ownership must follow the selected pair, not the unsorted queue head."""
    coordinator = SerialOffloadCoordinator()
    coordinator.register_ar(torch.nn.Linear(2, 2), torch.device("cpu"))
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.scheduler.get_coordinator",
        lambda: coordinator,
    )
    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", admit_prefill)
    scheduler.policy.enable_priority_scheduling = True
    requests = paired_requests([128, 128], 10)
    for index, request in enumerate(requests):
        request.priority = index // 2
        request.time_stats.wait_queue_entry_time = float(index // 2 + 1)
    scheduler.waiting_queue = requests.copy()
    plan = scheduler.get_new_batch_prefill(ScheduleBatch(reqs=[], batch_is_full=False))
    assert plan.batch_to_run.reqs == requests[2:]
    assert scheduler.waiting_queue == requests[:2]
    coordinator.begin_dit_handoff(requests[2].rid)
    plan = scheduler.get_new_batch_prefill(ScheduleBatch(reqs=[], batch_is_full=False))
    assert plan.batch_to_run.reqs == []
    assert scheduler.waiting_queue == requests[:2]


@pytest.mark.parametrize("prefill_max_requests", [3, 5])
def test_random_order_and_request_cap_preserve_pairs_until_queue_drains(
    scheduler: MiniMaxMusic3Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    prefill_max_requests: int,
) -> None:
    """Random sorting must see every waiting pair once before row-cap admission."""
    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", admit_prefill)
    scheduler.policy.policy = CacheAgnosticPolicy.RANDOM
    shuffled_queues: list[list[Req]] = []

    def reverse_rows(waiting_queue: list[Req]) -> None:
        shuffled_queues.append(waiting_queue.copy())
        waiting_queue.reverse()

    monkeypatch.setattr(
        "sglang.srt.managers.schedule_policy.random.shuffle", reverse_rows
    )
    requests = paired_requests([128] * 3, 10)
    scheduler.waiting_queue = requests.copy()
    admitted: list[Req] = []
    with get_context().override_server_args(prefill_max_requests=prefill_max_requests):
        for _ in range(3):
            before = scheduler.waiting_queue.copy()
            if not before:
                break
            expected_order = [
                request
                for index in range(len(before) - 2, -1, -2)
                for request in before[index : index + 2]
            ]
            plan = scheduler.get_new_batch_prefill(
                ScheduleBatch(reqs=[], batch_is_full=False)
            )
            assert shuffled_queues.pop() == before
            assert shuffled_queues == []
            batch_rows = prefill_max_requests - prefill_max_requests % 2
            assert plan.batch_to_run.reqs == expected_order[:batch_rows]
            assert scheduler.waiting_queue == expected_order[batch_rows:]
            admitted.extend(plan.batch_to_run.reqs)
    assert scheduler.waiting_queue == []
    assert len(admitted) == len(requests)
    assert set(admitted) == set(requests)


def test_sorted_pairs_wait_for_running_kv_then_resume(
    scheduler: MiniMaxMusic3Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sorting must preserve full decode reservations and the deferred queue."""
    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", admit_prefill)
    scheduler.policy.enable_priority_scheduling = True
    running_requests = paired_requests([697], 9001)
    allocated_pages = []
    for request in running_requests:
        request.output_ids = [1] * 4500
        token_count = len(request.origin_input_ids) + len(request.output_ids) - 1
        allocated_pages.append(
            scheduler.token_to_kv_pool_allocator.alloc(
                -(-token_count // scheduler.page_size) * scheduler.page_size
            )
        )
    requests = paired_requests([847, 847], 9001)
    for index, request in enumerate(requests):
        request.priority = index // 2
        request.time_stats.wait_queue_entry_time = float(index // 2 + 1)
    scheduler.waiting_queue = requests.copy()
    running_batch = ScheduleBatch(reqs=running_requests, batch_is_full=False)
    plan = scheduler.get_new_batch_prefill(running_batch)
    assert plan.batch_to_run.reqs == []
    assert scheduler.waiting_queue == requests[2:] + requests[:2]
    for pages in allocated_pages:
        scheduler.token_to_kv_pool_allocator.free(pages)
    running_batch.reqs = []
    plan = scheduler.get_new_batch_prefill(running_batch)
    assert plan.batch_to_run.reqs == requests[2:]
    assert scheduler.waiting_queue == requests[:2]


@pytest.mark.parametrize("raise_after_ordering", [False, True])
def test_deferred_pairs_and_input_budget_restored_on_early_exit(
    scheduler: MiniMaxMusic3Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    raise_after_ordering: bool,
) -> None:
    """An early exit after ordering must not lose parked pairs or enlarged budgets."""
    requests = paired_requests([128, 128], 10)
    scheduler.waiting_queue = requests.copy()
    scheduler.max_prefill_tokens = 100

    def stop_after_ordering(
        scheduler: MiniMaxMusic3Scheduler, running_batch: ScheduleBatch
    ) -> NextBatchPlan:
        scheduler.policy.calc_priority(scheduler.waiting_queue, running_batch)
        if raise_after_ordering:
            raise RuntimeError("Prefill interrupted")
        return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", stop_after_ordering)
    running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
    if raise_after_ordering:
        with pytest.raises(RuntimeError, match="Prefill interrupted"):
            scheduler.get_new_batch_prefill(running_batch)
    else:
        assert scheduler.get_new_batch_prefill(running_batch).batch_to_run is None
    assert scheduler.waiting_queue == requests
    assert scheduler.max_prefill_tokens == 100


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


@pytest.mark.parametrize(
    ("pool_size", "prefill_max_requests", "expected_error"),
    [
        (16000, None, "full CFG pair"),
        (29056, None, None),
        (29056, 1, "prefill_max_requests"),
    ],
)
def test_enqueue_checks_pair_capacity_before_queueing_both_rows(
    scheduler: MiniMaxMusic3Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    pool_size: int,
    prefill_max_requests: int | None,
    expected_error: str | None,
) -> None:
    """Real enqueue must reject impossible pairs before either row is queued."""
    scheduler.token_to_kv_pool_allocator.size = pool_size
    scheduler.session_bridge = None
    scheduler.deferred_request_payloads = {}
    scheduler.pending_stream_ingress = {}
    scheduler.aborted_request_ids = set()
    scheduler.request_admission_lock = RLock()
    scheduler.enable_priority_scheduling = False
    scheduler.max_queued_requests = None
    scheduler.waiting_queue = []
    scheduler.is_entry_rank = True
    scheduler.outbox = Queue()
    abort = Mock()
    monkeypatch.setattr(scheduler, "abort", abort)
    cond, uncond = paired_requests([697], 9001)
    payload = StagePayload(cond.rid, OmniRequest(inputs=None), None)
    req_data = MiniMaxMusic3SGLangRequestData(
        req=cond,
        enforce_request_limits=False,
        cfg_uncond=MiniMaxMusic3SGLangRequestData(req=uncond, is_cfg_uncond=True),
    )
    with get_context().override_server_args(prefill_max_requests=prefill_max_requests):
        scheduler.enqueue_built_request(payload, False, req_data)

    assert req_data.decode_path == "plain"
    if expected_error is not None:
        assert scheduler.waiting_queue == []
        message = scheduler.outbox.get_nowait()
        assert message.request_id == cond.rid and message.type == "error"
        assert isinstance(message.data, ValueError)
        assert expected_error in str(message.data)
        abort.assert_called_once_with(cond.rid)
    else:
        assert scheduler.waiting_queue == [cond, uncond]
        abort.assert_not_called()
    assert scheduler.outbox.empty()


@pytest.mark.parametrize(
    ("session_capacity_checked", "decode_path"),
    [(False, "plain"), (True, "spec"), (False, None)],
)
def test_capacity_callback_forwards_parent_admission_context(
    scheduler: MiniMaxMusic3Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    session_capacity_checked: bool,
    decode_path: Literal["plain", "spec"] | None,
) -> None:
    """Preserve the shared capacity decision and both inputs that inform it."""
    request = paired_requests([697], 9001)[0]
    error = f"Parent rejected {request.rid}"
    parent_check = Mock(return_value=error)
    monkeypatch.setattr(OmniScheduler, "request_kv_capacity_error", parent_check)
    assert (
        scheduler.request_kv_capacity_error(
            request,
            session_capacity_checked=session_capacity_checked,
            decode_path=decode_path,
        )
        == error
    )
    parent_check.assert_called_once_with(
        request,
        session_capacity_checked=session_capacity_checked,
        decode_path=decode_path,
    )
