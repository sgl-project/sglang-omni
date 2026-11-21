# SPDX-License-Identifier: Apache-2.0
"""Queue order and model path events of MiniMax Music 3 CFG row pairs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock
from types import SimpleNamespace

import pytest
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import CacheAgnosticPolicy, SchedulePolicy
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.minimax_music3.scheduler import (
    MiniMaxMusic3SchedulePolicy,
    MiniMaxMusic3Scheduler,
)
from sglang_omni.models.minimax_music3.sglang_request_builder import (
    MiniMaxMusic3SGLangRequestData,
    cfg_uncond_rid,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from tests.unit_test.fixtures.minimax_music3_scheduler import (
    admit_prefill,
    paired_requests,
)

pytest_plugins = ("tests.unit_test.fixtures.minimax_music3_scheduler",)


def queued_request(request_id: str) -> Req:
    return Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=1),
    )


def cfg_pair(request_id: str) -> list[Req]:
    return [queued_request(request_id), queued_request(cfg_uncond_rid(request_id))]


def test_cfg_pairs_stay_together_when_the_policy_reorders_the_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """schedule_policy=random placed each row alone, and the runner rejects a batch whose pairs are not adjacent."""
    queue = [*cfg_pair("a"), *cfg_pair("b")]

    def calc_priority(
        policy: SchedulePolicy,
        waiting_queue: list[Req],
        running_batch: ScheduleBatch | None,
        processed_tokens: int,
    ) -> None:
        # An order random.shuffle can produce.
        waiting_queue[:] = [waiting_queue[index] for index in (3, 0, 2, 1)]

    monkeypatch.setattr(SchedulePolicy, "calc_priority", calc_priority)
    policy = MiniMaxMusic3SchedulePolicy.__new__(MiniMaxMusic3SchedulePolicy)
    policy.calc_priority(queue, None, 0)
    assert [req.rid for req in queue] == ["a", "a-cfg", "b", "b-cfg"]


def test_finished_unconditioned_row_ends_its_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished unconditioned row emitted the start of its model path and never the end."""
    ended: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler._emit_model_path_end",
        lambda request_id, status: ended.append((request_id, status)),
    )
    parent_outputs: list[list[Req]] = []
    monkeypatch.setattr(
        OmniScheduler,
        "stream_output",
        lambda scheduler, reqs, return_logprob=False, skip_req=None: parent_outputs.append(
            list(reqs)
        ),
    )
    twin = SimpleNamespace(
        rid="a-cfg",
        omni_data=SimpleNamespace(is_cfg_uncond=True),
        finished=lambda: True,
    )
    scheduler = MiniMaxMusic3Scheduler.__new__(MiniMaxMusic3Scheduler)
    scheduler.prefill_start_done = {"a-cfg"}
    scheduler.session_bridge = None
    scheduler.request_admission_lock = RLock()
    scheduler.completed_request_ids = {}
    scheduler.pending_stream_ingress = {}
    scheduler.aborted_request_ids = set()
    scheduler.first_emit_done = set()
    scheduler.prefill_end_done = set()
    conditioned = queued_request("a")
    scheduler.stream_output([conditioned, twin])
    assert twin.omni_data is None
    scheduler.stream_output([conditioned, twin])
    assert parent_outputs == [[conditioned], [conditioned]]
    assert ended == [("a-cfg", "success")]


@pytest.mark.parametrize("scheduler", [16], indirect=True)
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


@pytest.mark.parametrize("scheduler", [16], indirect=True)
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


def test_abort_while_kv_deferred_removes_both_rows_before_next_admission(
    scheduler: MiniMaxMusic3Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An off-thread abort missed parked rows, which returned to the waiting queue."""
    requests = paired_requests([847, 847], 9001)
    scheduler.waiting_queue = requests.copy()
    allocated = scheduler.token_to_kv_pool_allocator.alloc(
        20000 // scheduler.page_size * scheduler.page_size
    )
    assert allocated is not None
    parked = Event()
    cancelled = Event()

    def cancel_request() -> None:
        assert parked.wait(timeout=10)
        try:
            scheduler.abort(requests[0].rid)
        finally:
            cancelled.set()

    def pause_after_ordering(
        scheduler: MiniMaxMusic3Scheduler, running_batch: ScheduleBatch
    ) -> NextBatchPlan:
        scheduler.policy.calc_priority(scheduler.waiting_queue, running_batch)
        assert scheduler.waiting_queue == []
        parked.set()
        assert cancelled.wait(timeout=10)
        return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", pause_after_ordering)
    monkeypatch.setenv("SGLANG_REQ_WAITING_TIMEOUT", "0")
    monkeypatch.setenv("SGLANG_REQ_RUNNING_TIMEOUT", "0")
    running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        cancellation = executor.submit(cancel_request)
        scheduler.get_new_batch_prefill(running_batch)
        cancellation.result(timeout=10)
    assert scheduler.recv_requests() == []
    assert scheduler.waiting_queue == requests[2:]
    assert scheduler.aborted_request_ids == {request.rid for request in requests[:2]}
    assert scheduler.inbox.empty()
    scheduler.token_to_kv_pool_allocator.free(allocated)
    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", admit_prefill)
    plan = scheduler.get_new_batch_prefill(running_batch)
    assert plan.batch_to_run.reqs == requests[2:]
    assert scheduler.waiting_queue == []


@pytest.mark.parametrize("scheduler", [16], indirect=True)
@pytest.mark.parametrize(
    ("enable_overlap", "is_entry_rank"), [(False, True), (True, True), (True, False)]
)
def test_off_thread_abort_is_forwarded_once_by_entry_rank(
    scheduler: MiniMaxMusic3Scheduler,
    enable_overlap: bool,
    is_entry_rank: bool,
) -> None:
    """Both CFG rows must be cancelled on the scheduler thread from one message."""
    scheduler.enable_overlap = enable_overlap
    scheduler.is_entry_rank = is_entry_rank
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(scheduler.abort, "cancelled").result(timeout=10)
    assert scheduler.aborted_request_ids == set()
    if is_entry_rank:
        message = scheduler.inbox.get_nowait()
        assert message.type == "abort"
        assert message.request_id == "cancelled"
    else:
        pass
    assert scheduler.inbox.empty()


@pytest.mark.parametrize("scheduler", [16], indirect=True)
def test_real_enqueue_stamps_both_rows_for_fcfs_and_waiting_timeout(
    scheduler: MiniMaxMusic3Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real enqueue left both timestamps at zero and waiting timeouts skipped them."""
    scheduler.enable_priority_scheduling = True
    scheduler.schedule_low_priority_values_first = False
    requests = paired_requests([128, 128], 10)
    for offset, enqueue_seconds in ((0, 100.0), (2, 105.0)):
        monkeypatch.setattr(
            "sglang_omni.scheduling.omni_scheduler.time.perf_counter",
            lambda: enqueue_seconds,
        )
        conditioned, unconditioned = requests[offset : offset + 2]
        request_data = MiniMaxMusic3SGLangRequestData(
            req=conditioned,
            enforce_request_limits=False,
            cfg_uncond=MiniMaxMusic3SGLangRequestData(
                req=unconditioned, is_cfg_uncond=True
            ),
        )
        scheduler.enqueue_built_request(
            StagePayload(conditioned.rid, OmniRequest(inputs=None), None),
            False,
            request_data,
        )
    assert [request.time_stats.wait_queue_entry_time for request in requests] == [
        100.0,
        100.0,
        105.0,
        105.0,
    ]
    assert all(request.priority == requests[0].priority for request in requests)
    scheduler.policy.enable_priority_scheduling = True
    scheduler.waiting_queue.reverse()
    scheduler.policy.calc_priority(scheduler.waiting_queue)
    assert scheduler.waiting_queue == requests
    monkeypatch.setenv("SGLANG_REQ_WAITING_TIMEOUT", "10")
    monkeypatch.setenv("SGLANG_REQ_RUNNING_TIMEOUT", "0")
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler.time.perf_counter", lambda: 112.0
    )
    assert [abort.rid for abort in scheduler.poll_request_timeout_aborts()] == [
        request.rid for request in requests[:2]
    ]
