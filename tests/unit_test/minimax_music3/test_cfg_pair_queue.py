# SPDX-License-Identifier: Apache-2.0
"""Queue order and model path events of MiniMax Music 3 CFG row pairs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import CacheAgnosticPolicy
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.minimax_music3.scheduler import MiniMaxMusic3Scheduler
from sglang_omni.models.minimax_music3.sglang_request_builder import (
    MiniMaxMusic3SGLangRequestData,
    build_sglang_minimax_request,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler, UpstreamAbortSender
from tests.unit_test.fixtures.minimax_music3_scheduler import (
    admit_prefill,
    paired_requests,
)

pytest_plugins = ("tests.unit_test.fixtures.minimax_music3_scheduler",)


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
    conditioned = Req(
        rid="a",
        origin_input_text="",
        origin_input_ids=[1, 2],
        sampling_params=SamplingParams(max_new_tokens=1),
    )
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

    def separate_pairs(waiting_queue: list[Req]) -> None:
        shuffled_queues.append(waiting_queue.copy())
        waiting_queue[:] = waiting_queue[::-2] + waiting_queue[-2::-2]

    monkeypatch.setattr(
        "sglang.srt.managers.schedule_policy.random.shuffle", separate_pairs
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
    assert scheduler.recv_requests() == []
    assert scheduler.waiting_queue == requests[2:]
    assert scheduler.aborted_request_ids == {request.rid for request in requests[:2]}
    error = scheduler.outbox.get_nowait()
    assert error.request_id == requests[0].rid
    assert "waiting timeout" in str(error.data)
    assert scheduler.outbox.empty()


@pytest.mark.parametrize("scheduler", [16], indirect=True)
def test_rejected_external_cfg_id_cannot_cancel_a_live_song(
    scheduler: MiniMaxMusic3Scheduler,
) -> None:
    """An invalid caller id must neither abort its prefix nor lose the rejection."""
    primary, twin = paired_requests([128], 10)
    scheduler.waiting_queue = [primary, twin]
    scheduler.request_build_executor = None
    tokenizer = Mock()
    scheduler.request_builder = lambda payload: build_sglang_minimax_request(
        payload, tokenizer
    )
    invalid = StagePayload(twin.rid, OmniRequest(inputs="A verse"), None)

    scheduler.process_input_requests([invalid])
    scheduler.abort(invalid.request_id)

    error = scheduler.outbox.get_nowait()
    assert error.request_id == invalid.request_id
    assert "suffix is reserved" in str(error.data)
    assert scheduler.outbox.empty()
    assert scheduler.waiting_queue == [primary, twin]
    assert scheduler.aborted_request_ids == set()
    tokenizer.assert_not_called()


@pytest.mark.parametrize("scheduler", [16], indirect=True)
def test_upstream_abort_for_internal_row_ends_song_once(
    scheduler: MiniMaxMusic3Scheduler,
) -> None:
    """An upstream failure for a known twin retires its main request visibly."""
    primary, twin = paired_requests([128], 10)
    scheduler.waiting_queue = [primary, twin]
    sender = UpstreamAbortSender(scheduler)
    sender.send_output(AbortReq(rid=twin.rid, abort_message="admission failed"), twin)

    error = scheduler.outbox.get_nowait()
    assert error.request_id == primary.rid
    assert str(error.data) == "admission failed"
    assert scheduler.outbox.empty()
    assert scheduler.waiting_queue == []
    assert scheduler.aborted_request_ids == {primary.rid, twin.rid}
