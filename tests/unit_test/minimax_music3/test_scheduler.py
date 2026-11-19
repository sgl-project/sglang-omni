# SPDX-License-Identifier: Apache-2.0
"""Music3 CFG admission across input and full-song KV limits."""

from __future__ import annotations

from typing import Literal
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.runtime_context import get_context, get_schedule

from sglang_omni.models.minimax_music3.scheduler import MiniMaxMusic3Scheduler
from sglang_omni.models.minimax_music3.serial_offload import SerialOffloadCoordinator
from sglang_omni.models.minimax_music3.sglang_request_builder import (
    MiniMaxMusic3SGLangRequestData,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from tests.unit_test.fixtures.minimax_music3_scheduler import (
    admit_prefill,
    paired_requests,
)

pytest_plugins = ("tests.unit_test.fixtures.minimax_music3_scheduler",)


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


@pytest.mark.parametrize("scheduler", [16], indirect=True)
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


def test_sorted_pairs_wait_for_running_kv_then_resume(
    scheduler: MiniMaxMusic3Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sorting must preserve full decode reservations and the deferred queue."""
    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", admit_prefill)
    scheduler.policy.enable_priority_scheduling = True
    running_requests = paired_requests([697], 9001)
    allocated_pages = []
    for request, request_id in zip(running_requests, ("running", "running-cfg")):
        request.rid = request_id
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


@pytest.mark.parametrize("exit_after_ordering", ["none", "empty", "error"])
@pytest.mark.parametrize("existing_owner", [False, True])
def test_deferred_pairs_and_input_budget_restored_on_early_exit(
    scheduler: MiniMaxMusic3Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    exit_after_ordering: Literal["none", "empty", "error"],
    existing_owner: bool,
) -> None:
    """An early exit after ordering must not lose parked pairs or enlarged budgets."""
    requests = paired_requests([128, 128], 10)
    scheduler.waiting_queue = requests.copy()
    scheduler.max_prefill_tokens = 100
    coordinator = SerialOffloadCoordinator()
    coordinator.register_ar(torch.nn.Linear(2, 2), torch.device("cpu"))
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.scheduler.get_coordinator",
        lambda: coordinator,
    )
    if existing_owner:
        assert coordinator.try_acquire_ar(requests[0].rid)
    else:
        pass

    def stop_after_ordering(
        scheduler: MiniMaxMusic3Scheduler, running_batch: ScheduleBatch
    ) -> NextBatchPlan:
        scheduler.policy.calc_priority(scheduler.waiting_queue, running_batch)
        if exit_after_ordering == "error":
            raise RuntimeError("Prefill interrupted")
        elif exit_after_ordering == "empty":
            return NextBatchPlan(
                batch_to_run=ScheduleBatch(reqs=[], batch_is_full=False),
                running_batch=running_batch,
            )
        else:
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", stop_after_ordering)
    running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
    if exit_after_ordering == "error":
        with pytest.raises(RuntimeError, match="Prefill interrupted"):
            scheduler.get_new_batch_prefill(running_batch)
    else:
        plan = scheduler.get_new_batch_prefill(running_batch)
        assert plan.batch_to_run is None or plan.batch_to_run.reqs == []
    assert scheduler.waiting_queue == requests
    assert scheduler.max_prefill_tokens == 100
    if existing_owner:
        assert coordinator.owner_request_id == requests[0].rid
    else:
        assert coordinator.owner_request_id is None
        assert coordinator.try_acquire_ar("next-request")


def test_rejected_prefill_releases_owner_and_allows_next_request(
    scheduler: MiniMaxMusic3Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pair rejected by the adder retained AR ownership and blocked later requests."""
    coordinator = SerialOffloadCoordinator()
    coordinator.register_ar(torch.nn.Linear(2, 2), torch.device("cpu"))
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.scheduler.get_coordinator",
        lambda: coordinator,
    )
    monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", admit_prefill)
    requests = paired_requests([128, 128], 10)
    scheduler.waiting_queue = requests.copy()
    scheduler.max_prefill_tokens = 100
    allocated_pages: list[torch.Tensor] = []
    verdicts: list[AddReqResult] = []
    add_one_request = PrefillAdder.add_one_req

    def consume_capacity_before_admission(
        adder: PrefillAdder,
        req: Req,
        has_chunked_req: bool,
        truncation_align_size: int | None,
    ) -> AddReqResult:
        assert coordinator.owner_request_id == requests[0].rid
        allocator = scheduler.token_to_kv_pool_allocator
        allocation = allocator.alloc(allocator.available_size())
        assert allocation is not None
        allocated_pages.append(allocation)
        verdict = add_one_request(adder, req, has_chunked_req, truncation_align_size)
        verdicts.append(verdict)
        return verdict

    running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
    with monkeypatch.context() as admission_patch:
        admission_patch.setattr(
            PrefillAdder, "add_one_req", consume_capacity_before_admission
        )
        plan = scheduler.get_new_batch_prefill(running_batch)
    assert verdicts == [AddReqResult.NO_TOKEN]
    assert plan.batch_to_run.reqs == []
    assert scheduler.waiting_queue == requests
    assert scheduler.max_prefill_tokens == 100
    assert coordinator.owner_request_id is None
    for allocation in allocated_pages:
        scheduler.token_to_kv_pool_allocator.free(allocation)
    scheduler.abort(requests[0].rid)
    assert scheduler.waiting_queue == requests[2:]
    plan = scheduler.get_new_batch_prefill(running_batch)
    assert plan.batch_to_run.reqs == requests[2:]
    assert scheduler.waiting_queue == []
    assert coordinator.owner_request_id == requests[2].rid


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
    prefill_max_requests: int | None,
    expected_error: str | None,
) -> None:
    """Real enqueue must reject impossible pairs before either row is queued."""
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
        allocated = scheduler.token_to_kv_pool_allocator.alloc(
            20000 // scheduler.page_size * scheduler.page_size
        )
        assert allocated is not None
        assert scheduler.request_kv_capacity_error(cond) is None
        monkeypatch.setattr(OmniScheduler, "get_new_batch_prefill", admit_prefill)
        running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        assert scheduler.get_new_batch_prefill(running_batch).batch_to_run.reqs == []
        assert scheduler.waiting_queue == [cond, uncond]
        scheduler.token_to_kv_pool_allocator.free(allocated)
        assert scheduler.get_new_batch_prefill(running_batch).batch_to_run.reqs == [
            cond,
            uncond,
        ]
        assert scheduler.waiting_queue == []
    assert scheduler.outbox.empty()


@pytest.mark.parametrize(
    ("session_capacity_checked", "decode_path"),
    [(False, "plain"), (True, "spec"), (False, None)],
)
@pytest.mark.parametrize("scheduler", [16], indirect=True)
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
