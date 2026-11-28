# SPDX-License-Identifier: Apache-2.0
"""Deterministic tests for Prefill handoff backpressure, chunk reservation, parked abort, and ready_event."""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import sglang.srt.managers.schedule_policy as schedule_policy_module
import torch
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.prefill_budget import PrefillBudget

from sglang_omni.admission import AdminDroppedRequestError
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_scheduler import (
    LeasedPrefillChunkReserveAdder,
    OmniPrefillScheduler,
    PDKVLifecycle,
    PrefillChunkReserveAdder,
)
from sglang_omni.scheduling.pd_utils import SGLangKVLease
from tests.unit_test.fixtures.pd_fakes import (
    FakeTreeCache,
    build_prefill_state,
    install_fake_cuda,
    make_paged_allocator,
    make_prefill_request,
    make_prefill_scheduler_stub,
)


def make_adder(
    adder_type: type[PrefillChunkReserveAdder],
    allocator: PagedTokenToKVPoolAllocator,
) -> PrefillChunkReserveAdder:
    return adder_type(
        page_size=16,
        tree_cache=FakeTreeCache(),
        token_to_kv_pool_allocator=allocator,
        running_batch=SimpleNamespace(reqs=[], batch_size=lambda: 0),
        new_token_ratio=0.5,
        rem_input_tokens=1024,
        rem_chunk_tokens=32,
    )


@pytest.mark.parametrize(
    (
        "base_budget",
        "max_in_flight",
        "outstanding",
        "running_bs",
        "held_rows",
        "expected",
    ),
    [
        (10, 4, 0, 0, 0, 4),
        (10, 4, 2, 1, 0, 1),
        (10, 4, 3, 0, 1, 1),
        (10, 4, 4, 0, 1, 0),
        (10, 4, 5, 1, 0, 0),
        (1, 4, 0, 0, 0, 1),
    ],
)
def test_prefill_get_num_allocatable_reqs_caps_by_in_flight_handoffs(
    monkeypatch: pytest.MonkeyPatch,
    base_budget: int,
    max_in_flight: int,
    outstanding: int,
    running_bs: int,
    held_rows: int,
    expected: int,
) -> None:
    """get_num_allocatable_reqs caps at max_in_flight_handoffs - outstanding - running_bs."""
    scheduler = make_prefill_scheduler_stub(max_in_flight_handoffs=max_in_flight)
    scheduler.pd_outstanding_releases = {f"req-{index}" for index in range(outstanding)}
    monkeypatch.setattr(
        OmniScheduler,
        "get_num_allocatable_reqs",
        lambda self, *args, **kwargs: base_budget,
    )
    assert (
        scheduler.get_num_allocatable_reqs(running_bs, num_held_rows=held_rows)
        == expected
    )


def test_prefill_batch_is_full_resets_only_when_handoff_budget_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """batch_is_full stays True while handoffs are at cap even if req_to_token_pool has free rows."""
    scheduler = make_prefill_scheduler_stub(max_in_flight_handoffs=2)
    scheduler.running_batch.batch_is_full = True
    scheduler.pd_outstanding_releases = {"req-1", "req-2"}
    monkeypatch.setattr(
        OmniScheduler, "get_num_allocatable_reqs", lambda self, *args, **kwargs: 8
    )
    monkeypatch.setattr(OmniScheduler, "get_next_batch_to_run", lambda self: None)

    scheduler.get_next_batch_to_run()
    assert scheduler.running_batch.batch_is_full is True
    scheduler.pd_due_releases.put(SimpleNamespace(rid="req-1"))
    scheduler.get_next_batch_to_run()
    assert (
        scheduler.pd_outstanding_releases == {"req-2"}
        and not scheduler.running_batch.batch_is_full
    )


def test_prefill_chunk_reserve_adder_parks_when_leases_pin_kv_and_reserves_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LeasedPrefillChunkReserveAdder parks chunked_req under lease pressure and reserves unscheduled tail pages."""
    monkeypatch.setattr(
        schedule_policy_module,
        "get_exec",
        lambda: SimpleNamespace(
            features=SimpleNamespace(enable_encoder_swa_bounded_replay=False)
        ),
    )
    scheduler = make_prefill_scheduler_stub()
    assert scheduler.get_prefill_adder_class() is PrefillChunkReserveAdder
    scheduler.pd_outstanding_releases.add("leased-1")
    assert scheduler.get_prefill_adder_class() is LeasedPrefillChunkReserveAdder

    allocator = make_paged_allocator(pages=4)
    pinned = allocator.alloc(64)
    assert pinned is not None
    adder = make_adder(LeasedPrefillChunkReserveAdder, allocator)
    chunked = make_prefill_request(
        "req-chunked", prompt_tokens=64, with_first_output=False
    )
    chunked.prefix_len = 16
    assert adder.add_chunked_req(chunked) is chunked
    assert adder.can_run_list == [] and adder.rem_chunk_tokens == 0
    assert adder.budget_state() == AddReqResult.OTHER

    allocator.free(pinned[:16])
    adder_clamped = make_adder(LeasedPrefillChunkReserveAdder, allocator)
    assert adder_clamped.add_chunked_req(chunked) is chunked
    assert chunked.extend_end == 32 and adder_clamped.can_run_list == [chunked]
    assert adder_clamped.memory_budget.remaining_current <= 0

    adder_new = make_adder(PrefillChunkReserveAdder, make_paged_allocator(pages=6))
    assert isinstance(adder_new.memory_budget, PrefillBudget)
    first = make_prefill_request(
        "req-new", prompt_tokens=64, max_new_tokens=1, with_first_output=False
    )
    first.last_node = None
    assert adder_new.add_one_req(first, False, None) == AddReqResult.OTHER
    assert adder_new.new_chunked_req is first and first.extend_end == 32
    assert adder_new.memory_budget.remaining_current == 16

    bystander = make_prefill_request(
        "req-bystander", prompt_tokens=32, max_new_tokens=1, with_first_output=False
    )
    bystander.last_node = None
    adder_new.rem_chunk_tokens = 32
    assert adder_new.add_one_req(bystander, True, None) == AddReqResult.NO_TOKEN


def test_parked_chunked_request_abort_finalizes_bookkeeping_and_wakes_off_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parked chunked_req abort clears prefill bookkeeping, fires callbacks once without duplication, and wakes off-thread."""
    monkeypatch.setattr(OmniScheduler, "get_next_batch_to_run", lambda self: None)
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler._emit_model_path_end", Mock()
    )

    finished_cb = Mock()
    scheduler = make_prefill_scheduler_stub(request_finished_callback=finished_cb)
    parked = make_prefill_request(
        "req-parked-1", prompt_tokens=64, with_first_output=False
    )
    scheduler.req_to_token_pool.alloc([parked])
    scheduler.chunked_req = parked
    scheduler.prefill_start_done.add("req-parked-1")
    scheduler.prefill_end_done.add("req-parked-1")
    assert scheduler.active_request_ids() == ["req-parked-1"]

    scheduler.abort("req-parked-1", defer_running_cleanup=True)
    assert (
        isinstance(parked.to_finish, FINISH_ABORT)
        and scheduler.inbox.get_nowait().type == "wakeup"
    )
    scheduler.get_next_batch_to_run()
    assert (
        scheduler.chunked_req is None and parked.finished() and parked.omni_data is None
    )
    assert "req-parked-1" in scheduler.completed_request_ids
    assert (
        "req-parked-1" not in scheduler.prefill_start_done | scheduler.prefill_end_done
    )
    scheduler.release_request_kv_cache.assert_called_once_with(parked)
    finished_cb.assert_called_once_with("req-parked-1")
    scheduler.abort_callback.assert_called_once_with("req-parked-1")

    # Race where super().abort() already ran abort_callback (to_finish is None) must not double-fire it.
    raced_cb = Mock()
    raced_scheduler = make_prefill_scheduler_stub(request_finished_callback=raced_cb)
    raced = make_prefill_request("req-raced", prompt_tokens=64, with_first_output=False)
    raced_scheduler.req_to_token_pool.alloc([raced])
    raced_scheduler.aborted_request_ids.add("req-raced")
    raced_scheduler.prefill_start_done.add("req-raced")
    raced_scheduler.chunked_req = raced
    raced_scheduler.get_next_batch_to_run()
    assert (
        raced_scheduler.chunked_req is None
        and "req-raced" not in raced_scheduler.prefill_start_done
    )
    raced_cb.assert_called_once_with("req-raced")
    raced_scheduler.abort_callback.assert_not_called()

    with ThreadPoolExecutor(max_workers=1) as threads:
        threads.submit(raced_scheduler.abort, "req-other").result(timeout=5.0)
    assert raced_scheduler.inbox.get_nowait().type == "wakeup"

    scheduler_all = make_prefill_scheduler_stub()
    parked_all = make_prefill_request(
        "req-all", prompt_tokens=64, with_first_output=False
    )
    scheduler_all.req_to_token_pool.alloc([parked_all])
    scheduler_all.chunked_req = parked_all
    assert (
        scheduler_all.abort_all_requests() == ["req-all"]
        and scheduler_all.chunked_req is None
    )
    assert isinstance(scheduler_all.outbox.get_nowait().data, AdminDroppedRequestError)


def test_kv_lease_release_coalesces_inbox_wakeups() -> None:
    """Multiple off-thread lease releases coalesce into one empty-id wakeup until reset_pd_wakeup."""
    scheduler = make_prefill_scheduler_stub()
    leases = [
        SGLangKVLease(
            SimpleNamespace(rid=f"req-{index}"),
            scheduler.pd_due_releases,
            wakeup_notifier=scheduler,
        )
        for index in range(12)
    ]
    for lease in leases[:8]:
        lease.release()
    assert scheduler.inbox.get_nowait().type == "wakeup" and scheduler.inbox.empty()
    scheduler.reset_pd_wakeup()
    for lease in leases[8:]:
        lease.release()
    assert scheduler.inbox.get_nowait().type == "wakeup" and scheduler.inbox.empty()


@pytest.mark.parametrize("partner_stage", [None, "decode"])
def test_handoff_records_single_ready_event_on_current_stream(
    monkeypatch: pytest.MonkeyPatch, partner_stage: str | None
) -> None:
    """handoff_prefilled_requests records one interprocess Event on current_stream(device) under torch.cuda.device."""
    cuda = install_fake_cuda(monkeypatch)
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler._emit_model_path_end", Mock()
    )
    scheduler = make_prefill_scheduler_stub(
        partner_stage=partner_stage, page_size=16, device="cuda:0"
    )
    first_req = make_prefill_request("req-1", prompt_tokens=18, with_first_output=True)
    second_req = make_prefill_request("req-2", prompt_tokens=16, with_first_output=True)
    scheduler.req_to_token_pool.alloc([first_req, second_req])
    scheduler.req_to_token_pool.req_to_token[first_req.kv.req_pool_idx, :18] = (
        torch.cat(
            [
                torch.arange(16, 32, dtype=torch.int64),
                torch.arange(48, 50, dtype=torch.int64),
            ]
        )
    )
    scheduler.req_to_token_pool.req_to_token[second_req.kv.req_pool_idx, :16] = (
        torch.arange(64, 80, dtype=torch.int64)
    )
    batch = SimpleNamespace(reqs=[first_req, second_req], batch_is_full=True)
    scheduler.handoff_prefilled_requests(batch, {id(first_req), id(second_req)})

    assert cuda.entered_devices == ["cuda:0"] and len(cuda.created_events) == 1
    assert cuda.created_events[0].interprocess is True
    assert cuda.created_events[0].recorded_streams == [cuda.compute_stream]
    first_msg, second_msg = scheduler.outbox.get_nowait(), scheduler.outbox.get_nowait()
    assert first_msg.data.ready_event is cuda.created_events[0]
    assert second_msg.data.ready_event is cuda.created_events[0]
    assert first_msg.data.source_page_indices == (
        1,
        3,
    ) and second_msg.data.source_page_indices == (4,)
    assert first_msg.data.target_pool_id == "kv"
    assert first_msg.data.to_stage == partner_stage


def test_prefill_init_warns_only_when_req_pool_smaller_than_max_in_flight_handoffs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """OmniPrefillScheduler warns when req_to_token_pool.size < max_in_flight_handoffs and stays silent otherwise."""
    mha_pool = MHATokenToKVPool(
        size=64,
        page_size=16,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=32,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size=64,
        page_size=16,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=mha_pool,
        need_sort=False,
    )
    active_pool = [
        ReqToTokenPool(
            size=4, max_context_len=64, device="cpu", enable_memory_saver=False
        )
    ]

    def fake_init(self: PDKVLifecycle, *args: int, **kwargs: str) -> None:
        self.pd_due_releases, self.pd_outstanding_releases = queue.SimpleQueue(), set()
        self.pd_wakeup_lock, self.pd_wakeup_pending = threading.Lock(), False
        self.session_adapter, self.req_to_token_pool = None, active_pool[0]
        self.token_to_kv_pool_allocator = allocator
        self.server_args = SimpleNamespace(disable_radix_cache=True)
        self.tp_size, self.page_size, self.async_prefill = 1, 16, False
        self.spec_algorithm = SimpleNamespace(is_none=lambda: True)

    monkeypatch.setattr(PDKVLifecycle, "__init__", fake_init)
    with pytest.raises(ValueError, match="max_in_flight_handoffs"):
        OmniPrefillScheduler(
            stage_name="prefill",
            state_builder=build_prefill_state,
            max_in_flight_handoffs=0,
        )
    with caplog.at_level(logging.WARNING):
        OmniPrefillScheduler(
            stage_name="prefill",
            state_builder=build_prefill_state,
            max_in_flight_handoffs=32,
        )
    assert any(
        "smaller than max_in_flight_handoffs" in rec.message for rec in caplog.records
    )
    caplog.clear()
    active_pool[0] = ReqToTokenPool(
        size=32, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    with caplog.at_level(logging.WARNING):
        OmniPrefillScheduler(
            stage_name="prefill",
            state_builder=build_prefill_state,
            max_in_flight_handoffs=32,
        )
    assert not caplog.records
