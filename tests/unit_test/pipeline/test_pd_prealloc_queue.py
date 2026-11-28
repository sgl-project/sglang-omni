# SPDX-License-Identifier: Apache-2.0
"""Deterministic tests for DecodeKVReceiver FIFO preallocation, allocator sync, and Decode admin/stream safety."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

from sglang_omni.admission import AdminDroppedRequestError, QueueFullError
from sglang_omni.comm.engine import KVTransferCancelled
from sglang_omni.proto.kv_transfer import KVTransferPrepareMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_receiver import (
    DecodeAdmission,
    DecodeKVReceiver,
    ReservedKV,
)
from sglang_omni.scheduling.pd_scheduler import OmniDecodeScheduler
from sglang_omni.scheduling.pd_utils import (
    continuation_from_req,
    request_from_continuation,
    serialize_kv_allocator,
)
from sglang_omni.serve.openai_errors import generation_error_status_code
from tests.unit_test.fixtures.pd_fakes import (
    FakeCapacityCheck,
    FakeCudaSlots,
    FakeKVAllocator,
    FakeRequestPool,
    StreamTrackingAllocator,
    build_prefill_state,
    install_fake_cuda,
    make_decode_scheduler_stub,
    make_prefill_request,
    make_prepare_message,
)


def prepare(
    rid: str, prompt_tokens: int, page_size: int = 1
) -> KVTransferPrepareMessage:
    return make_prepare_message(
        request_id=rid,
        transfer_id=f"tx-{rid}",
        prompt_tokens=prompt_tokens,
        page_size=page_size,
    )


def make_receiver(
    *,
    kv_size: int = 64,
    row_capacity: int = 4,
    decode_runway_tokens: int = 16,
    capacity_check: FakeCapacityCheck | None = None,
) -> tuple[DecodeKVReceiver, FakeKVAllocator, FakeRequestPool]:
    allocator, pool = FakeKVAllocator(size=kv_size), FakeRequestPool(
        capacity=row_capacity
    )
    receiver = DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=allocator,
        req_to_token_pool=pool,
        admissions=deque(),
        resume_schema="test-v1",
        decode_runway_tokens=decode_runway_tokens,
        capacity_check=capacity_check,
    )
    return receiver, allocator, pool


def test_strict_fifo_and_dual_pool_atomic_reservation() -> None:
    """FIFO prevents line-cutting and blocking on KV or request rows leaks neither pool."""

    async def run() -> None:
        receiver, allocator, pool = make_receiver(
            row_capacity=2, capacity_check=FakeCapacityCheck(active=True)
        )
        held_slots = allocator.alloc(40)
        assert held_slots is not None
        order: list[str] = []

        async def wait_reserve(rid: str, tokens: int) -> None:
            await receiver.reserve(prepare(rid, tokens), timeout_seconds=5.0)
            order.append(rid)

        large_task = asyncio.create_task(wait_reserve("large", 20))
        await asyncio.sleep(0)
        small_task = asyncio.create_task(wait_reserve("small", 4))
        await asyncio.sleep(0)
        assert not large_task.done() and not small_task.done()
        allocator.free(held_slots)
        receiver.notify_capacity_released()
        await asyncio.gather(large_task, small_task)
        assert order == ["large", "small"] and receiver.reserved_request_rows == 2

        allocations_before = allocator.allocation_calls
        third_task = asyncio.create_task(
            receiver.reserve(prepare("third", 4), timeout_seconds=5.0)
        )
        await asyncio.sleep(0)
        assert (
            not third_task.done() and allocator.allocation_calls == allocations_before
        )
        pool.capacity = 3
        receiver.notify_capacity_released()
        await third_task
        assert receiver.reserved_request_rows == 3

    asyncio.run(run())


def test_zero_active_bypass_unblocks_long_prompt_when_last_running_req_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the last running request finishes before filter_batch, zero-active bypass unblocks queued long prompts."""

    async def run() -> None:
        scheduler = make_decode_scheduler_stub(
            allocator=FakeKVAllocator(size=64), decode_runway_tokens=32
        )
        running_req = make_prefill_request("running", prompt_tokens=16)
        scheduler.req_to_token_pool.alloc([running_req])
        running_slots = scheduler.token_to_kv_pool_allocator.alloc(16)
        assert running_slots is not None
        scheduler.running_batch = SimpleNamespace(
            reqs=[running_req], is_empty=lambda: False
        )
        waiter_task = asyncio.create_task(
            scheduler.pd_receiver.reserve(prepare("long", 48), timeout_seconds=5.0)
        )
        await asyncio.sleep(0)
        assert not waiter_task.done()

        def finish_last_request(
            self: OmniDecodeScheduler,
            batch: ScheduleBatch,
            result: GenerationBatchResult,
        ) -> None:
            running_req.finished_reason = FINISH_LENGTH(1)
            self.token_to_kv_pool_allocator.free(running_slots)
            self.req_to_token_pool.free(running_req)

        monkeypatch.setattr(OmniScheduler, "process_batch_result", finish_last_request)
        scheduler.process_batch_result(scheduler.running_batch, SimpleNamespace())
        assert (
            len(scheduler.running_batch.reqs) == 1
            and not scheduler.has_active_decode_requests()
        )
        assert (
            len((await asyncio.wait_for(waiter_task, timeout=2.0)).page_indices) == 48
        )

    asyncio.run(run())


def test_permanent_oversize_maps_to_http_400_and_invalid_timeout_fails_fast() -> None:
    """Oversize transfers fail fast with a 400-classified ValueError; non-positive timeout fails fast."""

    async def run() -> None:
        receiver, allocator, _ = make_receiver(kv_size=16, decode_runway_tokens=8)
        with pytest.raises(ValueError) as exc_info:
            await receiver.reserve(prepare("big", 20), timeout_seconds=60.0)
        assert generation_error_status_code(RuntimeError(str(exc_info.value))) == 400
        assert not receiver.waiters and allocator.allocation_calls == 0
        with pytest.raises(ValueError, match="timeout_seconds"):
            await receiver.reserve(prepare("ok", 4), timeout_seconds=0.0)

    asyncio.run(run())


@pytest.mark.parametrize("unblock_mode", ["timeout", "cancel_request", "suspend"])
def test_waiter_unblocking_and_admin_suspension(unblock_mode: str) -> None:
    """Timeout, per-request abort, and admin suspension fail queued waiters cleanly and preserve fairness."""

    async def run() -> None:
        capacity = FakeCapacityCheck(active=True)
        receiver, allocator, _ = make_receiver(capacity_check=capacity)
        held_slots = allocator.alloc(40)
        assert held_slots is not None
        head_timeout = 0.02 if unblock_mode == "timeout" else 5.0
        head_task = asyncio.create_task(
            receiver.reserve(prepare("head", 20), timeout_seconds=head_timeout)
        )
        await asyncio.sleep(0)
        tail_msg = prepare("tail", 4)
        tail_task = asyncio.create_task(receiver.reserve(tail_msg, timeout_seconds=5.0))
        await asyncio.sleep(0)
        assert not tail_task.done()

        if unblock_mode == "timeout":
            with pytest.raises(QueueFullError) as exc_info:
                await head_task
            assert str(exc_info.value).startswith(
                QueueFullError.MESSAGE
            ) and QueueFullError.matches(exc_info.value)
            assert (
                len((await asyncio.wait_for(tail_task, timeout=2.0)).page_indices) == 4
            )
        elif unblock_mode == "cancel_request":
            receiver.cancel_request_waiters("head")
            with pytest.raises(KVTransferCancelled, match="head"):
                await head_task
            assert (
                len((await asyncio.wait_for(tail_task, timeout=2.0)).page_indices) == 4
            )
        else:
            with receiver.suspend_reservations():
                assert not receiver.waiters
                with pytest.raises(AdminDroppedRequestError) as head_exc:
                    await head_task
                assert AdminDroppedRequestError.matches(head_exc.value)
                for coro in (
                    tail_task,
                    receiver.reserve(tail_msg, timeout_seconds=5.0),
                ):
                    with pytest.raises(AdminDroppedRequestError):
                        await coro
            allocator.free(held_slots)
            capacity.active = False
            assert (
                len(
                    (await receiver.reserve(tail_msg, timeout_seconds=5.0)).page_indices
                )
                == 4
            )

    asyncio.run(run())


def test_same_tick_set_result_cancel_race_rolls_back_kv_and_row_credit() -> None:
    """Cancelling reserve in the same tick drain_waiters resolves it rolls back KV and row credit."""

    async def run() -> None:
        receiver, allocator, _ = make_receiver(
            row_capacity=1, capacity_check=FakeCapacityCheck(active=True)
        )
        first = prepare("first", 8)
        first_dest = await receiver.reserve(first, timeout_seconds=5.0)
        raced_task = asyncio.create_task(
            receiver.reserve(prepare("raced", 8), timeout_seconds=5.0)
        )
        await asyncio.sleep(0)
        third_task = asyncio.create_task(
            receiver.reserve(prepare("third", 8), timeout_seconds=5.0)
        )
        await asyncio.sleep(0)

        receiver.abort(first, first_dest, RuntimeError("done"))
        receiver.drain_waiters()
        assert "tx-raced" in receiver.reservations
        raced_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await raced_task

        assert len((await asyncio.wait_for(third_task, timeout=2.0)).page_indices) == 8
        assert (
            "tx-raced" not in receiver.reservations
            and "tx-raced" in receiver.transfer_tombstones
        )
        assert receiver.reserved_request_rows == 1 and allocator.available_size() == 56

    asyncio.run(run())


def test_allocator_stream_sync_device_guard_and_record_stream_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """serialize_kv_allocator creates its sync event on the allocator device, orders cross-stream mutations, and rebuild records stream before write."""
    cuda = install_fake_cuda(monkeypatch)
    allocator = FakeKVAllocator(size=16, page_size=1, device="cuda:1")
    serialize_kv_allocator(allocator, sync_cuda_streams=True)
    assert cuda.event_devices == ["cuda:1"]
    assert allocator.available_size() == 16 and not cuda.events_log

    # First mutation on compute_stream records without waiting.
    slots = allocator.alloc(2)
    assert slots is not None
    assert cuda.events_log == [("record", "compute")]
    cuda.events_log.clear()

    # Cross-stream mutation on alloc_stream waits on the prior event, then records.
    cuda.active_stream = cuda.alloc_stream
    assert allocator.alloc(2) is not None
    assert cuda.events_log == [("wait", "alloc"), ("record", "alloc")]
    cuda.events_log.clear()

    # Switching back to compute_stream waits once on stream change; subsequent same-stream mutations only record.
    cuda.active_stream = cuda.compute_stream
    allocator.free_group_begin()
    assert cuda.events_log == [("wait", "compute"), ("record", "compute")]
    cuda.events_log.clear()
    allocator.free(slots)
    assert cuda.events_log == [("record", "compute")]
    cuda.events_log.clear()
    allocator.free_group_end()
    assert cuda.events_log == [("record", "compute")]

    call_order: list[str] = []
    pool = FakeRequestPool(capacity=4)
    raw_write = pool.write
    pool.write = lambda row_slice, value: (
        call_order.append("write"),
        raw_write(row_slice, value),
    )
    fake_slots = FakeCudaSlots(
        torch.tensor([16, 17, 18, 19], dtype=torch.int64),
        cuda.compute_stream,
        call_order,
    )
    continuation = continuation_from_req(
        make_prefill_request("req-cuda", prompt_tokens=3),
        "tx-cuda",
        build_prefill_state,
    )
    request_from_continuation(
        continuation,
        ReservedKV(slots=fake_slots, page_indices=(1,), sequence_length=3),
        req_to_token_pool=pool,
        state_restorer=lambda restored, data, resume: None,
    )
    assert call_order == ["record_stream", "write"]


def test_process_batch_result_locks_free_group_and_abort_frees_on_allocation_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """process_batch_result holds pd_lifecycle_lock across free_group, and receiver/cleanup free on allocation_stream."""
    scheduler = make_decode_scheduler_stub()
    message = prepare("lock", 4)
    destination = asyncio.run(
        scheduler.pd_receiver.reserve(message, timeout_seconds=5.0)
    )
    in_group, release_group, abort_started, abort_done = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def fake_pbr(
        self: OmniDecodeScheduler, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        self.token_to_kv_pool_allocator.free_group_begin()
        in_group.set()
        assert release_group.wait(timeout=5.0)
        self.token_to_kv_pool_allocator.free_group_end()

    monkeypatch.setattr(OmniScheduler, "process_batch_result", fake_pbr)
    with ThreadPoolExecutor(max_workers=2) as threads:
        p_fut = threads.submit(
            scheduler.process_batch_result, SimpleNamespace(reqs=[]), SimpleNamespace()
        )
        assert in_group.wait(timeout=5.0)
        a_fut = threads.submit(
            lambda: (
                abort_started.set(),
                scheduler.pd_receiver.abort(
                    message, destination, RuntimeError("aborted")
                ),
                abort_done.set(),
            )
        )
        assert abort_started.wait(timeout=5.0) and not abort_done.wait(timeout=0.05)
        release_group.set()
        p_fut.result(timeout=5.0)
        a_fut.result(timeout=5.0)
    assert scheduler.token_to_kv_pool_allocator.available_size() == 64

    cuda = install_fake_cuda(monkeypatch)
    tracking = StreamTrackingAllocator(cuda.stream_stack)
    stream_sched = make_decode_scheduler_stub(allocator=tracking)
    for path in ("abort", "aborted_admit", "discard"):
        msg = prepare(path, 4)
        dest = asyncio.run(stream_sched.pd_receiver.reserve(msg, timeout_seconds=5.0))
        if path == "abort":
            stream_sched.pd_receiver.abort(msg, dest, RuntimeError("abort"))
        elif path == "aborted_admit":
            stream_sched.pd_receiver.commit(msg, dest)
            stream_sched.aborted_request_ids.add(path)
            assert stream_sched.drain_decode_admissions() is True
        else:
            stream_sched.pd_receiver.commit(msg, dest)
            monkeypatch.setattr(
                OmniScheduler, "discard_pending_request_admissions", lambda self: None
            )
            stream_sched.discard_pending_request_admissions()
    assert (
        tracking.observed
        == [("alloc", "allocation_stream"), ("free", "allocation_stream")] * 3
    )


def test_decode_pd_admissions_visible_in_active_and_drained_by_admin_abort() -> None:
    """Committed pd_admissions appear in active_request_ids and drain cleanly on admin pause/weight-update abort."""
    scheduler = make_decode_scheduler_stub(row_capacity=4)
    first_msg = prepare("pause", 4)
    scheduler.pd_receiver.commit(
        first_msg,
        asyncio.run(scheduler.pd_receiver.reserve(first_msg, timeout_seconds=5.0)),
    )
    assert scheduler.active_request_ids() == ["pause"]
    pause_result = scheduler.admin_pause_generation({"mode": "abort"})
    assert (
        pause_result["success"] is True
        and pause_result["data"]["num_paused_requests"] == 1
    )
    assert not scheduler.pd_admissions and not scheduler.pd_holds_kv()
    assert scheduler.pd_receiver.reserved_request_rows == 0
    assert scheduler.token_to_kv_pool_allocator.available_size() == 64
    error_msg = scheduler.outbox.get_nowait()
    assert scheduler.outbox.empty()
    assert (
        error_msg.type == "error"
        and error_msg.request_id == "pause"
        and isinstance(error_msg.data, AdminDroppedRequestError)
    )

    scheduler.admin_continue_generation({"torch_empty_cache": False})
    second_msg = prepare("weight", 4)
    scheduler.pd_receiver.commit(
        second_msg,
        asyncio.run(scheduler.pd_receiver.reserve(second_msg, timeout_seconds=5.0)),
    )
    update_fn = Mock(return_value=(True, "updated"))
    assert (
        scheduler.run_weight_update_with_lifecycle(
            {"abort_all_requests": False, "flush_cache": False}, update_fn, {}
        )["success"]
        is False
    )
    update_fn.assert_not_called()
    assert (
        scheduler.run_weight_update_with_lifecycle(
            {"abort_all_requests": True, "flush_cache": False}, update_fn, {}
        )["success"]
        is True
    )
    update_fn.assert_called_once()
    assert (
        not scheduler.pd_holds_kv() and scheduler.pd_receiver.reserved_request_rows == 0
    )


@pytest.mark.accelerator
def test_cuda_paged_allocator_stream_sync_and_record_stream_round_trip() -> None:
    """Real CUDA paged allocator exercises cross-stream event sync, allocation_stream, and record_stream."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA accelerator is required")
    device, page_size = ("cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"), 16
    allocator = PagedTokenToKVPoolAllocator(
        size=page_size * 8,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=device,
        kvcache=None,
        need_sort=False,
    )
    pool = ReqToTokenPool(
        size=4, max_context_len=64, device=device, enable_memory_saver=False
    )
    admissions: deque[DecodeAdmission] = deque()
    receiver = DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=allocator,
        req_to_token_pool=pool,
        admissions=admissions,
        resume_schema="test-v1",
        lifecycle_lock=serialize_kv_allocator(allocator, sync_cuda_streams=True),
        page_size=page_size,
    )
    message = prepare("gpu", 35, page_size=page_size)
    receiver.commit(
        message, asyncio.run(receiver.reserve(message, timeout_seconds=5.0))
    )
    admission = admissions.popleft()
    assert admission.allocation.slots.is_cuda
    with torch.cuda.device(device):
        rebuilt = request_from_continuation(
            admission.continuation,
            admission.allocation,
            req_to_token_pool=pool,
            state_restorer=lambda restored, data, resume: None,
        )
        torch.cuda.synchronize(device)
        written = pool.req_to_token[rebuilt.kv.req_pool_idx, :35]
        assert written.tolist() == admission.allocation.slots[:35].tolist()
        allocator.free_segment(written, start_pos=0)
        pool.free(rebuilt)
        torch.cuda.synchronize(device)
    assert (
        allocator.available_size() == page_size * 8
        and receiver.reserved_request_rows == 0
    )
