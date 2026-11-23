# SPDX-License-Identifier: Apache-2.0
"""PD ownership and routing regressions, without model weights or GPU kernels."""

import asyncio
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler as _Upstream
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.managers.scheduler_components.state_check import (
    CheckPhase,
    StateCheckError,
)
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.observability.metrics_collector import SchedulerStats
from sglang.srt.observability.scheduler_stage_metrics import (
    SchedulerStageMetricsRecorder,
)
from sglang.srt.runtime_context import get_context, get_disagg, get_parallel
from sglang.srt.sampling.penalties import PenaltyStatusPool
from sglang.srt.session.session_controller import SessionController

from sglang_omni.comm import KVPageTransfer
from sglang_omni.scheduling import pd_utils
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_scheduler import (
    OmniDecodeScheduler,
    OmniPrefillScheduler,
    validate_pd_runtime,
)
from sglang_omni.scheduling.pd_state import (
    PDInvariantChecker,
    PDLoadInquirer,
    PDMetricsReporter,
)
from sglang_omni.scheduling.pd_utils import (
    DecodeKVReceiver,
    ReservedKV,
    SGLangKVLease,
    req_from_continuation,
    serialize_kv_allocator,
)
from sglang_omni.scheduling.sglang_backend.cache import create_tree_cache
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from tests.unit_test.pipeline.helpers import make_stage
from tests.unit_test.pipeline.test_pd_utils import (
    KVAllocator,
    ReqPool,
    make_allocation,
    make_continuation,
    make_message,
    make_receiver,
    prefill_req,
    state_builder,
)


@pytest.mark.parametrize("grouped", [False, True])
def test_real_allocator_free_cannot_restore_a_concurrent_reservation(
    monkeypatch, grouped
):
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

    allocator = TokenToKVPoolAllocator(4, torch.float32, "cpu", None, False)
    holder = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    serialize_kv_allocator(allocator)
    assert type(holder.token_to_kv_pool_allocator) is TokenToKVPoolAllocator
    slots = allocator.alloc(1)
    if grouped:
        allocator.free_group_begin()
        allocator.free(slots)
    free_read, resume_free = threading.Event(), threading.Event()
    alloc_started, alloc_done = threading.Event(), threading.Event()
    cat = torch.cat

    def paused_cat(values, *args, **kwargs):
        if len(values) == 2:
            free_read.set()
            assert resume_free.wait(5)
        return cat(values, *args, **kwargs)

    def reserve():
        alloc_started.set()
        result = allocator.alloc(1)
        alloc_done.set()
        return result

    monkeypatch.setattr(torch, "cat", paused_cat)
    with ThreadPoolExecutor(2) as threads:
        free = threads.submit(
            holder.token_to_kv_pool_allocator.free_group_end
            if grouped
            else lambda: holder.token_to_kv_pool_allocator.free_segment(
                slots, start_pos=0
            )
        )
        try:
            assert free_read.wait(5)
            allocation = threads.submit(reserve)
            assert alloc_started.wait(5)
            assert not alloc_done.wait(0.05)
        finally:
            resume_free.set()
        free.result(timeout=5)
        live = allocation.result(timeout=5)
    remaining = allocator.alloc(3)
    assert set(live.tolist()).isdisjoint(remaining.tolist())
    assert sorted(live.tolist() + remaining.tolist()) == [1, 2, 3, 4]


def test_prefill_ack_releases_once_on_the_scheduler_thread(monkeypatch):
    scheduler = object.__new__(OmniPrefillScheduler)
    scheduler.pd_due_releases = queue.SimpleQueue()
    scheduler.pd_outstanding_releases = {"request-1"}
    scheduler.running_batch = SimpleNamespace(
        is_empty=lambda: True, batch_is_full=False
    )
    released = []
    scheduler.release_request_kv_cache = lambda req: released.append(
        (req, threading.get_ident())
    )
    monkeypatch.setattr(OmniScheduler, "get_next_batch_to_run", lambda self: None)
    monkeypatch.setattr(_Upstream, "is_fully_idle", lambda self, **kwargs: True)
    req = SimpleNamespace(rid="request-1")
    lease = SGLangKVLease(req, scheduler.pd_due_releases)
    assert scheduler.is_fully_idle() is False
    assert scheduler.is_fully_idle(for_health_check=True) is True
    with ThreadPoolExecutor(2) as threads:
        list(threads.map(lambda _: lease.release(), range(4)))
    assert released == []
    assert scheduler.is_fully_idle() is False
    scheduler.get_next_batch_to_run()
    scheduler.get_next_batch_to_run()
    assert released == [(req, threading.get_ident())]
    assert scheduler.is_fully_idle() is True


def upstream_idle_by_waiting_queue(self, for_health_check=False, ignore_waiting=False):
    return ignore_waiting or len(self.waiting_queue) == 0


def test_paused_flush_skips_the_waiting_queue_on_prefill_only(monkeypatch):
    monkeypatch.setattr(_Upstream, "is_fully_idle", upstream_idle_by_waiting_queue)
    prefill = object.__new__(OmniPrefillScheduler)
    prefill.pd_outstanding_releases = set()
    prefill.waiting_queue = [SimpleNamespace(rid="retracted")]
    decode = object.__new__(OmniDecodeScheduler)
    decode.pd_lifecycle_lock = threading.RLock()
    decode.pd_deferred_admission = None
    decode.pd_admissions = queue.SimpleQueue()
    decode.pd_receiver = SimpleNamespace(owned_allocations=lambda: ())
    decode.pd_outstanding_releases = set()
    decode.waiting_queue = [SimpleNamespace(rid="admitted")]

    assert prefill.is_fully_idle(ignore_waiting=True) is True
    assert decode.is_fully_idle(ignore_waiting=True) is False
    decode.waiting_queue = []
    assert decode.is_fully_idle(ignore_waiting=True) is True


def prefill_scheduler_for_handoff(*, request_finished_callback=None):
    scheduler = object.__new__(OmniPrefillScheduler)
    scheduler.pd_state_builder = state_builder
    scheduler.pd_pool_id = "prefill:kv"
    scheduler.pd_partner_stage = "decode"
    scheduler.pd_due_releases = queue.SimpleQueue()
    scheduler.pd_outstanding_releases = set()
    scheduler.pd_lifecycle_lock = threading.RLock()
    scheduler.disagg_prefill_transfer_queue = IndexedReqList(ListKind.PD_SENDING)
    scheduler.req_to_token_pool = ReqPool()
    scheduler.outbox = queue.Queue()
    scheduler.is_entry_rank = True
    scheduler.request_finished_callback = request_finished_callback
    scheduler.abort_callback = Mock()
    scheduler.request_admission_lock = threading.RLock()
    scheduler.completed_request_ids = {}
    scheduler.pending_stream_ingress = {}
    scheduler.first_emit_done = set()
    scheduler.prefill_start_done = set()
    scheduler.prefill_end_done = set()
    scheduler.aborted_request_ids = set()
    scheduler.release_request_kv_cache = Mock()
    return scheduler


def prefill_handoff_batch(scheduler):
    req = prefill_req()
    scheduler.req_to_token_pool.alloc([req])
    scheduler.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :3] = torch.tensor(
        [1, 2, 3]
    )
    batch = SimpleNamespace(reqs=[req], batch_is_full=True, chunked_req=None)
    scheduler.first_emit_done.add(req.rid)
    scheduler.prefill_start_done.add(req.rid)
    scheduler.prefill_end_done.add(req.rid)
    return req, batch


def test_prefill_handoff_runs_terminal_cleanup_and_closes_bookkeeping(
    monkeypatch,
) -> None:
    model_path_end = Mock()
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler._emit_model_path_end",
        model_path_end,
    )
    monkeypatch.setattr(_Upstream, "is_fully_idle", lambda self, **kwargs: True)
    finished_callback = Mock()
    scheduler = prefill_scheduler_for_handoff(
        request_finished_callback=finished_callback
    )
    req, batch = prefill_handoff_batch(scheduler)

    with envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True):
        scheduler.handoff_prefilled_requests(batch, {id(req)})

    assert req.owner_list is scheduler.disagg_prefill_transfer_queue
    assert req.send_via == "omni_pd_transfer"

    message = scheduler.outbox.get_nowait()
    assert message.type == "kv_transfer"
    assert message.request_id == req.rid
    assert message.data.source_page_indices == (1, 2, 3)
    finished_callback.assert_called_once_with(req.rid)
    model_path_end.assert_called_once_with(req.rid, status="success")
    scheduler.release_request_kv_cache.assert_not_called()
    assert scheduler.is_fully_idle() is False
    assert req.omni_data is None
    assert req.rid in scheduler.completed_request_ids
    assert req.rid not in scheduler.first_emit_done
    assert req.rid not in scheduler.prefill_start_done
    assert req.rid not in scheduler.prefill_end_done
    assert batch.reqs == []
    assert not batch.batch_is_full


def test_prefill_handoff_cleanup_failure_emits_error_and_releases_kv(
    monkeypatch,
) -> None:
    model_path_end = Mock()
    monkeypatch.setattr(
        "sglang_omni.scheduling.omni_scheduler._emit_model_path_end",
        model_path_end,
    )
    monkeypatch.setattr(_Upstream, "is_fully_idle", lambda self, **kwargs: True)
    cleanup_error = RuntimeError("terminal cleanup failed")
    finished_callback = Mock(side_effect=cleanup_error)
    scheduler = prefill_scheduler_for_handoff(
        request_finished_callback=finished_callback
    )
    req, batch = prefill_handoff_batch(scheduler)

    scheduler.handoff_prefilled_requests(batch, {id(req)})

    message = scheduler.outbox.get_nowait()
    assert message.type == "error"
    assert message.request_id == req.rid
    assert message.data is cleanup_error
    assert scheduler.outbox.empty()
    finished_callback.assert_called_once_with(req.rid)
    model_path_end.assert_called_once_with(req.rid, status="error")
    scheduler.release_request_kv_cache.assert_called_once_with(req)
    assert scheduler.is_fully_idle() is True
    assert req.omni_data is None
    assert req.rid in scheduler.completed_request_ids
    assert req.rid not in scheduler.first_emit_done
    assert req.rid not in scheduler.prefill_start_done
    assert req.rid not in scheduler.prefill_end_done
    assert batch.reqs == []
    assert not batch.batch_is_full


def test_a_failing_release_does_not_strand_the_rest_of_the_queue(monkeypatch):
    scheduler = decode_scheduler()
    released = []
    monkeypatch.setattr(_Upstream, "is_fully_idle", lambda self, **kwargs: True)

    def release(req):
        if req.rid == "bad":
            raise RuntimeError("release failed")
        released.append(req.rid)

    scheduler.release_request_kv_cache = release
    for rid in ("first", "bad", "last"):
        scheduler.pd_outstanding_releases.add(rid)
        scheduler.pd_due_releases.put(SimpleNamespace(rid=rid))
    scheduler.drain_due_releases()
    assert released == ["first", "last"]
    assert scheduler.pd_due_releases.empty()
    assert scheduler.is_fully_idle() is False


@pytest.mark.parametrize("finish", ["commit", "abort"])
def test_receiver_close_retains_copy_pages_until_comm_finishes(finish):
    receiver = make_receiver()
    message = make_message()
    destination = receiver.reserve(message)
    receiver.close()
    assert receiver.allocator.freed == []
    with pytest.raises(RuntimeError, match="closed"):
        receiver.reserve(message)
    if finish == "commit":
        with pytest.raises(RuntimeError, match="live reservation"):
            receiver.commit(message, destination)
    receiver.abort(message, destination, RuntimeError("closed"))
    receiver.abort(message, destination, RuntimeError("late abort"))
    assert len(receiver.allocator.freed) == 1
    assert receiver.admissions.empty()


def test_receiver_rejects_mismatched_pages_and_bounds_finished_ids(monkeypatch):
    receiver = make_receiver()
    message = make_message()
    with pytest.raises(ValueError, match="per prompt token"):
        receiver.reserve(replace(message, source_page_indices=(1,)))
    assert receiver.allocator.next_slot == 7
    monkeypatch.setattr(pd_utils, "_TRANSFER_TOMBSTONE_LIMIT", 2)
    for index in range(3):
        continuation = replace(make_continuation(), transfer_id=f"transfer-{index}")
        message = replace(
            message,
            transfer_id=continuation.transfer_id,
            metadata={"decode_continuation": continuation.encode()},
        )
        receiver.commit(message, receiver.reserve(message))
    with pytest.raises(RuntimeError, match="duplicate"):
        receiver.reserve(message)
    assert len(receiver.transfer_tombstones) == 2
    assert "transfer-0" not in receiver.transfer_tombstones
    assert receiver.allocator.freed == []


def decode_scheduler():
    scheduler = object.__new__(OmniDecodeScheduler)
    scheduler.pd_admissions = queue.SimpleQueue()
    scheduler.pd_due_releases = queue.SimpleQueue()
    scheduler.pd_outstanding_releases = set()
    scheduler.pd_deferred_admission = None
    scheduler.pd_lifecycle_lock = threading.RLock()
    scheduler.pd_state_restorer = lambda *args: None
    scheduler.aborted_request_ids = set()
    scheduler.req_to_token_pool = ReqPool()
    scheduler.token_to_kv_pool_allocator = KVAllocator()
    scheduler.waiting_queue = IndexedReqList(ListKind.WAITING)
    scheduler.invariant_checker = Mock()
    scheduler.outbox = queue.Queue()
    scheduler.decode_offload_manager = None
    scheduler.pd_receiver = DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=scheduler.token_to_kv_pool_allocator,
        admissions=scheduler.pd_admissions,
        resume_schema="test-v1",
        lifecycle_lock=scheduler.pd_lifecycle_lock,
    )
    return scheduler


def test_decode_kv_remains_live_across_ownership_transitions(monkeypatch):
    scheduler = decode_scheduler()
    scheduler.req_to_token_pool.capacity = 0
    released = []
    scheduler.release_request_kv_cache = lambda req: released.append(
        (req.rid, threading.get_ident())
    )
    monkeypatch.setattr(
        _Upstream,
        "is_fully_idle",
        lambda self, **kwargs: not self.waiting_queue,
    )

    monkeypatch.setattr(
        _Upstream,
        "get_next_disagg_decode_batch_to_run",
        lambda self, running_batch: SimpleNamespace(
            running_batch=None, batch_to_run=None
        ),
        raising=False,
    )

    assert scheduler.is_fully_idle() is True
    message = make_message()
    destination = scheduler.pd_receiver.reserve(message)
    assert scheduler.is_fully_idle() is False
    assert scheduler.is_fully_idle(for_health_check=True) is True

    scheduler.pd_receiver.commit(message, destination)
    assert scheduler.is_fully_idle() is False
    scheduler.drain_decode_admissions()
    assert scheduler.outbox.empty()
    assert scheduler.is_fully_idle() is False

    scheduler.req_to_token_pool.capacity = 4
    scheduler.drain_decode_admissions()
    assert [req.rid for req in scheduler.waiting_queue] == ["request-1"]
    assert scheduler.outbox.get_nowait().type == "admitted"
    assert scheduler.is_fully_idle() is False

    scheduler.session_bridge = None
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.enable_overlap = False
    scheduler.is_entry_rank = True
    scheduler.inbox = queue.Queue()
    with ThreadPoolExecutor(1) as threads:
        threads.submit(scheduler.abort, "request-1").result(timeout=5)
    assert released == []
    assert scheduler.is_fully_idle() is False
    # The scheduler has already consumed this iteration's inbox. Its release
    # drain must not race ahead of accepting the cancellation next iteration.
    scheduler.drain_due_releases()
    assert released == []
    assert [request.rid for request in scheduler.waiting_queue] == ["request-1"]
    assert scheduler.pd_due_releases.empty()
    message = scheduler.inbox.get_nowait()
    assert (message.type, message.request_id) == ("abort", "request-1")


def test_decode_flush_drains_releases_and_gates_new_reservations(monkeypatch):
    scheduler = decode_scheduler()
    req = SimpleNamespace(rid="request-1")
    scheduler.waiting_queue = [req]
    scheduler.release_request_kv_cache = Mock()
    message = make_message()

    def upstream_flush(self):
        self.release_request_kv_cache.assert_called_once_with(req)
        with pytest.raises(RuntimeError, match="not accepting reservations"):
            self.pd_receiver.reserve(message)
        return True

    monkeypatch.setattr(_Upstream, "flush_cache", upstream_flush)
    scheduler.defer_pd_kv_release(req)
    assert scheduler.flush_cache() is True

    destination = scheduler.pd_receiver.reserve(message)
    scheduler.pd_receiver.abort(message, destination, RuntimeError("test cleanup"))


def test_weight_update_waits_for_pd_kv_and_gates_reservations(monkeypatch):
    scheduler = decode_scheduler()

    def run_update(self, payload, update_fn, result_data, **kwargs):
        success, message = update_fn(payload)
        return {"success": success, "message": message}

    monkeypatch.setattr(OmniScheduler, "run_weight_update_with_lifecycle", run_update)
    update = Mock(return_value=(True, "updated"))
    held_message = make_message()
    held_destination = scheduler.pd_receiver.reserve(held_message)
    result = scheduler.run_weight_update_with_lifecycle({}, update, {})
    assert result == {"success": False, "message": "PD-owned KV is still in flight"}
    update.assert_not_called()

    scheduler.pd_receiver.abort(
        held_message, held_destination, RuntimeError("test cleanup")
    )
    message = make_message(transfer_id="transfer-2")

    def gated_update(payload):
        with pytest.raises(RuntimeError, match="not accepting reservations"):
            scheduler.pd_receiver.reserve(message)
        return True, "updated"

    result = scheduler.run_weight_update_with_lifecycle({}, gated_update, {})
    assert result == {"success": True, "message": "updated"}
    destination = scheduler.pd_receiver.reserve(message)
    scheduler.pd_receiver.abort(message, destination, RuntimeError("test cleanup"))


def test_deferred_admission_abort_frees_committed_pages_once():
    scheduler = decode_scheduler()
    scheduler.req_to_token_pool.capacity = 0
    message = make_message()
    scheduler.pd_receiver.commit(message, scheduler.pd_receiver.reserve(message))
    scheduler.drain_decode_admissions()
    assert scheduler.pd_deferred_admission is not None
    assert scheduler.token_to_kv_pool_allocator.freed == []
    scheduler.aborted_request_ids.add("request-1")
    scheduler.drain_decode_admissions()
    scheduler.drain_decode_admissions()
    assert len(scheduler.token_to_kv_pool_allocator.freed) == 1
    assert not scheduler.waiting_queue


def test_stop_string_requires_and_uses_model_tokenizer():
    req = prefill_req()
    req.sampling_params.stop_strs = ["END"]
    req.sampling_params.stop_str_max_len = 3
    continuation = replace(
        make_continuation(),
        sampling_params=pd_utils.sampling_params_to_dict(req.sampling_params),
    )
    pool = ReqPool()
    with pytest.raises(ValueError, match="tokenizer"):
        req_from_continuation(
            continuation,
            make_allocation(),
            req_to_token_pool=pool,
            state_restorer=lambda *args: None,
        )
    assert pool.active == {}
    tokenizer = SimpleNamespace(
        eos_token_id=2, additional_stop_token_ids=[], decode=lambda ids: "END"
    )
    restored = req_from_continuation(
        continuation,
        make_allocation(),
        req_to_token_pool=pool,
        state_restorer=lambda req, *_: setattr(req, "tokenizer", tokenizer),
    )
    restored.update_finish_state()
    assert restored.finished_reason.to_json()["type"] == "stop"


def make_transfer(request_id="request-1", **updates):
    return KVPageTransfer(
        **{
            "request_id": request_id,
            "transfer_id": f"{request_id}-transfer",
            "source_pool_id": "prefill:kv",
            "target_pool_id": "decode:kv",
            "source_page_indices": (1, 2, 3),
            "to_stage": "decode",
            "lease": Mock(),
            **updates,
        }
    )


def test_discarded_stage_transfer_releases_source_lease():
    transfer = make_transfer()

    make_stage().discard_kv_transfer(transfer)

    transfer.lease.release.assert_called_once_with()

    async def wait_for_release(inactive: KVPageTransfer) -> None:
        while not inactive.lease.release.called:
            await asyncio.sleep(0.01)

    async def run() -> None:
        stage = make_stage()
        stage.running = True
        inactive = make_transfer("inactive")
        stage.scheduler.outbox.put(
            OutgoingMessage(inactive.request_id, "kv_transfer", inactive)
        )
        drain = asyncio.create_task(stage.drain_outbox())
        try:
            await asyncio.wait_for(wait_for_release(inactive), 5)
            inactive.lease.release.assert_called_once_with()
            assert not stage.receive_tasks
        finally:
            stage.running = False
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)

    asyncio.run(run())


def test_slow_ack_does_not_block_outbox_and_early_cancellation_releases():
    async def run():
        stage = make_stage()
        stage.running = True
        slow_ack, fast_sent = asyncio.Event(), asyncio.Event()

        async def send_kv_pages(*, request_id, lease, **kwargs):
            try:
                if request_id == "slow":
                    await slow_ack.wait()
                else:
                    fast_sent.set()
            finally:
                lease.release()

        stage.comm.send_kv_pages = send_kv_pages
        from sglang_omni.scheduling.message import OutgoingMessage

        transfers = [make_transfer("slow"), make_transfer("fast")]
        for transfer in transfers:
            stage.active_requests.add(transfer.request_id)
            stage.scheduler.outbox.put(
                OutgoingMessage(transfer.request_id, "kv_transfer", transfer)
            )
        drain = asyncio.create_task(stage.drain_outbox())
        try:
            await asyncio.wait_for(fast_sent.wait(), 5)
            transfers[0].lease.release.assert_not_called()
            slow_ack.set()
            await asyncio.gather(*tuple(stage.receive_tasks))
            for transfer in transfers:
                transfer.lease.release.assert_called_once()
            cancelled = make_transfer("cancelled")
            stage.launch_kv_transfer(cancelled)
            task = next(iter(stage.receive_tasks))
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            cancelled.lease.release.assert_called_once()
        finally:
            stage.running = False
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)

    asyncio.run(run())


def test_missing_binding_releases_before_comm_takes_ownership():
    async def run():
        stage = make_stage(replica_topology={"decode": ["decode@r0", "decode@r1"]})
        transfer = make_transfer()
        stage.active_requests.add(transfer.request_id)
        await stage.send_kv_transfer(transfer)
        transfer.lease.release.assert_called_once()
        assert transfer.request_id not in stage.active_requests
        assert "no replica binding" in stage.control_plane.completions[0].error

    asyncio.run(run())


def test_memory_pressure_fails_one_request_without_upstream_rebootstrap():
    scheduler = decode_scheduler()
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.5)
    scheduler.enable_skip_finishing_decode = False
    scheduler.tree_cache = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(state_slots=None),
    )
    scheduler.request_retirer = Mock()
    scheduler.enable_plain_fallback = False
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.server_args = SimpleNamespace()
    scheduler.abort = Mock()
    scheduler.emit_request_error = Mock()
    batch = SimpleNamespace(
        batch_size=lambda: 1,
        filter_batch=lambda *, reqs_to_exclude: None,
        is_empty=lambda: False,
        check_decode_mem=lambda: False,
        retract_decode=lambda request_retirer: ([SimpleNamespace(rid="full")], 0.5, []),
        prepare_for_decode=Mock(),
    )
    scheduler.update_running_batch(batch)
    scheduler.abort.assert_called_once_with("full")
    assert "cannot resume" in str(scheduler.emit_request_error.call_args.args[1])
    batch.prepare_for_decode.assert_called_once()


def test_non_pd_scheduler_does_not_need_kv_registration():
    scheduler = SimpleScheduler(lambda payload: payload)
    stage = make_stage(scheduler=scheduler)
    assert stage.scheduler is scheduler
    assert stage.comm.kv_pools == {}


def test_binding_survives_comm_handoff_admission_and_next_stage(monkeypatch):
    import sglang_omni.platforms as platforms
    from sglang_omni.comm.data_ref import TransportKind
    from sglang_omni.scheduling.message import OutgoingMessage
    from tests.unit_test.pipeline.test_kv_transfer import make_pool, start_pair

    monkeypatch.setattr(
        platforms.current_platform,
        "get_intra_node_transport",
        lambda: TransportKind.CUDA_IPC,
    )

    async def run():
        _, source, destination = await start_pair()
        scheduler = decode_scheduler()
        receiver = DecodeKVReceiver(
            pool_id="decode:kv",
            allocator=scheduler.token_to_kv_pool_allocator,
            admissions=scheduler.pd_admissions,
            resume_schema="test-v1",
        )
        receiver.allocator.next_slot = 0
        scheduler.pd_receiver = receiver
        source.register_kv_pool(make_pool("prefill:kv"))
        destination.register_kv_pool(make_pool("decode:kv"))
        destination.register_kv_receiver("decode:kv", receiver)
        topology = {"post": ["post@r0", "post@r1"]}
        prefill = make_stage(name="source", replica_topology=topology)
        prefill.comm = source
        prefill.record_replica_bindings("request-1", {"post": 1})
        dispatched = asyncio.Event()

        async def send_payload(**kwargs):
            dispatched.set()

        dispatcher = SimpleNamespace(send_payload=AsyncMock(side_effect=send_payload))
        decode = make_stage(
            name="destination",
            scheduler=scheduler,
            replica_topology=topology,
            get_next=lambda *_: "post",
            endpoints={"post@r1": "inproc://post1"},
            same_process_targets={"post@r1"},
            local_dispatcher=dispatcher,
        )
        drain = None
        try:
            continuation = make_continuation()
            transfer = make_transfer(
                transfer_id=continuation.transfer_id,
                to_stage="destination",
                metadata={"decode_continuation": continuation.encode()},
            )
            await prefill.send_kv_transfer(transfer)
            transfer.lease.release.assert_called_once()
            scheduler.drain_decode_admissions()
            scheduler.outbox.put(
                OutgoingMessage(
                    "request-1",
                    "result",
                    scheduler.waiting_queue[0].omni_data.stage_payload,
                )
            )
            decode.running = True
            drain = asyncio.create_task(decode.drain_outbox())
            await asyncio.wait_for(dispatched.wait(), 5)
            kwargs = dispatcher.send_payload.call_args.kwargs
            assert kwargs["to_stage"] == "post@r1"
            assert kwargs["replica_bindings"] == {"post": 1}
        finally:
            decode.running = False
            if drain is not None:
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)
            await source.close()
            await destination.close()

    asyncio.run(run())


def test_replicated_decode_target_uses_bound_instance_and_pool():
    async def run():
        stage = make_stage(replica_topology={"decode": ["decode@r0", "decode@r1"]})
        stage.record_replica_bindings("request-1", {"decode": 1})
        transfer = make_transfer()

        async def send(**kwargs):
            kwargs["lease"].release()

        stage.comm.send_kv_pages = AsyncMock(side_effect=send)
        await stage.send_kv_transfer(transfer)
        kwargs = stage.comm.send_kv_pages.call_args.kwargs
        assert kwargs["to_stage"] == "decode@r1"
        assert kwargs["target_pool_id"] == "decode@r1:kv"
        transfer.lease.release.assert_called_once()

    asyncio.run(run())


@pytest.mark.parametrize("scheduler_type", [OmniPrefillScheduler, OmniDecodeScheduler])
@pytest.mark.parametrize("session_enabled", [False, True])
def test_pd_allows_session_flag_without_adapter(
    scheduler_type: type[OmniPrefillScheduler] | type[OmniDecodeScheduler],
    session_enabled: bool,
) -> None:
    scheduler = object.__new__(scheduler_type)
    scheduler.session_adapter = None
    scheduler.req_to_token_pool = ReqPool()
    scheduler.token_to_kv_pool_allocator = KVAllocator()
    scheduler.server_args = SimpleNamespace(
        enable_streaming_session=session_enabled, disable_radix_cache=True
    )
    scheduler.tp_size = 1
    scheduler.page_size = 1
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    with get_context().override_server_args():
        validate_pd_runtime(scheduler)


@pytest.mark.parametrize("scheduler_type", [OmniPrefillScheduler, OmniDecodeScheduler])
def test_pd_rejects_retained_session_adapter(
    scheduler_type: type[OmniPrefillScheduler] | type[OmniDecodeScheduler],
) -> None:
    scheduler = object.__new__(scheduler_type)
    scheduler.session_adapter = Mock()
    with pytest.raises(NotImplementedError, match="retained-state ownership"):
        validate_pd_runtime(scheduler)


def test_pd_rebuilt_request_releases_real_kv_once() -> None:
    with get_context().override_server_args(
        speculative_algorithm=None, strip_thinking_cache=False, disable_radix_cache=True
    ):
        allocator = TokenToKVPoolAllocator(8, torch.float32, "cpu", None, False)
        request_pool = ReqToTokenPool(2, 32, "cpu", enable_memory_saver=False)
        cache = create_tree_cache(request_pool, allocator, 1, None)
        scheduler = decode_scheduler()
        scheduler.tree_cache = cache
        scheduler.request_retirer = RequestRetirer(
            tree_cache=cache,
            beam_coordinator=Mock(),
            hisparse_coordinator=None,
            uocr_runtime=None,
            penalty_status_pool=PenaltyStatusPool(vocab_size=128, device="cpu"),
            send_abort_output=Mock(),
            stream_output=Mock(),
        )
        initial_free = allocator.available_size()
        slots = allocator.alloc(3)
        request = req_from_continuation(
            make_continuation(),
            ReservedKV(slots=slots, page_indices=tuple(slots.tolist()), seq_len=3),
            req_to_token_pool=request_pool,
            state_restorer=lambda request, payload, resume: None,
        )
        assert allocator.available_size() == initial_free - 3
        assert request_pool.available_size() == 1
        scheduler.defer_pd_kv_release(request)
        assert scheduler.pd_holds_kv()
        scheduler.drain_due_releases()
        assert not scheduler.pd_holds_kv()
        assert allocator.available_size() == initial_free
        assert request_pool.available_size() == 2
        assert not request.kv.holds_kv
        scheduler.release_request_kv_cache(request)
        assert allocator.available_size() == initial_free
        assert request_pool.available_size() == 2


@pytest.fixture
def real_pd_scheduler():
    with (
        get_context().override_server_args(disable_radix_cache=True),
        get_parallel().override(dp_rank=None),
        envs.SGLANG_ENABLE_RUNNING_REQS_CHECK.override(True),
        envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.override(True),
        envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
        envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(True),
    ):
        scheduler = decode_scheduler()
        scheduler.req_to_token_pool = ReqToTokenPool(
            2, 32, "cpu", enable_memory_saver=False
        )
        scheduler.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            16, torch.float32, "cpu", None, False
        )
        serialize_kv_allocator(
            scheduler.token_to_kv_pool_allocator, lock=scheduler.pd_lifecycle_lock
        )
        scheduler.tree_cache = create_tree_cache(
            scheduler.req_to_token_pool, scheduler.token_to_kv_pool_allocator, 1, None
        )
        scheduler.running_batch = ScheduleBatch(reqs=[])
        scheduler.running_reqs = KeyedReqList(ListKind.DECODING)
        scheduler.request_lists = lambda: {
            ListKind.WAITING: scheduler.waiting_queue,
            ListKind.DECODING: scheduler.running_reqs,
        }
        scheduler.pd_receiver = DecodeKVReceiver(
            pool_id="decode:kv",
            allocator=scheduler.token_to_kv_pool_allocator,
            admissions=scheduler.pd_admissions,
            resume_schema="test-v1",
            lifecycle_lock=scheduler.pd_lifecycle_lock,
        )
        scheduler.pool_stats_observer = SchedulerPoolStatsObserver(
            tree_cache=scheduler.tree_cache,
            storage=kv_storage_of(
                scheduler.req_to_token_pool, scheduler.token_to_kv_pool_allocator
            ),
            token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
            session_controller=SessionController(scheduler.tree_cache),
            hisparse_coordinator=None,
            enable_hisparse=False,
        )
        checker = SchedulerInvariantChecker(
            disaggregation_mode=DisaggregationMode.DECODE,
            page_size=1,
            tree_cache=scheduler.tree_cache,
            token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
            req_to_token_pool=scheduler.req_to_token_pool,
            pool_stats_observer=scheduler.pool_stats_observer,
            scheduler_stage_metrics=SchedulerStageMetricsRecorder(enabled=False),
            get_request_lists=scheduler.request_lists,
            get_running_batches=lambda: (scheduler.running_batch,),
        )
        scheduler.invariant_checker = PDInvariantChecker(checker, scheduler)
        scheduler.request_retirer = RequestRetirer(
            tree_cache=scheduler.tree_cache,
            beam_coordinator=Mock(),
            hisparse_coordinator=None,
            uocr_runtime=None,
            penalty_status_pool=PenaltyStatusPool(vocab_size=128, device="cpu"),
            send_abort_output=Mock(),
            stream_output=Mock(),
        )
        yield scheduler


def test_real_pd_slots_remain_accounted_until_request_release(real_pd_scheduler):
    """Strict checks must see received KV even before a Req row exists."""
    scheduler = real_pd_scheduler
    receiver = scheduler.pd_receiver
    message = make_message()
    destination = receiver.reserve(message)
    scheduler.invariant_checker.run(CheckPhase.IDLE)
    assert scheduler.token_to_kv_pool_allocator.available_size() == 13
    receiver.commit(message, destination)
    scheduler.invariant_checker.run(CheckPhase.IDLE)
    assert len(receiver.owned_allocations()) == 1
    scheduler.drain_decode_admissions()
    assert receiver.owned_allocations() == ()
    request = next(iter(scheduler.waiting_queue))
    scheduler.invariant_checker.run(CheckPhase.BUSY, scheduler.running_batch)
    scheduler.invariant_checker.cancel_output(request)
    request.finish_outside_step(FINISH_ABORT())
    scheduler.request_retirer.release(request, checkpoint=False)
    scheduler.invariant_checker.run(CheckPhase.IDLE)
    assert scheduler.token_to_kv_pool_allocator.available_size() == 16
    assert scheduler.req_to_token_pool.available_size() == 2
    assert not scheduler.invariant_checker.unended


@pytest.mark.parametrize("corruption", ["freed", "aliased"])
def test_pd_checker_rejects_corrupt_unbound_slots(real_pd_scheduler, corruption):
    scheduler = real_pd_scheduler
    receiver = scheduler.pd_receiver
    message = make_message()
    receiver.reserve(message)
    allocation = next(iter(receiver.reservations.values())).allocation
    if corruption == "freed":
        scheduler.token_to_kv_pool_allocator.free(allocation.slots)
    else:
        allocation.slots[1] = allocation.slots[0]
    with pytest.raises(StateCheckError, match="PD transfer slots alias"):
        scheduler.invariant_checker.check_pd_slots()


def test_pd_load_counts_reserved_committed_and_bound_tokens_once(real_pd_scheduler):
    """A committed waiting request already occupies KV and must not be counted twice."""
    scheduler = real_pd_scheduler
    scheduler.max_total_num_tokens = 16
    reporter = object.__new__(PDMetricsReporter)
    reporter.scheduler = scheduler
    reporter.stats = SchedulerStats()
    scheduler.metrics_reporter = reporter
    common = SchedulerLoadInquirer(
        disaggregation_mode=DisaggregationMode.DECODE,
        server_args=None,
        max_total_num_tokens=16,
        max_running_requests=2,
        pool_stats_observer=scheduler.pool_stats_observer,
        tp_worker=SimpleNamespace(),
        token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        get_running_batch=lambda: scheduler.running_batch,
        get_waiting_queue=lambda: scheduler.waiting_queue,
        waiting_queue_prefix_matched=lambda: True,
        get_recent_cache_hit_rate=lambda: 0.0,
        get_stats=lambda: reporter.stats,
        get_chunked_req=lambda: None,
        get_request_lists=scheduler.request_lists,
        get_disagg_decode_prealloc_queue=lambda: None,
        get_spec_total_num_accept_tokens=lambda: 0,
        get_spec_total_num_forward_ct=lambda: 0,
        get_total_prefill_uncached_tokens=lambda: 0,
        get_total_prefill_busy_us=lambda: 0,
        get_decode_moment_totals=lambda: [0.0] * 6,
    )
    loads = PDLoadInquirer(common, scheduler)
    message = make_message()
    receiver = scheduler.pd_receiver
    destination = receiver.reserve(message)
    reserved = loads.get_loads()
    assert (
        reserved.num_waiting_reqs,
        reserved.num_total_tokens,
        reserved.num_active_tokens,
    ) == (1, 3, 0)
    assert reserved.disaggregation.decode_transfer_queue_reqs == 1
    receiver.commit(message, destination)
    committed = loads.get_loads()
    assert (
        committed.num_waiting_reqs,
        committed.num_total_tokens,
        committed.num_active_tokens,
    ) == (1, 3, 3)
    assert committed.num_prealloc_ready_tokens == 3
    assert committed.queues.prealloc_ready == 1
    scheduler.drain_decode_admissions()
    bound = loads.get_loads()
    assert (
        bound.num_waiting_reqs,
        bound.num_total_tokens,
        bound.num_active_tokens,
    ) == (1, 3, 3)
    assert bound.num_waiting_uncached_tokens == bound.num_prealloc_ready_tokens == 0
    reporter._update_decode_queue_stats(
        True
    )  # noqa: leading-underscore  # exercise the reporter contract
    queued = (
        reporter._build_queued_request_metrics()
    )  # noqa: leading-underscore  # exercise the reporter contract
    assert queued.num_decode_requests == 1
    assert queued.sum_decode_kv_tokens == 3
    assert (
        "#transfer-req: 0" in reporter._decode_queue_log_suffix()
    )  # noqa: leading-underscore  # exercise the reporter contract


def test_receiver_abort_keeps_ownership_and_free_atomic(real_pd_scheduler, monkeypatch):
    """A checker must not observe popped ownership before the slots are freed."""
    scheduler = real_pd_scheduler
    receiver = scheduler.pd_receiver
    message = make_message()
    destination = receiver.reserve(message)
    free_started, finish_free = threading.Event(), threading.Event()
    original_free = receiver.allocator.free

    def paused_free(slots):
        free_started.set()
        assert finish_free.wait(5)
        original_free(slots)

    monkeypatch.setattr(receiver.allocator, "free", paused_free)
    with ThreadPoolExecutor(1) as threads:
        abort = threads.submit(
            receiver.abort, message, destination, RuntimeError("cancelled")
        )
        try:
            assert free_started.wait(5)
            acquired = scheduler.pd_lifecycle_lock.acquire(blocking=False)
            if acquired:
                scheduler.pd_lifecycle_lock.release()
            else:
                pass
            assert not acquired
        finally:
            finish_free.set()
        abort.result(timeout=5)
    scheduler.invariant_checker.run(CheckPhase.IDLE)
    assert receiver.owned_allocations() == ()
    assert receiver.allocator.available_size() == 16


def test_live_committed_transfer_survives_tombstone_eviction(
    real_pd_scheduler, monkeypatch
):
    """A live transfer must reject duplicates after its bounded tombstone expires."""
    receiver = real_pd_scheduler.pd_receiver
    monkeypatch.setattr(pd_utils, "_TRANSFER_TOMBSTONE_LIMIT", 2)
    first = make_message()
    receiver.commit(first, receiver.reserve(first))
    for index in range(3):
        later = make_message(transfer_id=f"aborted-{index}")
        receiver.abort(later, receiver.reserve(later), RuntimeError("cancelled"))
    assert first.transfer_id not in receiver.transfer_tombstones
    before = receiver.allocator.available_size()
    with pytest.raises(RuntimeError, match="duplicate KV transfer"):
        receiver.reserve(first)
    assert receiver.allocator.available_size() == before
    assert len(receiver.owned_allocations()) == 1
    real_pd_scheduler.invariant_checker.run(CheckPhase.IDLE)


def test_idle_pd_does_not_run_disabled_page_diagnostics(real_pd_scheduler, monkeypatch):
    """An idle transfer must not trigger tensor alias checks in ordinary serving."""
    scheduler = real_pd_scheduler
    scheduler.pd_receiver.reserve(make_message())
    monkeypatch.setattr(
        scheduler.invariant_checker,
        "check_pd_slots",
        Mock(side_effect=AssertionError("unexpected page scan")),
    )
    with envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(False):
        scheduler.invariant_checker.run(CheckPhase.IDLE)


@pytest.mark.parametrize("phase", ["receiving", "committed", "deferred"])
def test_admin_abort_covers_unbound_pd_and_never_reuses_copy_destination(
    real_pd_scheduler, phase
):
    """An admin abort must not miss received KV outside the normal waiting list."""
    scheduler = real_pd_scheduler
    scheduler.session_bridge = None
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.enable_overlap = False
    scheduler.is_entry_rank = True
    scheduler.admin_lock = threading.Lock()
    scheduler.request_admission_lock = threading.RLock()
    scheduler.resolve_pending_async = lambda: None
    scheduler.cur_batch = scheduler.last_batch = None
    scheduler.async_pending = None
    scheduler.result_queue = deque()
    scheduler.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
    scheduler.decode_batch_states = {}
    scheduler.pending_request_builds = {}
    scheduler.pending_request_admissions = {}
    scheduler.backlogged_request_build_payloads = deque()
    scheduler.early_request_builds = False
    scheduler.deferred_request_payloads = {}
    scheduler.pending_stream_ingress = {}
    scheduler.dirty_deferred_request_ids = set()
    scheduler.first_emit_done = set()
    scheduler.prefill_start_done = set()
    scheduler.prefill_end_done = set()
    scheduler.aborted_request_id_order = deque()
    scheduler.abort_callback = None
    scheduler.inbox = queue.Queue()
    receiver = scheduler.pd_receiver
    receiver.cancelled_request_ids = scheduler.aborted_request_ids
    message = make_message()
    destination = receiver.reserve(message)
    if phase != "receiving":
        receiver.commit(message, destination)
    if phase == "deferred":
        scheduler.pd_deferred_admission = scheduler.pd_admissions.get_nowait()
    result = scheduler.admin_pause_generation({"mode": "abort"})
    assert result["data"]["num_paused_requests"] == 1
    assert scheduler.outbox.get_nowait().type == "error"
    scheduler.admin_continue_generation({"torch_empty_cache": False})
    if phase == "receiving":
        assert receiver.allocator.available_size() == 13
        scheduler.invariant_checker.run(CheckPhase.IDLE)
        # Even eviction of the bounded request tombstone cannot recycle a
        # destination while the comm thread is still copying into it.
        scheduler.aborted_request_ids.clear()
        scheduler.admin_pause_generation({"mode": "abort"})
        assert scheduler.outbox.empty()
        assert receiver.allocator.available_size() == 13
        receiver.commit(message, destination)
    scheduler.drain_decode_admissions()
    assert not scheduler.waiting_queue
    assert scheduler.outbox.empty()
    assert receiver.allocator.available_size() == 16
    assert not scheduler.pd_holds_kv()
    scheduler.invariant_checker.run(CheckPhase.IDLE)


@pytest.mark.parametrize(
    "option,value,reason",
    [
        ("disaggregation_decode_host_receive_threshold", 0.5, "host KV receive"),
        ("disaggregation_enable_kv_checksum", True, "checksum metadata"),
    ],
)
def test_pd_rejects_native_transport_options(option, value, reason):
    scheduler = decode_scheduler()
    scheduler.session_adapter = None
    scheduler.tp_size = scheduler.page_size = 1
    scheduler.server_args = SimpleNamespace(disable_radix_cache=True)
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    with get_context().override_server_args(), get_disagg().override(**{option: value}):
        with pytest.raises(NotImplementedError, match=reason):
            validate_pd_runtime(scheduler)


@pytest.mark.parametrize("scheduler_type", [OmniPrefillScheduler, OmniDecodeScheduler])
@pytest.mark.parametrize("pool_kind", ["linear", "pure_swa", "hybrid_swa"])
def test_pd_rejects_state_not_carried_by_page_transfer(
    scheduler_type: type[OmniPrefillScheduler] | type[OmniDecodeScheduler],
    pool_kind: str,
) -> None:
    scheduler = object.__new__(scheduler_type)
    scheduler.session_adapter = None
    scheduler.req_to_token_pool = ReqPool()
    scheduler.token_to_kv_pool_allocator = KVAllocator()
    if pool_kind == "linear":
        scheduler.req_to_token_pool = object.__new__(HybridReqToTokenPool)
        message = "linear-attention state"
    else:
        allocator_type = (
            PureSWATokenToKVPoolAllocator
            if pool_kind == "pure_swa"
            else SWATokenToKVPoolAllocator
        )
        scheduler.token_to_kv_pool_allocator = object.__new__(allocator_type)
        message = "sliding-window KV ownership"
    with pytest.raises(NotImplementedError, match=message):
        validate_pd_runtime(scheduler)
