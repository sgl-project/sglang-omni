# SPDX-License-Identifier: Apache-2.0
"""PD ownership and routing regressions, without model weights or GPU kernels."""

import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from sglang.srt.managers.scheduler import Scheduler as _Upstream

from sglang_omni.comm import KVPageTransfer
from sglang_omni.scheduling import pd_utils
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_scheduler import (
    OmniDecodeScheduler,
    OmniPrefillScheduler,
)
from sglang_omni.scheduling.pd_utils import (
    DecodeAdmission,
    DecodeKVReceiver,
    SGLangKVLease,
    req_from_continuation,
    serialize_kv_allocator,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from tests.unit_test.pipeline.helpers import make_stage
from tests.unit_test.pipeline.test_pd_utils import (
    _allocation,
    _continuation,
    _KVAllocator,
    _message,
    _prefill_req,
    _receiver,
    _ReqPool,
    _state_builder,
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
    scheduler._pd_due_releases = queue.SimpleQueue()
    scheduler._pd_outstanding_releases = {"request-1"}
    scheduler.running_batch = SimpleNamespace(
        is_empty=lambda: True, batch_is_full=False
    )
    released = []
    scheduler._release_request_kv_cache = lambda req: released.append(
        (req, threading.get_ident())
    )
    monkeypatch.setattr(OmniScheduler, "get_next_batch_to_run", lambda self: None)
    monkeypatch.setattr(_Upstream, "is_fully_idle", lambda self, **kwargs: True)
    req = SimpleNamespace(rid="request-1")
    lease = SGLangKVLease(req, scheduler._pd_due_releases)
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


def _prefill_scheduler_for_handoff(*, request_finished_callback=None):
    scheduler = object.__new__(OmniPrefillScheduler)
    scheduler._pd_state_builder = _state_builder
    scheduler._pd_pool_id = "prefill:kv"
    scheduler._pd_partner_stage = "decode"
    scheduler._pd_due_releases = queue.SimpleQueue()
    scheduler._pd_outstanding_releases = set()
    scheduler.req_to_token_pool = _ReqPool()
    scheduler.outbox = queue.Queue()
    scheduler.is_entry_rank = True
    scheduler._request_finished_callback = request_finished_callback
    scheduler._abort_callback = Mock()
    scheduler._request_admission_lock = threading.RLock()
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {}
    scheduler._first_emit_done = set()
    scheduler._prefill_start_done = set()
    scheduler._prefill_end_done = set()
    scheduler._aborted_request_ids = set()
    scheduler._release_request_kv_cache = Mock()
    return scheduler


def _prefill_handoff_batch(scheduler):
    req = _prefill_req()
    scheduler.req_to_token_pool.alloc([req])
    scheduler.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :3] = torch.tensor(
        [1, 2, 3]
    )
    batch = SimpleNamespace(reqs=[req], batch_is_full=True)
    scheduler._first_emit_done.add(req.rid)
    scheduler._prefill_start_done.add(req.rid)
    scheduler._prefill_end_done.add(req.rid)
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
    scheduler = _prefill_scheduler_for_handoff(
        request_finished_callback=finished_callback
    )
    req, batch = _prefill_handoff_batch(scheduler)

    scheduler._handoff_prefilled_requests(batch, {id(req)})

    message = scheduler.outbox.get_nowait()
    assert message.type == "kv_transfer"
    assert message.request_id == req.rid
    assert message.data.source_page_indices == (1, 2, 3)
    finished_callback.assert_called_once_with(req.rid)
    model_path_end.assert_called_once_with(req.rid, status="success")
    scheduler._release_request_kv_cache.assert_not_called()
    assert scheduler.is_fully_idle() is False
    assert req._omni_data is None
    assert req.rid in scheduler._completed_request_ids
    assert req.rid not in scheduler._first_emit_done
    assert req.rid not in scheduler._prefill_start_done
    assert req.rid not in scheduler._prefill_end_done
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
    scheduler = _prefill_scheduler_for_handoff(
        request_finished_callback=finished_callback
    )
    req, batch = _prefill_handoff_batch(scheduler)

    scheduler._handoff_prefilled_requests(batch, {id(req)})

    message = scheduler.outbox.get_nowait()
    assert message.type == "error"
    assert message.request_id == req.rid
    assert message.data is cleanup_error
    assert scheduler.outbox.empty()
    finished_callback.assert_called_once_with(req.rid)
    model_path_end.assert_called_once_with(req.rid, status="error")
    scheduler._release_request_kv_cache.assert_called_once_with(req)
    assert scheduler.is_fully_idle() is True
    assert req._omni_data is None
    assert req.rid in scheduler._completed_request_ids
    assert req.rid not in scheduler._first_emit_done
    assert req.rid not in scheduler._prefill_start_done
    assert req.rid not in scheduler._prefill_end_done
    assert batch.reqs == []
    assert not batch.batch_is_full


def test_a_failing_release_does_not_strand_the_rest_of_the_queue(monkeypatch):
    scheduler = _decode_scheduler()
    released = []
    monkeypatch.setattr(_Upstream, "is_fully_idle", lambda self, **kwargs: True)

    def release(req):
        if req.rid == "bad":
            raise RuntimeError("release failed")
        released.append(req.rid)

    scheduler._release_request_kv_cache = release
    for rid in ("first", "bad", "last"):
        scheduler._pd_outstanding_releases.add(rid)
        scheduler._pd_due_releases.put(SimpleNamespace(rid=rid))
    scheduler._drain_due_releases()
    assert released == ["first", "last"]
    assert scheduler._pd_due_releases.empty()
    assert scheduler.is_fully_idle() is False


@pytest.mark.parametrize("finish", ["commit", "abort"])
def test_receiver_close_retains_copy_pages_until_comm_finishes(finish):
    receiver = _receiver()
    message = _message()
    destination = receiver.reserve(message)
    receiver.close()
    assert receiver._allocator.freed == []
    with pytest.raises(RuntimeError, match="closed"):
        receiver.reserve(message)
    if finish == "commit":
        with pytest.raises(RuntimeError, match="live reservation"):
            receiver.commit(message, destination)
    receiver.abort(message, destination, RuntimeError("closed"))
    receiver.abort(message, destination, RuntimeError("late abort"))
    assert len(receiver._allocator.freed) == 1
    assert receiver._admissions.empty()


def test_receiver_rejects_mismatched_pages_and_bounds_finished_ids(monkeypatch):
    receiver = _receiver()
    message = _message()
    with pytest.raises(ValueError, match="per prompt token"):
        receiver.reserve(replace(message, source_page_indices=(1,)))
    assert receiver._allocator.next_slot == 7
    monkeypatch.setattr(pd_utils, "_TRANSFER_TOMBSTONE_LIMIT", 2)
    for index in range(3):
        continuation = replace(_continuation(), transfer_id=f"transfer-{index}")
        message = replace(
            message,
            transfer_id=continuation.transfer_id,
            metadata={"decode_continuation": continuation.encode()},
        )
        receiver.commit(message, receiver.reserve(message))
    with pytest.raises(RuntimeError, match="duplicate"):
        receiver.reserve(message)
    assert len(receiver._transfer_tombstones) == 2
    assert "transfer-0" not in receiver._transfer_tombstones
    assert receiver._allocator.freed == []


def _decode_scheduler():
    scheduler = object.__new__(OmniDecodeScheduler)
    scheduler._pd_admissions = queue.SimpleQueue()
    scheduler._pd_due_releases = queue.SimpleQueue()
    scheduler._pd_outstanding_releases = set()
    scheduler._pd_deferred_admission = None
    scheduler._pd_lifecycle_lock = threading.RLock()
    scheduler._pd_state_restorer = lambda *args: None
    scheduler._aborted_request_ids = set()
    scheduler.req_to_token_pool = _ReqPool()
    scheduler.token_to_kv_pool_allocator = _KVAllocator()
    scheduler.waiting_queue = []
    scheduler.outbox = queue.Queue()
    scheduler._pd_receiver = DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=scheduler.token_to_kv_pool_allocator,
        admissions=scheduler._pd_admissions,
        resume_schema="test-v1",
        lifecycle_lock=scheduler._pd_lifecycle_lock,
    )
    return scheduler


def test_decode_kv_remains_live_across_ownership_transitions(monkeypatch):
    scheduler = _decode_scheduler()
    scheduler.req_to_token_pool.capacity = 0
    released = []
    scheduler._release_request_kv_cache = lambda req: released.append(
        (req.rid, threading.get_ident())
    )
    monkeypatch.setattr(
        _Upstream,
        "is_fully_idle",
        lambda self, **kwargs: not self.waiting_queue,
    )

    def abort(self, request_id, **kwargs):
        self._aborted_request_ids.add(request_id)
        self.waiting_queue = [
            req for req in self.waiting_queue if req.rid != request_id
        ]

    monkeypatch.setattr(OmniScheduler, "abort", abort)
    monkeypatch.setattr(
        _Upstream,
        "get_next_disagg_decode_batch_to_run",
        lambda self, running_batch: SimpleNamespace(
            running_batch=None, batch_to_run=None
        ),
        raising=False,
    )

    assert scheduler.is_fully_idle() is True
    message = _message()
    destination = scheduler._pd_receiver.reserve(message)
    assert scheduler.is_fully_idle() is False
    assert scheduler.is_fully_idle(for_health_check=True) is True

    scheduler._pd_receiver.commit(message, destination)
    assert scheduler.is_fully_idle() is False
    scheduler._drain_decode_admissions()
    assert scheduler.outbox.empty()
    assert scheduler.is_fully_idle() is False

    scheduler.req_to_token_pool.capacity = 4
    scheduler._drain_decode_admissions()
    assert [req.rid for req in scheduler.waiting_queue] == ["request-1"]
    assert scheduler.outbox.get_nowait().type == "admitted"
    assert scheduler.is_fully_idle() is False

    with ThreadPoolExecutor(1) as threads:
        threads.submit(scheduler.abort, "request-1").result(timeout=5)
    assert released == []
    assert scheduler.is_fully_idle() is False

    scheduler.running_batch = None
    scheduler.get_next_batch_to_run()
    assert released == [("request-1", threading.get_ident())]
    assert scheduler.is_fully_idle() is True


def test_decode_flush_drains_releases_and_gates_new_reservations(monkeypatch):
    scheduler = _decode_scheduler()
    req = SimpleNamespace(rid="request-1")
    scheduler.waiting_queue = [req]
    scheduler._release_request_kv_cache = Mock()
    message = _message()

    def abort(self, request_id, **kwargs):
        self.waiting_queue = [
            req for req in self.waiting_queue if req.rid != request_id
        ]

    monkeypatch.setattr(OmniScheduler, "abort", abort)

    def upstream_flush(self):
        self._release_request_kv_cache.assert_called_once_with(req)
        with pytest.raises(RuntimeError, match="not accepting reservations"):
            self._pd_receiver.reserve(message)
        return True

    monkeypatch.setattr(_Upstream, "flush_cache", upstream_flush)
    scheduler.abort(req.rid)
    assert scheduler.flush_cache() is True

    destination = scheduler._pd_receiver.reserve(message)
    scheduler._pd_receiver.abort(message, destination, RuntimeError("test cleanup"))


def test_weight_update_waits_for_pd_kv_and_gates_reservations(monkeypatch):
    scheduler = _decode_scheduler()

    def run_update(self, payload, update_fn, result_data, **kwargs):
        success, message = update_fn(payload)
        return {"success": success, "message": message}

    monkeypatch.setattr(OmniScheduler, "_run_weight_update_with_lifecycle", run_update)
    update = Mock(return_value=(True, "updated"))
    held_message = _message()
    held_destination = scheduler._pd_receiver.reserve(held_message)
    result = scheduler._run_weight_update_with_lifecycle({}, update, {})
    assert result == {"success": False, "message": "PD-owned KV is still in flight"}
    update.assert_not_called()

    scheduler._pd_receiver.abort(
        held_message, held_destination, RuntimeError("test cleanup")
    )
    message = _message(transfer_id="transfer-2")

    def gated_update(payload):
        with pytest.raises(RuntimeError, match="not accepting reservations"):
            scheduler._pd_receiver.reserve(message)
        return True, "updated"

    result = scheduler._run_weight_update_with_lifecycle({}, gated_update, {})
    assert result == {"success": True, "message": "updated"}
    destination = scheduler._pd_receiver.reserve(message)
    scheduler._pd_receiver.abort(message, destination, RuntimeError("test cleanup"))


def test_deferred_admission_abort_frees_committed_pages_once():
    scheduler = _decode_scheduler()
    scheduler.req_to_token_pool.capacity = 0
    scheduler._pd_admissions.put(DecodeAdmission(_continuation(), _allocation()))
    scheduler._drain_decode_admissions()
    assert scheduler._pd_deferred_admission is not None
    assert scheduler.token_to_kv_pool_allocator.freed == []
    scheduler._aborted_request_ids.add("request-1")
    scheduler._drain_decode_admissions()
    scheduler._drain_decode_admissions()
    assert len(scheduler.token_to_kv_pool_allocator.freed) == 1
    assert scheduler.waiting_queue == []


def test_stop_string_requires_and_uses_model_tokenizer():
    req = _prefill_req()
    req.sampling_params.stop_strs = ["END"]
    req.sampling_params.stop_str_max_len = 3
    continuation = replace(
        _continuation(),
        sampling_params=pd_utils._sampling_params_to_dict(req.sampling_params),
    )
    pool = _ReqPool()
    with pytest.raises(ValueError, match="tokenizer"):
        req_from_continuation(
            continuation,
            _allocation(),
            req_to_token_pool=pool,
            state_restorer=lambda *args: None,
        )
    assert pool.active == {}
    tokenizer = SimpleNamespace(
        eos_token_id=2, additional_stop_token_ids=[], decode=lambda ids: "END"
    )
    restored = req_from_continuation(
        continuation,
        _allocation(),
        req_to_token_pool=pool,
        state_restorer=lambda req, *_: setattr(req, "tokenizer", tokenizer),
    )
    restored.update_finish_state()
    assert restored.finished_reason.to_json()["type"] == "stop"


def _transfer(request_id="request-1", **updates):
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
    transfer = _transfer()

    make_stage()._discard_kv_transfer(transfer)

    transfer.lease.release.assert_called_once_with()


def test_slow_ack_does_not_block_outbox_and_early_cancellation_releases():
    async def run():
        stage = make_stage()
        stage._running = True
        slow_ack, fast_sent = asyncio.Event(), asyncio.Event()

        async def send_kv_pages(*, request_id, lease, **kwargs):
            try:
                if request_id == "slow":
                    await slow_ack.wait()
                else:
                    fast_sent.set()
            finally:
                lease.release()

        stage._comm.send_kv_pages = send_kv_pages
        from sglang_omni.scheduling.messages import OutgoingMessage

        transfers = [_transfer("slow"), _transfer("fast")]
        for transfer in transfers:
            stage._active_requests.add(transfer.request_id)
            stage.scheduler.outbox.put(
                OutgoingMessage(transfer.request_id, "kv_transfer", transfer)
            )
        drain = asyncio.create_task(stage._drain_outbox())
        try:
            await asyncio.wait_for(fast_sent.wait(), 5)
            transfers[0].lease.release.assert_not_called()
            slow_ack.set()
            await asyncio.gather(*tuple(stage._receive_tasks))
            for transfer in transfers:
                transfer.lease.release.assert_called_once()
            cancelled = _transfer("cancelled")
            stage._launch_kv_transfer(cancelled)
            task = next(iter(stage._receive_tasks))
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            cancelled.lease.release.assert_called_once()
        finally:
            stage._running = False
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)

    asyncio.run(run())


def test_missing_binding_releases_before_comm_takes_ownership():
    async def run():
        stage = make_stage(replica_topology={"decode": ["decode@r0", "decode@r1"]})
        transfer = _transfer()
        stage._active_requests.add(transfer.request_id)
        await stage._send_kv_transfer(transfer)
        transfer.lease.release.assert_called_once()
        assert transfer.request_id not in stage._active_requests
        assert "no replica binding" in stage.control_plane.completions[0].error

    asyncio.run(run())


def test_memory_pressure_fails_one_request_without_upstream_rebootstrap():
    from sglang.srt.disaggregation.utils import DisaggregationMode

    scheduler = _decode_scheduler()
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.5)
    scheduler.tree_cache = SimpleNamespace(req_to_token_pool=SimpleNamespace())
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.server_args = SimpleNamespace()
    scheduler.abort = Mock()
    scheduler._emit_request_error = Mock()
    batch = SimpleNamespace(
        batch_size=lambda: 1,
        filter_batch=lambda: None,
        is_empty=lambda: False,
        check_decode_mem=lambda: False,
        retract_decode=lambda: ([SimpleNamespace(rid="full")], 0.5, []),
        prepare_for_decode=Mock(),
    )
    scheduler.update_running_batch(batch)
    scheduler.abort.assert_called_once_with("full")
    assert "cannot resume" in str(scheduler._emit_request_error.call_args.args[1])
    batch.prepare_for_decode.assert_called_once()


def test_non_pd_scheduler_does_not_need_kv_registration():
    scheduler = SimpleScheduler(lambda payload: payload)
    stage = make_stage(scheduler=scheduler)
    assert stage.scheduler is scheduler
    assert stage._comm._kv_pools == {}


def test_binding_survives_comm_handoff_admission_and_next_stage(monkeypatch):
    import sglang_omni.platforms as platforms
    from sglang_omni.comm.data_ref import TransportKind
    from sglang_omni.scheduling.messages import OutgoingMessage
    from tests.unit_test.pipeline.test_kv_transfer import _pool, _start_pair

    monkeypatch.setattr(
        platforms.current_platform,
        "get_intra_node_transport",
        lambda: TransportKind.CUDA_IPC,
    )

    async def run():
        _, source, destination = await _start_pair()
        scheduler = _decode_scheduler()
        receiver = DecodeKVReceiver(
            pool_id="decode:kv",
            allocator=scheduler.token_to_kv_pool_allocator,
            admissions=scheduler._pd_admissions,
            resume_schema="test-v1",
        )
        receiver._allocator.next_slot = 0
        source.register_kv_pool(_pool("prefill:kv"))
        destination.register_kv_pool(_pool("decode:kv"))
        destination.register_kv_receiver("decode:kv", receiver)
        topology = {"post": ["post@r0", "post@r1"]}
        prefill = make_stage(name="source", replica_topology=topology)
        prefill._comm = source
        prefill._record_replica_bindings("request-1", {"post": 1})
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
            continuation = _continuation()
            transfer = _transfer(
                transfer_id=continuation.transfer_id,
                to_stage="destination",
                metadata={"decode_continuation": continuation.encode()},
            )
            await prefill._send_kv_transfer(transfer)
            transfer.lease.release.assert_called_once()
            scheduler._drain_decode_admissions()
            scheduler.outbox.put(
                OutgoingMessage(
                    "request-1",
                    "result",
                    scheduler.waiting_queue[0]._omni_data.stage_payload,
                )
            )
            decode._running = True
            drain = asyncio.create_task(decode._drain_outbox())
            await asyncio.wait_for(dispatched.wait(), 5)
            kwargs = dispatcher.send_payload.call_args.kwargs
            assert kwargs["to_stage"] == "post@r1"
            assert kwargs["replica_bindings"] == {"post": 1}
        finally:
            decode._running = False
            if drain is not None:
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)
            await source.close()
            await destination.close()

    asyncio.run(run())


def test_replicated_decode_target_uses_bound_instance_and_pool():
    async def run():
        stage = make_stage(replica_topology={"decode": ["decode@r0", "decode@r1"]})
        stage._record_replica_bindings("request-1", {"decode": 1})
        transfer = _transfer()

        async def send(**kwargs):
            kwargs["lease"].release()

        stage._comm.send_kv_pages = AsyncMock(side_effect=send)
        await stage._send_kv_transfer(transfer)
        kwargs = stage._comm.send_kv_pages.call_args.kwargs
        assert kwargs["to_stage"] == "decode@r1"
        assert kwargs["target_pool_id"] == "decode@r1:kv"
        transfer.lease.release.assert_called_once()

    asyncio.run(run())
