# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import queue
import threading
from array import array
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.comm import KVPageTransfer
from sglang_omni.pipeline.replicas import ReplicaTopology
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.proto import (
    KVBufferSpec,
    KVPoolLayout,
    KVTransferPrepareMessage,
    OmniRequest,
    StagePayload,
)
from sglang_omni.scheduling import pd_utils as pd_utils_module
from sglang_omni.scheduling.pd_utils import (
    DecodeAdmission,
    DecodeContinuation,
    DecodeKVReceiver,
    ReservedKV,
    continuation_from_req,
    defer_first_token_finish,
    drain_due_releases,
    req_from_continuation,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


class _ReqPool:
    def __init__(self, capacity: int = 4) -> None:
        self.capacity = capacity
        self.req_to_token = torch.zeros((capacity, 32), dtype=torch.int64)
        self.active: dict[int, Req] = {}

    @property
    def size(self) -> int:
        return self.capacity

    def alloc(self, reqs):
        if len(self.active) + len(reqs) > self.capacity:
            return None
        indices = []
        for req in reqs:
            index = next(i for i in range(self.capacity) if i not in self.active)
            req.req_pool_idx = index
            self.active[index] = req
            indices.append(index)
        return torch.tensor(indices, dtype=torch.int64)

    def write(self, key, value) -> None:
        self.req_to_token[key] = value

    def free(self, req) -> None:
        self.active.pop(req.req_pool_idx, None)
        req.req_pool_idx = None


class _KVAllocator:
    def __init__(self, *, available: int = 32) -> None:
        self.next_slot = 7
        self.available = available
        self.alloc_calls = 0
        self.freed = []

    def available_size(self) -> int:
        return self.available

    def alloc(self, count: int):
        self.alloc_calls += 1
        slots = torch.arange(self.next_slot, self.next_slot + count)
        self.next_slot += count
        return slots

    def free(self, slots) -> None:
        self.freed.append(slots)


def _prefill_req(*, max_new_tokens: int = 16) -> Req:
    sampling = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        stop_token_ids={2},
        sampling_seed=17,
    )
    sampling.normalize(None)
    req = Req(
        rid="request-1",
        origin_input_text="",
        origin_input_ids=array("q", [10, 11, 12]),
        sampling_params=sampling,
        vocab_size=128,
        eos_token_ids={2},
    )
    req.output_ids.append(42)
    payload = StagePayload(
        request_id=req.rid,
        request=OmniRequest(inputs=None, params={"stream": True}),
        data={"prompt": [10, 11, 12]},
    )
    req._omni_data = SGLangARRequestData(
        input_ids=torch.tensor([10, 11, 12]),
        output_ids=req.output_ids,
        req=req,
        stage_payload=payload,
    )
    return req


def _state_builder(req):
    return req._omni_data.stage_payload.to_dict(), None, list(req.origin_input_ids)


def _continuation() -> DecodeContinuation:
    return continuation_from_req(_prefill_req(), "transfer-1", _state_builder)


def _allocation() -> ReservedKV:
    slots = torch.tensor([7, 8, 9], dtype=torch.int64)
    return ReservedKV(slots=slots, page_indices=(7, 8, 9), seq_len=3)


def _message(
    continuation: DecodeContinuation | None = None,
    *,
    request_id: str | None = None,
    transfer_id: str | None = None,
) -> KVTransferPrepareMessage:
    continuation = continuation or _continuation()
    if transfer_id is not None:
        continuation = replace(continuation, transfer_id=transfer_id)
    return KVTransferPrepareMessage(
        request_id=request_id or continuation.request_id,
        transfer_id=transfer_id or continuation.transfer_id,
        from_stage="prefill",
        to_stage="decode",
        source_pool_id="prefill:kv",
        target_pool_id="decode:kv",
        source_page_indices=(1, 2, 3),
        source_layout=KVPoolLayout(
            layout_id="test",
            page_size=1,
            buffers=(KVBufferSpec("kv", 4),),
        ),
        metadata={"decode_continuation": continuation.encode()},
    )


def _receiver(allocator=None, admissions=None) -> DecodeKVReceiver:
    return DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=allocator or _KVAllocator(),
        admissions=admissions or queue.SimpleQueue(),
        allowed_resume_schemas=frozenset({"test-v1"}),
    )


def test_continuation_round_trip_rebuilds_prebuilt_request() -> None:
    continuation = DecodeContinuation.decode(_continuation().encode())
    req = req_from_continuation(
        continuation,
        _allocation(),
        req_to_token_pool=_ReqPool(),
        state_restorer=lambda req, _data, _resume: setattr(req, "tokenizer", None),
    )

    assert list(req.origin_input_ids) == [10, 11, 12]
    assert list(req.output_ids) == [42]
    assert req.sampling_params.stop_token_ids == {2}
    assert req.prefix_indices.tolist() == [7, 8, 9]
    assert req.kv.kv_allocated_len == 3


def test_decode_receiver_commits_directly_to_admission_queue() -> None:
    admissions = queue.SimpleQueue()
    receiver = _receiver(admissions=admissions)
    continuation = _continuation()
    message = _message(continuation)

    destination = receiver.reserve(message)
    receiver.commit(message, destination)

    admission = admissions.get_nowait()
    assert admission.continuation.request_id == "request-1"
    assert admission.allocation.page_indices == destination.page_indices


def test_prefill_defers_first_token_stop_policy_to_decode() -> None:
    req = _prefill_req(max_new_tokens=1)
    del req.output_ids[:]
    original_max = req.sampling_params.max_new_tokens

    with defer_first_token_finish([req]):
        req.output_ids.append(2)
        req.update_finish_state()
        assert not req.finished()

    assert req.sampling_params.max_new_tokens == original_max
    req.update_finish_state()
    assert req.finished()


def test_decode_scheduler_admits_committed_request_without_controller(
    monkeypatch,
) -> None:
    from sglang.srt import runtime_context

    monkeypatch.setattr(runtime_context, "get_model", lambda: None, raising=False)
    monkeypatch.setattr(runtime_context, "get_serving", lambda: None, raising=False)
    from sglang_omni.scheduling.pd_scheduler import OmniDecodeScheduler

    scheduler = object.__new__(OmniDecodeScheduler)
    scheduler._pd_admissions = queue.SimpleQueue()
    scheduler._pd_admissions.put(DecodeAdmission(_continuation(), _allocation()))
    scheduler._pd_deferred_admission = None
    scheduler._pd_admission_lock = threading.Lock()
    scheduler._pd_state_restorer = lambda req, _data, _resume: setattr(
        req, "tokenizer", None
    )
    scheduler._aborted_request_ids = set()
    scheduler.req_to_token_pool = _ReqPool()
    scheduler.token_to_kv_pool_allocator = _KVAllocator()
    scheduler.waiting_queue = []
    scheduler.outbox = queue.Queue()
    scheduler.is_entry_rank = True

    scheduler._drain_decode_admissions()

    assert [req.rid for req in scheduler.waiting_queue] == ["request-1"]
    assert scheduler.outbox.get_nowait().type == "admitted"


def test_discarded_stage_transfer_releases_source_lease() -> None:
    released = []
    lease = SimpleNamespace(release=lambda: released.append(True))
    transfer = KVPageTransfer(
        request_id="request-1",
        transfer_id="transfer-1",
        source_pool_id="prefill:kv",
        target_pool_id="decode:kv",
        source_page_indices=(1,),
        to_stage="decode",
        lease=lease,
    )

    Stage._discard_kv_transfer(transfer)

    assert released == [True]


def test_continuation_rejects_unknown_version_keys_and_id_mismatch() -> None:
    encoded = _continuation().encode()
    values = msgspec.msgpack.decode(encoded)
    values["version"] = 99
    with pytest.raises(ValueError, match="unsupported decode continuation version"):
        DecodeContinuation.decode(msgspec.msgpack.encode(values))

    values["version"] = 1
    values["future_field"] = True
    with pytest.raises(ValueError, match="unknown decode continuation fields"):
        DecodeContinuation.decode(msgspec.msgpack.encode(values))

    allocator = _KVAllocator()
    with pytest.raises(ValueError, match="ids differ"):
        _receiver(allocator=allocator).reserve(_message(request_id="another-request"))
    assert allocator.alloc_calls == 0


@pytest.mark.parametrize(
    "update,message",
    [
        ({"request_id": ""}, "ids must be non-empty"),
        ({"transfer_id": ""}, "ids must be non-empty"),
        ({"output_ids": []}, "requires the Prefill token"),
        ({"vocab_size": 0}, "vocab_size must be positive"),
    ],
)
def test_continuation_rejects_missing_boundary_state(update, message) -> None:
    values = msgspec.msgpack.decode(_continuation().encode())
    values.update(update)
    with pytest.raises(ValueError, match=message):
        DecodeContinuation.decode(msgspec.msgpack.encode(values))


def test_receiver_rejects_unsupported_resume_schema_before_allocation() -> None:
    allocator = _KVAllocator()
    continuation = replace(
        _continuation(),
        multimodal_resume={"schema": "future"},
    )
    with pytest.raises(ValueError, match="unsupported multimodal resume schema"):
        _receiver(allocator=allocator).reserve(_message(continuation))
    assert allocator.alloc_calls == 0


def test_pd_runtime_rejects_speculative_scheduler_at_construction(monkeypatch) -> None:
    from sglang.srt import runtime_context

    monkeypatch.setattr(runtime_context, "get_model", lambda: None, raising=False)
    monkeypatch.setattr(runtime_context, "get_serving", lambda: None, raising=False)
    from sglang_omni.scheduling.pd_scheduler import _validate_pd_runtime

    scheduler = SimpleNamespace(
        tp_size=1,
        page_size=1,
        server_args=SimpleNamespace(disable_radix_cache=True),
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
    )
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        _validate_pd_runtime(scheduler)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("json_schema", "{}", "structured-output"),
        ("regex", ".*", "structured-output"),
        ("ebnf", "root", "structured-output"),
        ("structural_tag", "tag", "structured-output"),
    ],
)
def test_prefill_rejects_unsupported_sampling_before_state_projection(
    field, value, message
) -> None:
    req = _prefill_req()
    setattr(req.sampling_params, field, value)
    called = []
    with pytest.raises(NotImplementedError, match=message):
        continuation_from_req(
            req,
            "transfer",
            lambda _req: called.append(True),
        )
    assert called == []


def test_prefill_rejects_custom_logits_projected_inputs_and_bad_sampling_state() -> (
    None
):
    req = _prefill_req()
    req.custom_logit_processor = "processor"
    with pytest.raises(NotImplementedError, match="custom logit"):
        continuation_from_req(req, "transfer", _state_builder)

    req = _prefill_req()
    req._omni_data.input_embeds_are_projected = True
    with pytest.raises(NotImplementedError, match="projected input"):
        continuation_from_req(req, "transfer", _state_builder)

    req = _prefill_req()
    req.sampling_params.custom_params = {"bad": object()}
    with pytest.raises(ValueError, match="sampling state is not serializable"):
        continuation_from_req(req, "transfer", _state_builder)


def test_receiver_duplicate_commit_mismatch_abort_and_close_free_exactly_once() -> None:
    allocator = _KVAllocator()
    admissions = queue.SimpleQueue()
    receiver = _receiver(allocator=allocator, admissions=admissions)
    message = _message()
    destination = receiver.reserve(message)

    with pytest.raises(RuntimeError, match="duplicate KV transfer"):
        receiver.reserve(message)
    assert allocator.alloc_calls == 1

    wrong = type(destination)(destination.pool_id, (99, 98, 97))
    with pytest.raises(RuntimeError, match="does not match"):
        receiver.commit(message, wrong)
    assert len(allocator.freed) == 1
    receiver.abort(message, destination, RuntimeError("late"))
    assert len(allocator.freed) == 1

    second = _message(transfer_id="transfer-2")
    receiver.reserve(second)
    receiver.abort(second, None, RuntimeError("receive failed"))
    receiver.abort(second, None, RuntimeError("duplicate abort"))
    assert len(allocator.freed) == 2

    wrong_pool = _message(transfer_id="transfer-wrong-pool")
    reserved = receiver.reserve(wrong_pool)
    wrong_destination = type(reserved)("another-pool", reserved.page_indices)
    with pytest.raises(RuntimeError, match="does not match"):
        receiver.commit(wrong_pool, wrong_destination)
    assert len(allocator.freed) == 3

    third = _message(transfer_id="transfer-3")
    receiver.reserve(third)
    receiver.close()
    receiver.close()
    assert len(allocator.freed) == 4
    with pytest.raises(RuntimeError, match="receiver is closed"):
        receiver.reserve(_message(transfer_id="transfer-4"))
    assert allocator.alloc_calls == 4


def test_receiver_commit_transfers_ownership_to_admission() -> None:
    allocator = _KVAllocator()
    admissions = queue.SimpleQueue()
    receiver = _receiver(allocator=allocator, admissions=admissions)
    message = _message()
    destination = receiver.reserve(message)
    receiver.commit(message, destination)
    with pytest.raises(RuntimeError, match="duplicate KV transfer"):
        receiver.reserve(message)
    receiver.abort(message, destination, RuntimeError("late"))
    receiver.close()

    admission = admissions.get_nowait()
    assert admission.allocation.page_indices == destination.page_indices
    assert allocator.freed == []


def test_receiver_transfer_tombstones_are_bounded_and_evict_oldest(
    monkeypatch,
) -> None:
    monkeypatch.setattr(pd_utils_module, "_TRANSFER_TOMBSTONE_LIMIT", 2)
    allocator = _KVAllocator()
    receiver = _receiver(allocator=allocator)

    def abort_transfer(transfer_id: str) -> None:
        message = _message(transfer_id=transfer_id)
        destination = receiver.reserve(message)
        receiver.abort(message, destination, RuntimeError("test abort"))

    abort_transfer("transfer-0")
    abort_transfer("transfer-1")
    with pytest.raises(RuntimeError, match="duplicate KV transfer"):
        receiver.reserve(_message(transfer_id="transfer-1"))

    abort_transfer("transfer-2")

    assert list(receiver._transfer_tombstones) == ["transfer-1", "transfer-2"]
    with pytest.raises(RuntimeError, match="duplicate KV transfer"):
        receiver.reserve(_message(transfer_id="transfer-2"))

    # Once the oldest ID leaves the bounded window, it can be reused.
    abort_transfer("transfer-0")
    assert list(receiver._transfer_tombstones) == ["transfer-2", "transfer-0"]
    assert allocator.alloc_calls == 4


def test_decode_pool_exhaustion_defers_without_losing_committed_kv(monkeypatch) -> None:
    from sglang.srt import runtime_context

    monkeypatch.setattr(runtime_context, "get_model", lambda: None, raising=False)
    monkeypatch.setattr(runtime_context, "get_serving", lambda: None, raising=False)
    from sglang_omni.scheduling.pd_scheduler import OmniDecodeScheduler

    req_pool = _ReqPool(capacity=0)
    allocator = _KVAllocator()
    admission = DecodeAdmission(_continuation(), _allocation())
    scheduler = object.__new__(OmniDecodeScheduler)
    scheduler._pd_admissions = queue.SimpleQueue()
    scheduler._pd_admissions.put(admission)
    scheduler._pd_deferred_admission = None
    scheduler._pd_admission_lock = threading.Lock()
    scheduler._pd_state_restorer = lambda req, _data, _resume: setattr(
        req, "tokenizer", None
    )
    scheduler._aborted_request_ids = set()
    scheduler.req_to_token_pool = req_pool
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler.waiting_queue = []
    scheduler.outbox = queue.Queue()
    scheduler.is_entry_rank = True

    scheduler._drain_decode_admissions()
    scheduler._drain_decode_admissions()
    assert scheduler._pd_deferred_admission is admission
    assert allocator.freed == []

    req_pool.capacity = 1
    req_pool.req_to_token = torch.zeros((1, 32), dtype=torch.int64)
    scheduler._drain_decode_admissions()
    assert scheduler._pd_deferred_admission is None
    assert [req.rid for req in scheduler.waiting_queue] == ["request-1"]
    assert allocator.freed == []


def test_aborted_committed_admission_and_invalid_restore_free_once(monkeypatch) -> None:
    from sglang.srt import runtime_context

    monkeypatch.setattr(runtime_context, "get_model", lambda: None, raising=False)
    monkeypatch.setattr(runtime_context, "get_serving", lambda: None, raising=False)
    from sglang_omni.scheduling.pd_scheduler import OmniDecodeScheduler

    allocator = _KVAllocator()
    scheduler = object.__new__(OmniDecodeScheduler)
    scheduler._pd_admissions = queue.SimpleQueue()
    scheduler._pd_admissions.put(DecodeAdmission(_continuation(), _allocation()))
    scheduler._pd_deferred_admission = None
    scheduler._pd_admission_lock = threading.Lock()
    scheduler._pd_state_restorer = lambda req, data, resume: None
    scheduler._aborted_request_ids = {"request-1"}
    scheduler.req_to_token_pool = _ReqPool()
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler.waiting_queue = []
    scheduler.outbox = queue.Queue()
    scheduler.is_entry_rank = True

    scheduler._drain_decode_admissions()
    scheduler._drain_decode_admissions()
    assert len(allocator.freed) == 1

    scheduler._aborted_request_ids.clear()
    scheduler._pd_state_restorer = lambda req, data, resume: (_ for _ in ()).throw(
        ValueError("bad resume")
    )
    scheduler._pd_admissions.put(DecodeAdmission(_continuation(), _allocation()))
    scheduler._drain_decode_admissions()
    assert len(allocator.freed) == 2
    assert scheduler.outbox.get_nowait().type == "admitted"
    assert scheduler.outbox.get_nowait().type == "error"


def test_decode_shutdown_drains_receiver_deferred_and_queued_ownership(
    monkeypatch,
) -> None:
    from sglang.srt import runtime_context

    monkeypatch.setattr(runtime_context, "get_model", lambda: None, raising=False)
    monkeypatch.setattr(runtime_context, "get_serving", lambda: None, raising=False)
    from sglang_omni.scheduling.pd_scheduler import OmniDecodeScheduler

    allocator = _KVAllocator()
    scheduler = object.__new__(OmniDecodeScheduler)
    scheduler._request_admission_lock = threading.Lock()
    scheduler._pending_request_admissions = {}
    scheduler._pd_admission_lock = threading.Lock()
    scheduler._pd_deferred_admission = DecodeAdmission(_continuation(), _allocation())
    scheduler._pd_admissions = queue.SimpleQueue()
    scheduler._pd_admissions.put(DecodeAdmission(_continuation(), _allocation()))
    scheduler.token_to_kv_pool_allocator = allocator
    receiver_closed = []
    scheduler._pd_receiver = SimpleNamespace(close=lambda: receiver_closed.append(True))

    scheduler._discard_pending_request_admissions()
    scheduler._discard_pending_request_admissions()

    assert receiver_closed == [True, True]
    assert len(allocator.freed) == 2


def test_prefill_queues_one_real_token_handoff_and_source_lease_is_idempotent(
    monkeypatch,
) -> None:
    from sglang.srt import runtime_context

    monkeypatch.setattr(runtime_context, "get_model", lambda: None, raising=False)
    monkeypatch.setattr(runtime_context, "get_serving", lambda: None, raising=False)
    from sglang_omni.scheduling.pd_scheduler import OmniPrefillScheduler

    released = []
    monkeypatch.setattr(
        "sglang.srt.mem_cache.common.release_kv_cache",
        lambda req, tree: released.append((req.rid, tree)),
    )
    req = _prefill_req(max_new_tokens=1)
    req.req_pool_idx = 0
    req.inflight_middle_chunks = 0
    req_pool = _ReqPool()
    req_pool.req_to_token[0, :3] = torch.tensor([3, 4, 5])
    scheduler = object.__new__(OmniPrefillScheduler)
    scheduler._pd_state_builder = _state_builder
    scheduler._pd_stage_name = "thinker_prefill"
    scheduler._pd_partner_stage = "thinker_decode"
    scheduler._pd_decode_targets = ("thinker_decode",)
    scheduler._pd_handoff_seq = 0
    scheduler._pd_due_releases = queue.SimpleQueue()
    scheduler._pd_pool_id = "thinker_prefill:kv"
    scheduler.req_to_token_pool = req_pool
    scheduler.tree_cache = "tree"
    scheduler.outbox = queue.Queue()
    batch = SimpleNamespace(reqs=[req], batch_is_full=True)

    scheduler._handoff_prefilled_requests(batch, {id(req)})

    message = scheduler.outbox.get_nowait()
    transfer = message.data
    continuation = DecodeContinuation.decode(transfer.metadata["decode_continuation"])
    assert continuation.output_ids == [42]
    assert transfer.source_page_indices == (3, 4, 5)
    assert transfer.to_stage == "thinker_decode"
    assert batch.reqs == []
    assert req._omni_data is None
    assert released == []

    # The comm event loop calls release; the tree belongs to this thread, so
    # the lease only records that a release is due.
    transfer.lease.release()
    transfer.lease.release()
    assert released == []

    assert drain_due_releases(scheduler._pd_due_releases, "tree") == 1
    assert released == [("request-1", "tree")]
    assert drain_due_releases(scheduler._pd_due_releases, "tree") == 0


def test_chunked_prefill_does_not_handoff_before_final_prefill_chunk(
    monkeypatch,
) -> None:
    from sglang.srt import runtime_context

    monkeypatch.setattr(runtime_context, "get_model", lambda: None, raising=False)
    monkeypatch.setattr(runtime_context, "get_serving", lambda: None, raising=False)
    from sglang_omni.scheduling.pd_scheduler import OmniPrefillScheduler

    req = _prefill_req()
    req.inflight_middle_chunks = 1
    scheduler = object.__new__(OmniPrefillScheduler)
    scheduler.outbox = queue.Queue()
    batch = SimpleNamespace(reqs=[req], batch_is_full=False)

    scheduler._handoff_prefilled_requests(batch, {id(req)})

    assert batch.reqs == [req]
    assert scheduler.outbox.empty()


def test_slow_ack_does_not_block_an_unrelated_transfer() -> None:
    async def run() -> None:
        slow_ack = asyncio.Event()
        completed = []

        class _Comm:
            async def send_kv_pages(self, *, request_id, lease, **kwargs):
                del kwargs
                try:
                    if request_id == "slow":
                        await slow_ack.wait()
                    completed.append(request_id)
                finally:
                    lease.release()

        stage = object.__new__(Stage)
        # The send resolves a replicated target through the request's binding.
        stage._replica_topology = ReplicaTopology(replicas={})
        stage._replica_bindings = {}
        stage._comm = _Comm()
        stage._receive_tasks = set()
        stage._clear_request_state = lambda request_id: completed.append(
            f"clear:{request_id}"
        )
        stage._on_background_task_done = lambda task, context: None
        releases = []

        def transfer(request_id):
            return KVPageTransfer(
                request_id=request_id,
                transfer_id=f"{request_id}-transfer",
                source_pool_id="prefill:kv",
                target_pool_id="decode:kv",
                source_page_indices=(1,),
                to_stage="decode",
                lease=SimpleNamespace(release=lambda: releases.append(request_id)),
            )

        stage._launch_kv_transfer(transfer("slow"))
        stage._launch_kv_transfer(transfer("fast"))
        for _ in range(20):
            if "clear:fast" in completed:
                break
            await asyncio.sleep(0)
        assert "fast" in completed
        assert "slow" not in completed
        assert releases == ["fast"]

        slow_ack.set()
        await asyncio.gather(*tuple(stage._receive_tasks))
        assert releases == ["fast", "slow"]

    asyncio.run(run())


def test_send_failure_releases_source_and_reports_request_failure() -> None:
    async def run() -> None:
        released = []
        failures = []

        class _Comm:
            async def send_kv_pages(self, *, lease, **kwargs):
                del kwargs
                try:
                    raise RuntimeError("copy failed")
                finally:
                    lease.release()

        stage = object.__new__(Stage)
        stage.name = "prefill"
        stage._replica_topology = ReplicaTopology(replicas={})
        stage._replica_bindings = {}
        stage._comm = _Comm()
        stage._clear_request_state = lambda request_id: None

        async def fail(request_id, error):
            failures.append((request_id, error))

        stage._send_failure = fail
        transfer = KVPageTransfer(
            request_id="request",
            transfer_id="transfer",
            source_pool_id="prefill:kv",
            target_pool_id="decode:kv",
            source_page_indices=(1,),
            to_stage="decode",
            lease=SimpleNamespace(release=lambda: released.append(True)),
        )
        await stage._send_kv_transfer(transfer)

        assert released == [True]
        assert failures == [("request", "copy failed")]

    asyncio.run(run())


def test_priority_survives_the_handoff() -> None:
    """A request admitted ahead of others must not be demoted on Decode."""
    from sglang_omni.scheduling.pd_utils import DecodeContinuation

    carried = DecodeContinuation(
        request_id="r",
        transfer_id="t",
        origin_input_ids=[1, 2],
        output_ids=[3],
        vocab_size=32,
        sampling_params={},
        stage_payload={},
        priority=7,
    )

    assert DecodeContinuation.decode(carried.encode()).priority == 7


def test_a_continuation_without_a_priority_stays_valid() -> None:
    """Most requests have none, and that must not be an error."""
    from sglang_omni.scheduling.pd_utils import DecodeContinuation

    plain = DecodeContinuation(
        request_id="r",
        transfer_id="t",
        origin_input_ids=[1],
        output_ids=[2],
        vocab_size=32,
        sampling_params={},
        stage_payload={},
    )

    assert DecodeContinuation.decode(plain.encode()).priority is None
