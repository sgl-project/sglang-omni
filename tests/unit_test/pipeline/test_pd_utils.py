# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
from array import array
from types import SimpleNamespace

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.comm import KVPageTransfer
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.proto import (
    KVBufferSpec,
    KVPoolLayout,
    KVTransferPrepareMessage,
    OmniRequest,
    StagePayload,
)
from sglang_omni.scheduling.pd_utils import (
    DecodeAdmission,
    DecodeContinuation,
    DecodeKVReceiver,
    ReservedKV,
    continuation_from_req,
    defer_first_token_finish,
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
    def __init__(self) -> None:
        self.next_slot = 7
        self.freed = []

    def available_size(self) -> int:
        return 32

    def alloc(self, count: int):
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
    receiver = DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=_KVAllocator(),
        admissions=admissions,
        resume_schema="test-v1",
    )
    continuation = _continuation()
    message = KVTransferPrepareMessage(
        request_id=continuation.request_id,
        transfer_id=continuation.transfer_id,
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
