# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
from array import array
from dataclasses import replace

import torch
from sglang.srt.managers.schedule_batch import CaptureHiddenMode, Req
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto import (
    KVBufferSpec,
    KVPoolLayout,
    KVTransferPrepareMessage,
    OmniRequest,
    StagePayload,
)
from sglang_omni.scheduling.pd_utils import (
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
            req.kv.req_pool_idx = index
            self.active[index] = req
            indices.append(index)
        return indices

    def write(self, key, value) -> None:
        self.req_to_token[key] = value

    def free(self, req) -> None:
        assert req.kv.req_pool_idx is not None
        self.active.pop(req.kv.req_pool_idx, None)
        req.kv.req_pool_idx = None


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


def _prefill_req(
    *,
    max_new_tokens: int = 16,
    custom_params: dict | None = None,
    return_hidden_states: bool | str = False,
) -> Req:
    sampling = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        stop_token_ids={2},
        sampling_seed=17,
        custom_params=custom_params,
    )
    sampling.normalize(None)
    req = Req(
        rid="request-1",
        origin_input_text="",
        origin_input_ids=array("q", [10, 11, 12]),
        sampling_params=sampling,
        vocab_size=128,
        eos_token_ids={2},
        return_hidden_states=return_hidden_states,
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


def _message(*, transfer_id="transfer-1", **metadata):
    continuation = replace(_continuation(), transfer_id=transfer_id)
    return KVTransferPrepareMessage(
        request_id=continuation.request_id,
        transfer_id=continuation.transfer_id,
        from_stage="prefill",
        to_stage="decode",
        source_pool_id="prefill:kv",
        target_pool_id="decode:kv",
        source_page_indices=(1, 2, 3),
        source_layout=KVPoolLayout("test", 1, (KVBufferSpec("kv", 4),)),
        metadata={"decode_continuation": continuation.encode(), **metadata},
    )


def _receiver(admissions=None):
    return DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=_KVAllocator(),
        admissions=admissions if admissions is not None else queue.SimpleQueue(),
        resume_schema="test-v1",
    )


def test_continuation_round_trip_rebuilds_prebuilt_request() -> None:
    """Rebuild against SGLang's real request-row ownership contract."""

    continuation = DecodeContinuation.decode(_continuation().encode())
    req_to_token_pool = ReqToTokenPool(
        size=4,
        max_context_len=32,
        device="cpu",
        enable_memory_saver=False,
    )
    req = req_from_continuation(
        continuation,
        _allocation(),
        req_to_token_pool=req_to_token_pool,
        state_restorer=lambda req, _data, _resume: setattr(req, "tokenizer", None),
    )

    assert list(req.origin_input_ids) == [10, 11, 12]
    assert list(req.output_ids) == [42]
    assert req.sampling_params.stop_token_ids == {2}
    assert req.prefix_indices.tolist() == [7, 8, 9]
    assert req.kv.req_pool_idx is not None
    assert req_to_token_pool.req_to_token[req.kv.req_pool_idx, :3].tolist() == [7, 8, 9]
    assert req.kv.kv_committed_len == 3
    assert req.kv.kv_allocated_len == 3

    req_to_token_pool.free(req)
    assert req.kv.req_pool_idx is None
    assert req_to_token_pool.available_size() == req_to_token_pool.size


def _rebuilt_req(source):
    continuation = DecodeContinuation.decode(
        continuation_from_req(source, "transfer-1", _state_builder).encode()
    )
    return req_from_continuation(
        continuation,
        _allocation(),
        req_to_token_pool=_ReqPool(),
        state_restorer=lambda req, _data, _resume: setattr(req, "tokenizer", None),
    )


def test_continuation_preserves_the_hidden_state_mode() -> None:
    for mode, capture in (
        (False, CaptureHiddenMode.NULL),
        (True, CaptureHiddenMode.FULL),
        ("last", CaptureHiddenMode.LAST),
    ):
        req = _rebuilt_req(_prefill_req(return_hidden_states=mode))
        assert req.return_hidden_states == mode
        assert req.return_hidden_states_mode is capture


def test_continuation_strips_the_live_req_out_of_custom_params() -> None:
    source = _prefill_req(custom_params={"segment_timestamps": True})
    req = _rebuilt_req(source)
    assert req.sampling_params.custom_params["segment_timestamps"] is True
    assert req.sampling_params.custom_params["__req__"] is req
    assert source.sampling_params.custom_params["__req__"] is source


def test_decode_receiver_commits_directly_to_admission_queue() -> None:
    admissions = queue.SimpleQueue()
    receiver = _receiver(admissions)
    message = _message()

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
