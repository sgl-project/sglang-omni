# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
from array import array
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.managers.schedule_batch import CaptureHiddenMode, Req
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto import (
    KVBufferSpec,
    KVPoolLayout,
    KVTransferPrepareMessage,
    OmniRequest,
    StagePayload,
)
from sglang_omni.proto.session import SessionIdentity, SessionOperation
from sglang_omni.scheduling.pd_utils import (
    DecodeContinuation,
    DecodeKVReceiver,
    ReservedKV,
    continuation_from_req,
    defer_first_token_finish,
    req_from_continuation,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


class ReqPool:
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


class KVAllocator:
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


def prefill_req(
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
    req.omni_data = SGLangARRequestData(
        input_ids=torch.tensor([10, 11, 12]),
        output_ids=req.output_ids,
        req=req,
        stage_payload=payload,
    )
    return req


def state_builder(req):
    return req.omni_data.stage_payload.to_dict(), None, list(req.origin_input_ids)


def make_continuation() -> DecodeContinuation:
    return continuation_from_req(prefill_req(), "transfer-1", state_builder)


def make_allocation() -> ReservedKV:
    slots = torch.tensor([7, 8, 9], dtype=torch.int64)
    return ReservedKV(slots=slots, page_indices=(7, 8, 9), seq_len=3)


def make_message(*, transfer_id="transfer-1", **metadata):
    continuation = replace(make_continuation(), transfer_id=transfer_id)
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


def make_receiver(admissions=None):
    return DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=KVAllocator(),
        admissions=admissions if admissions is not None else queue.SimpleQueue(),
        resume_schema="test-v1",
    )


def test_continuation_round_trip_rebuilds_prebuilt_request() -> None:
    """Rebuild against SGLang's real request-row ownership contract."""

    continuation = DecodeContinuation.decode(make_continuation().encode())
    req_to_token_pool = ReqToTokenPool(
        size=4,
        max_context_len=32,
        device="cpu",
        enable_memory_saver=False,
    )
    req = req_from_continuation(
        continuation,
        make_allocation(),
        req_to_token_pool=req_to_token_pool,
        state_restorer=lambda req, data, _resume: setattr(req, "tokenizer", None),
    )

    assert list(req.origin_input_ids) == [10, 11, 12]
    assert list(req.output_ids) == [42]
    assert req.sampling_params.stop_token_ids == {2}
    assert req.prefix_len == 3
    assert req.extend_end == 3
    assert req.kv.req_pool_idx is not None
    assert req_to_token_pool.req_to_token[req.kv.req_pool_idx, :3].tolist() == [7, 8, 9]
    assert req.kv.kv_committed_len == 3
    assert req.kv.kv_allocated_len == 3

    req_to_token_pool.free(req)
    assert req.kv.req_pool_idx is None
    assert req_to_token_pool.available_size() == req_to_token_pool.size


def rebuilt_req(source):
    continuation = DecodeContinuation.decode(
        continuation_from_req(source, "transfer-1", state_builder).encode()
    )
    return req_from_continuation(
        continuation,
        make_allocation(),
        req_to_token_pool=ReqPool(),
        state_restorer=lambda req, data, _resume: setattr(req, "tokenizer", None),
    )


def test_continuation_preserves_the_hidden_state_mode() -> None:
    for mode, capture in (
        (False, CaptureHiddenMode.NULL),
        (True, CaptureHiddenMode.FULL),
        ("last", CaptureHiddenMode.LAST),
    ):
        req = rebuilt_req(prefill_req(return_hidden_states=mode))
        assert req.return_hidden_states == mode
        assert req.return_hidden_states_mode is capture


def test_continuation_strips_the_live_req_out_of_custom_params() -> None:
    source = prefill_req(custom_params={"segment_timestamps": True})
    source.sampling_params.custom_params["__req__"] = source
    req = rebuilt_req(source)
    assert req.sampling_params.custom_params["segment_timestamps"] is True
    assert "__req__" not in req.sampling_params.custom_params
    assert source.sampling_params.custom_params["__req__"] is source


def test_decode_receiver_commits_directly_to_admission_queue() -> None:
    admissions = queue.SimpleQueue()
    receiver = make_receiver(admissions)
    message = make_message()

    destination = receiver.reserve(message)
    receiver.commit(message, destination)

    admission = admissions.get_nowait()
    assert admission.continuation.request_id == "request-1"
    assert admission.allocation.page_indices == destination.page_indices


def test_prefill_defers_first_token_stop_policy_to_decode() -> None:
    req = prefill_req(max_new_tokens=1)
    del req.output_ids[:]
    original_max = req.sampling_params.max_new_tokens

    with defer_first_token_finish([req]):
        req.output_ids.append(2)
        req.update_finish_state()
        assert not req.finished()

    assert req.sampling_params.max_new_tokens == original_max
    req.update_finish_state()
    assert req.finished()


@pytest.mark.parametrize("priority", [None, -7, 0, 7])
def test_continuation_preserves_request_priority(priority: int | None) -> None:
    request = prefill_req()
    request.priority = priority
    assert rebuilt_req(request).priority == priority


@pytest.mark.parametrize("priority", [True, 1.5, "7"])
def test_continuation_rejects_invalid_priority(priority: object) -> None:
    request = prefill_req()
    request.priority = priority
    with pytest.raises(ValueError, match="priority"):
        continuation_from_req(request, "transfer", state_builder)


def test_pd_rejects_retained_session_metadata() -> None:
    request = prefill_req()
    request.omni_data.stage_payload.request.metadata["omni_session"] = SessionOperation(
        operation="open",
        session_identity=SessionIdentity("session"),
        stages=("thinker",),
    ).to_dict()
    with pytest.raises(NotImplementedError, match="session"):
        continuation_from_req(request, "transfer", state_builder)


def test_continuation_rejects_native_session_without_metadata() -> None:
    request = prefill_req()
    request.session = Mock()
    with pytest.raises(NotImplementedError, match="retained streaming-session"):
        continuation_from_req(request, "transfer-1", state_builder)


@pytest.mark.parametrize(
    "rollback_parameter",
    ["_asr_streaming_rollback_tokens", "_asr_streaming_rollback_chars"],
)
def test_continuation_allows_asr_refresh_without_retained_kv(
    rollback_parameter: str,
) -> None:
    request = prefill_req()
    parameters = request.omni_data.stage_payload.request.params
    parameters.update(
        {
            "_asr_streaming": True,
            "_asr_streaming_prefix_text": "Previously recognized text",
            rollback_parameter: 5,
        }
    )
    restored = rebuilt_req(request)
    assert restored.session is None
    assert restored.omni_data.stage_payload.request.params == parameters


@pytest.mark.parametrize("finish", ["commit", "abort", "invalid_destination"])
def test_hybrid_reservation_owns_token_pages_and_state_until_terminal(finish):
    token_allocator = KVAllocator()
    state_allocator = KVAllocator()
    state_allocator.next_slot = 1
    admissions = queue.SimpleQueue()
    receiver = DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=token_allocator,
        state_allocator=state_allocator,
        state_buffer_names=("linear_state.0", "linear_state.1"),
        admissions=admissions,
        resume_schema="test-v1",
    )
    message = replace(
        make_message(),
        source_buffer_page_indices={"linear_state.0": (4,), "linear_state.1": (4,)},
    )
    destination = receiver.reserve(message)
    assert destination.page_indices == (7, 8, 9)
    assert destination.buffer_page_indices == {
        "linear_state.0": (1,),
        "linear_state.1": (1,),
    }
    assert admissions.empty()
    assert not token_allocator.freed and not state_allocator.freed
    if finish == "commit":
        receiver.commit(message, destination)
        admission = admissions.get_nowait()
        assert admission.allocation.state_slots.tolist() == [1]
        assert not token_allocator.freed and not state_allocator.freed
        receiver.release_allocation(admission.allocation)
    elif finish == "abort":
        receiver.abort(message, destination, RuntimeError("copy failed"))
    else:
        with pytest.raises(RuntimeError, match="live reservation"):
            receiver.commit(message, replace(destination, buffer_page_indices={}))
    receiver.abort(message, destination, RuntimeError("duplicate cleanup"))
    assert len(token_allocator.freed) == len(state_allocator.freed) == 1
    assert token_allocator.freed[0].tolist() == [7, 8, 9]
    assert state_allocator.freed[0].tolist() == [1]
    assert not receiver.has_reservations()


def test_hybrid_state_exhaustion_rolls_back_token_reservation():
    token_allocator = KVAllocator()
    state_allocator = Mock()
    state_allocator.alloc.return_value = None
    receiver = DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=token_allocator,
        state_allocator=state_allocator,
        state_buffer_names=("linear_state.0",),
        admissions=queue.SimpleQueue(),
        resume_schema="test-v1",
    )
    message = replace(
        make_message(), source_buffer_page_indices={"linear_state.0": (4,)}
    )
    with pytest.raises(RuntimeError, match="linear-state pool exhausted"):
        receiver.reserve(message)
    assert token_allocator.freed[0].tolist() == [7, 8, 9]
    assert not receiver.has_reservations()
    state_allocator.free.assert_not_called()


def test_restored_request_adopts_upstream_state_slot_without_clear_or_reallocation():
    pool = HybridReqToTokenPool(
        size=2,
        mamba_size=4,
        mamba_spec_state_size=2,
        max_context_len=32,
        device="cpu",
        enable_memory_saver=False,
        mamba_layer_ids=[0],
        enable_mamba_extra_buffer=False,
        enable_overlap_schedule=False,
        cache_params=Mamba2CacheParams(
            layers=[0],
            shape=Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4,
                n_groups=1,
                num_heads=2,
                head_dim=2,
                state_size=2,
                conv_kernel=3,
            ),
        ),
    )
    initial_capacity = pool.mamba_allocator.available_size()
    state_slots = pool.mamba_allocator.alloc(1)
    assert state_slots is not None
    request = req_from_continuation(
        make_continuation(),
        replace(make_allocation(), state_slots=state_slots),
        req_to_token_pool=pool,
        state_restorer=lambda req, data, resume: setattr(req, "tokenizer", None),
    )
    assert request.kv.mamba_pool_idx.item() == state_slots[0].item()
    assert not request.kv.mamba_needs_clear
    assert pool.mamba_allocator.available_size() == initial_capacity - 1
    assert (
        pool.get_mamba_indices(torch.tensor([request.kv.req_pool_idx])).item()
        == state_slots[0].item()
    )
    pool.alloc([request])
    assert pool.mamba_allocator.available_size() == initial_capacity - 1
    assert not request.kv.mamba_needs_clear
    pool.free_mamba_cache(request)
    pool.free(request)
    assert pool.mamba_allocator.available_size() == initial_capacity
    assert pool.available_size() == 2
