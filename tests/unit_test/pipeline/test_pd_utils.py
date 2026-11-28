# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from array import array
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import CaptureHiddenMode, Req
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto.kv_transfer import (
    KVBufferSpec,
    KVPoolLayout,
    KVTransferPrepareMessage,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, SessionOperation
from sglang_omni.scheduling.pd_receiver import (
    DecodeAdmission,
    DecodeContinuation,
    DecodeKVReceiver,
    ReservedKV,
)
from sglang_omni.scheduling.pd_scheduler import (
    OmniDecodeScheduler,
    OmniPrefillScheduler,
    validate_pd_runtime,
)
from sglang_omni.scheduling.pd_utils import (
    build_kv_pool,
    continuation_from_req,
    defer_first_token_finish,
    request_from_continuation,
    request_page_indices,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from tests.unit_test.fixtures.pd_fakes import (
    FakeKVAllocator,
    FakeRequestPool,
    make_prefill_request,
)


class ReqPool(FakeRequestPool):
    """Request pool test double kept for existing PD unit tests."""

    def __init__(self, capacity: int = 4, max_context_len: int = 32) -> None:
        super().__init__(capacity=capacity, max_context_len=max_context_len)


class KVAllocator(FakeKVAllocator):
    """Paged/token KV allocator test double starting at slot 7."""

    def __init__(
        self,
        *,
        size: int = 1024,
        page_size: int = 1,
        device: str = "cpu",
        kvcache: KVCache | None = None,
    ) -> None:
        super().__init__(
            size=size,
            page_size=page_size,
            device=device,
            kvcache=kvcache,
            starting_slot=7,
        )

    @property
    def next_slot(self) -> int:
        return self.free_slot_heads[0]

    @next_slot.setter
    def next_slot(self, value: int) -> None:
        self.starting_slot = value
        self.free_slot_heads = list(range(value, value + self.size, self.page_size))


def prefill_req(
    *,
    max_new_tokens: int = 16,
    custom_params: dict[str, object] | None = None,
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


def state_builder(
    request: Req,
) -> tuple[dict[str, object], dict[str, object] | None, list[int]]:
    return (
        request.omni_data.stage_payload.to_dict(),
        None,
        list(request.origin_input_ids),
    )


def make_continuation() -> DecodeContinuation:
    return continuation_from_req(prefill_req(), "transfer-1", state_builder)


def make_allocation() -> ReservedKV:
    slots = torch.tensor([7, 8, 9], dtype=torch.int64)
    return ReservedKV(slots=slots, page_indices=(7, 8, 9), sequence_length=3)


def make_message(
    *, transfer_id: str = "transfer-1", **metadata: object
) -> KVTransferPrepareMessage:
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


def make_receiver(
    admissions: deque[DecodeAdmission] | None = None,
    req_to_token_pool: ReqPool | None = None,
) -> DecodeKVReceiver:
    return DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=KVAllocator(),
        req_to_token_pool=(
            req_to_token_pool if req_to_token_pool is not None else ReqPool()
        ),
        admissions=admissions if admissions is not None else deque(),
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
    req = request_from_continuation(
        continuation,
        make_allocation(),
        req_to_token_pool=req_to_token_pool,
        state_restorer=lambda request, data, resume: setattr(
            request, "tokenizer", None
        ),
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


def rebuilt_req(source: Req) -> Req:
    continuation = DecodeContinuation.decode(
        continuation_from_req(source, "transfer-1", state_builder).encode()
    )
    return request_from_continuation(
        continuation,
        make_allocation(),
        req_to_token_pool=ReqPool(),
        state_restorer=lambda request, data, resume: setattr(
            request, "tokenizer", None
        ),
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
    admissions: deque[DecodeAdmission] = deque()
    receiver = make_receiver(admissions)
    message = make_message()

    destination = asyncio.run(receiver.reserve(message, timeout_seconds=5.0))
    receiver.commit(message, destination)

    admission = admissions.popleft()
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


class CustomMHAKVPool(MHATokenToKVPool):
    """Distinct MHA subclass that must not inherit SeparateKVAllocationsPool aliasing."""


@pytest.mark.parametrize(
    ("pool_kind", "page_size"),
    [("mha", 1), ("mha", 16), ("mla", 1), ("mla", 16)],
)
def test_build_kv_pool_supports_mha_mla_and_validates_pointers(
    pool_kind: str, page_size: int
) -> None:
    if pool_kind == "mha":
        kv_cache: KVCache = MHATokenToKVPool(
            size=64,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=16,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        expected_buffers = 4
        expected_bytes_per_page = page_size * 2 * 16 * 2
    else:
        kv_cache = MLATokenToKVPool(
            size=64,
            page_size=page_size,
            dtype=torch.bfloat16,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        expected_buffers = 2
        expected_bytes_per_page = page_size * 24 * 2

    pool = build_kv_pool(kv_cache, pool_id="stage:kv")
    assert pool.page_size == page_size
    assert len(pool.buffers) == expected_buffers
    assert all(
        region.bytes_per_page == expected_bytes_per_page for region in pool.buffers
    )

    if pool_kind == "mha" and page_size == 16:
        custom_cache = CustomMHAKVPool(
            size=64,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=16,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        custom_pool = build_kv_pool(custom_cache, pool_id="stage:kv")
        assert custom_pool.layout_id != pool.layout_id

        pointers, offsets, sizes = kv_cache.get_contiguous_buf_infos()
        kv_cache.get_contiguous_buf_infos = lambda: (
            [pointers[0] + 64, *pointers[1:]],
            offsets,
            sizes,
        )
        with pytest.raises(ValueError, match="do not match contiguous buffer pointers"):
            build_kv_pool(kv_cache, pool_id="stage:kv")


@pytest.mark.parametrize("sequence_length", [1, 16, 32, 35])
def test_paged_request_indices_and_rebuilt_request_slice_and_free(
    sequence_length: int,
) -> None:
    page_size = 16
    allocator = PagedTokenToKVPoolAllocator(
        size=page_size * 8,
        page_size=page_size,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    req_to_token_pool = ReqToTokenPool(
        size=4, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    receiver = DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=allocator,
        req_to_token_pool=req_to_token_pool,
        admissions=deque(),
        resume_schema="test-v1",
        page_size=page_size,
    )

    request = make_prefill_request(
        f"req-{sequence_length}", prompt_tokens=sequence_length, with_first_output=True
    )
    continuation = continuation_from_req(request, "tx-paged", state_builder)
    expected_pages = (sequence_length + page_size - 1) // page_size
    slots = allocator.alloc(expected_pages * page_size)
    assert slots is not None
    receiver.reserved_request_rows = 1
    allocation = ReservedKV(
        slots=slots,
        page_indices=tuple(int(s) // page_size for s in slots[::page_size].tolist()),
        sequence_length=sequence_length,
        row_credit_owner=receiver,
        row_credit_held=True,
    )

    rebuilt = request_from_continuation(
        continuation,
        allocation,
        req_to_token_pool=req_to_token_pool,
        state_restorer=lambda restored, data, resume: setattr(
            restored, "tokenizer", None
        ),
    )
    assert receiver.reserved_request_rows == 0
    assert allocation.row_credit_held is False
    assert rebuilt.kv.req_pool_idx is not None
    written = req_to_token_pool.req_to_token[rebuilt.kv.req_pool_idx, :sequence_length]
    assert written.tolist() == slots[:sequence_length].tolist()
    assert (
        request_page_indices(req_to_token_pool, rebuilt, page_size=page_size)
        == allocation.page_indices
    )

    allocator.free_segment(written, start_pos=0)
    req_to_token_pool.free(rebuilt)
    assert allocator.available_size() == page_size * 8

    # Unaligned page head must be rejected by request_page_indices.
    if sequence_length == 35:
        req_to_token_pool.alloc([request])
        req_to_token_pool.req_to_token[request.kv.req_pool_idx, 0] = 3
        with pytest.raises(RuntimeError, match="unaligned page head"):
            request_page_indices(req_to_token_pool, request, page_size=page_size)


@pytest.mark.parametrize("scheduler_type", [OmniPrefillScheduler, OmniDecodeScheduler])
def test_validate_pd_runtime_allows_paged_kv_and_rejects_unsupported_modes(
    scheduler_type: type[OmniPrefillScheduler] | type[OmniDecodeScheduler],
) -> None:
    def make_runtime(
        *,
        page_size: int = 16,
        async_prefill: bool = False,
        post_capture_active: bool = False,
    ) -> OmniPrefillScheduler | OmniDecodeScheduler:
        scheduler = object.__new__(scheduler_type)
        scheduler.session_adapter = None
        scheduler.req_to_token_pool = ReqPool()
        scheduler.token_to_kv_pool_allocator = KVAllocator(
            page_size=max(page_size, 1),
            kvcache=SimpleNamespace(post_capture_active=post_capture_active),
        )
        scheduler.server_args = SimpleNamespace(disable_radix_cache=True)
        scheduler.tp_size = 1
        scheduler.page_size = page_size
        scheduler.async_prefill = async_prefill
        scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        return scheduler

    for page_size in (1, 16, 64):
        validate_pd_runtime(make_runtime(page_size=page_size))

    with pytest.raises(ValueError, match="page_size"):
        validate_pd_runtime(make_runtime(page_size=0))
    with pytest.raises(NotImplementedError, match="async_prefill"):
        validate_pd_runtime(make_runtime(async_prefill=True))
    with pytest.raises(NotImplementedError, match="post-capture"):
        validate_pd_runtime(make_runtime(post_capture_active=True))
