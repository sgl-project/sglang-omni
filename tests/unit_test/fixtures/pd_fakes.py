# SPDX-License-Identifier: Apache-2.0
"""Shared deterministic test fakes and builders for PD unit tests."""

from __future__ import annotations

import contextlib
import queue
import threading
from array import array
from collections import deque
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto.kv_transfer import (
    KVBufferSpec,
    KVPoolLayout,
    KVTransferPrepareMessage,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.pd_receiver import DecodeContinuation, DecodeKVReceiver
from sglang_omni.scheduling.pd_scheduler import (
    OmniDecodeScheduler,
    OmniPrefillScheduler,
)
from sglang_omni.scheduling.pd_utils import (
    continuation_from_req,
    serialize_kv_allocator,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


class FakeRequestPool:
    """In-memory request-to-token row table matching ReqToTokenPool's contract."""

    def __init__(self, capacity: int = 4, max_context_len: int = 128) -> None:
        self.capacity = capacity
        self.max_context_len = max_context_len
        self.req_to_token = torch.zeros((capacity, max_context_len), dtype=torch.int64)
        self.active: dict[int, Req] = {}

    @property
    def size(self) -> int:
        return self.capacity

    def available_size(self) -> int:
        return max(self.capacity - len(self.active), 0)

    def alloc(self, requests: list[Req]) -> list[int] | None:
        if len(self.active) + len(requests) > self.capacity:
            return None
        indices: list[int] = []
        for request in requests:
            index = next(i for i in range(self.capacity) if i not in self.active)
            request.kv.req_pool_idx = index
            self.active[index] = request
            indices.append(index)
        return indices

    def write(self, row_slice: tuple[int, slice], value: torch.Tensor) -> None:
        self.req_to_token[row_slice] = value

    def free(self, request: Req) -> None:
        assert request.kv.req_pool_idx is not None
        self.active.pop(request.kv.req_pool_idx, None)
        request.kv.req_pool_idx = None


class FakeKVAllocator(BaseTokenToKVPoolAllocator):
    """In-memory paged/token KV allocator subclassing BaseTokenToKVPoolAllocator."""

    def __init__(
        self,
        *,
        size: int = 1024,
        page_size: int = 1,
        device: str = "cpu",
        kvcache: KVCache | None = None,
        starting_slot: int = 0,
    ) -> None:
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=torch.bfloat16,
            device=device,
            kvcache=kvcache,
            need_sort=False,
        )
        self.kvcache = kvcache
        self.starting_slot = starting_slot
        self.allocation_calls = 0
        self.freed: list[torch.Tensor] = []
        self.clear()

    def clear(self) -> None:
        self.free_slot_heads: list[int] = list(
            range(self.starting_slot, self.starting_slot + self.size, self.page_size)
        )
        self.free_group = None
        self.release_pages = torch.empty((0,), dtype=torch.int64)

    def get_kvcache(self) -> KVCache | None:
        return self.kvcache

    def available_size(self) -> int:
        return len(self.free_slot_heads) * self.page_size

    def get_all_free_pages(self) -> torch.Tensor:
        return torch.tensor(self.free_slot_heads, dtype=torch.int64)

    def alloc(self, need_size: int) -> torch.Tensor | None:
        self.allocation_calls += 1
        page_count = need_size // self.page_size
        if page_count > len(self.free_slot_heads):
            return None
        chosen = self.free_slot_heads[:page_count]
        self.free_slot_heads = self.free_slot_heads[page_count:]
        slots = [head + offset for head in chosen for offset in range(self.page_size)]
        return torch.tensor(slots, dtype=torch.int64)

    def free(self, free_index: torch.Tensor) -> None:
        if self.free_group is not None:
            self.free_group.append(free_index.clone())
            return
        self.freed.append(free_index)
        heads = sorted({int(slot) for slot in free_index[:: self.page_size].tolist()})
        self.free_slot_heads.extend(heads)


class FakeTreeNode:
    """Typed stand-in for a prefix cache tree node."""


class FakeTreeCache:
    disable = True
    buffer_pipeline = None

    def supports_mamba(self) -> bool:
        return False

    def supports_prefix_sharing(self) -> bool:
        return False

    def evictable_size(self) -> int:
        return 0

    def lock(self, node: FakeTreeNode | None) -> FakeTreeNode | None:
        return node

    def unlock(self, lock: FakeTreeNode | None) -> None:
        return None

    def finish_storage_prefetch_admission(
        self,
        handle: FakeTreeNode | None,
        *,
        fulfilled_tokens: int,
        reason: str | None,
    ) -> None:
        return None


def build_prefill_state(
    request: Req,
) -> tuple[dict[str, object], dict[str, object] | None, list[int]]:
    return (
        request.omni_data.stage_payload.to_dict(),
        None,
        list(request.origin_input_ids),
    )


def make_prefill_request(
    request_id: str = "request-1",
    prompt_tokens: int = 3,
    *,
    max_new_tokens: int = 16,
    with_first_output: bool = True,
) -> Req:
    sampling = SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0)
    sampling.normalize(None)
    prompt_ids = list(range(10, 10 + prompt_tokens))
    request = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=array("q", prompt_ids),
        sampling_params=sampling,
        vocab_size=256,
    )
    if with_first_output:
        request.output_ids.append(42)
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None, params={}),
        data={},
    )
    request.omni_data = SGLangARRequestData(
        input_ids=torch.tensor(prompt_ids, dtype=torch.long),
        output_ids=request.output_ids,
        req=request,
        stage_payload=payload,
    )
    request.refresh_fill_ids()
    return request


def make_prepare_message(
    *,
    request_id: str,
    transfer_id: str,
    prompt_tokens: int,
    page_size: int = 1,
) -> KVTransferPrepareMessage:
    request = make_prefill_request(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        with_first_output=True,
    )
    continuation: DecodeContinuation = continuation_from_req(
        request, transfer_id, build_prefill_state
    )
    page_count = (prompt_tokens + page_size - 1) // page_size
    return KVTransferPrepareMessage(
        request_id=request_id,
        transfer_id=transfer_id,
        from_stage="prefill",
        to_stage="decode",
        source_pool_id="prefill:kv",
        target_pool_id="decode:kv",
        source_page_indices=tuple(range(page_count)),
        source_layout=KVPoolLayout("test", page_size, (KVBufferSpec("kv", 4),)),
        metadata={"decode_continuation": continuation.encode()},
    )


def make_prefill_scheduler_stub(
    *,
    max_in_flight_handoffs: int = 4,
    partner_stage: str | None = "decode",
    page_size: int = 16,
    device: str = "cpu",
    request_finished_callback: Mock | None = None,
) -> OmniPrefillScheduler:
    scheduler = object.__new__(OmniPrefillScheduler)
    scheduler.pd_stage_name = "prefill"
    scheduler.pd_partner_stage = partner_stage
    scheduler.pd_pool_id = "prefill:kv"
    scheduler.max_in_flight_handoffs = max_in_flight_handoffs
    scheduler.page_size = page_size
    scheduler.device = device
    scheduler.pd_due_releases = queue.SimpleQueue()
    scheduler.pd_outstanding_releases = set()
    scheduler.pd_wakeup_lock = threading.Lock()
    scheduler.pd_wakeup_pending = False
    scheduler.chunked_req = None
    scheduler.running_batch = SimpleNamespace(
        reqs=[],
        is_empty=lambda: True,
        batch_is_full=False,
        filter_batch=lambda: None,
    )
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler.previous_pending_decode = None
    scheduler.async_pending = None
    scheduler.result_queue = deque()
    scheduler.decode_batch_states = {}
    scheduler.waiting_queue = []
    scheduler.pending_request_builds = {}
    scheduler.pending_request_admissions = {}
    scheduler.backlogged_request_build_payloads = deque()
    scheduler.early_request_builds = False
    scheduler.early_request_build_futures = {}
    scheduler.early_request_build_declined_ids = set()
    scheduler.deferred_request_payloads = {}
    scheduler.dirty_deferred_request_ids = set()
    scheduler.session_bridge = None
    scheduler.enable_overlap = False
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.tp_size = 1
    scheduler.req_to_token_pool = ReqToTokenPool(
        size=8, max_context_len=128, device="cpu", enable_memory_saver=False
    )
    scheduler.inbox = queue.Queue()
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
    scheduler.aborted_request_id_order = deque()
    scheduler.release_request_kv_cache = Mock()
    scheduler.pd_state_builder = build_prefill_state
    return scheduler


def make_decode_scheduler_stub(
    *,
    allocator: FakeKVAllocator | None = None,
    row_capacity: int = 2,
    decode_runway_tokens: int = 0,
) -> OmniDecodeScheduler:
    scheduler = object.__new__(OmniDecodeScheduler)
    scheduler.pd_admissions = deque()
    scheduler.pd_due_releases = queue.SimpleQueue()
    scheduler.pd_outstanding_releases = set()
    scheduler.pd_wakeup_lock = threading.Lock()
    scheduler.pd_wakeup_pending = False
    scheduler.pd_lifecycle_lock = threading.RLock()
    scheduler.pd_state_restorer = lambda restored, data, resume: None
    scheduler.admin_lock = threading.RLock()
    scheduler.request_admission_lock = threading.RLock()
    scheduler.aborted_request_ids = set()
    scheduler.aborted_request_id_order = deque()
    scheduler.completed_request_ids = {}
    scheduler.pending_request_builds = {}
    scheduler.pending_request_admissions = {}
    scheduler.backlogged_request_build_payloads = deque()
    scheduler.early_request_builds = False
    scheduler.early_request_build_futures = {}
    scheduler.deferred_request_payloads = {}
    scheduler.session_bridge = None
    scheduler.draft_worker = None
    scheduler.enable_overlap = False
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.tp_size = 1
    scheduler.is_entry_rank = True
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler.previous_pending_decode = None
    scheduler.async_pending = None
    scheduler.result_queue = deque()
    scheduler.last_pause_mode = None
    scheduler._engine_paused = False  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    scheduler.req_to_token_pool = FakeRequestPool(capacity=row_capacity)
    scheduler.token_to_kv_pool_allocator = (
        allocator if allocator is not None else FakeKVAllocator(size=64, page_size=1)
    )
    serialize_kv_allocator(
        scheduler.token_to_kv_pool_allocator, lock=scheduler.pd_lifecycle_lock
    )
    scheduler.waiting_queue = []
    scheduler.running_batch = SimpleNamespace(
        reqs=[],
        is_empty=lambda: True,
        filter_batch=lambda: None,
        batch_is_full=False,
    )
    scheduler.inbox = queue.Queue()
    scheduler.outbox = queue.Queue()
    scheduler.abort_callback = Mock()
    scheduler.resolve_pending_async = lambda: None
    scheduler.advance_prompt_cache_epoch = lambda: None
    scheduler.pd_receiver = DecodeKVReceiver(
        pool_id="decode:kv",
        allocator=scheduler.token_to_kv_pool_allocator,
        req_to_token_pool=scheduler.req_to_token_pool,
        admissions=scheduler.pd_admissions,
        resume_schema="test-v1",
        lifecycle_lock=scheduler.pd_lifecycle_lock,
        decode_runway_tokens=decode_runway_tokens,
        wakeup_notifier=scheduler,
        capacity_check=scheduler,
    )
    return scheduler


class FakeCapacityCheck:
    def __init__(self, active: bool = False) -> None:
        self.active = active

    def has_active_decode_requests(self) -> bool:
        return self.active


class StreamTrackingAllocator(FakeKVAllocator):
    def __init__(
        self, stream_stack: list[str], *, size: int = 64, device: str = "cuda:0"
    ) -> None:
        super().__init__(size=size, page_size=1, device=device)
        self.stream_stack = stream_stack
        self.observed: list[tuple[str, str | None]] = []

    def alloc(self, need_size: int) -> torch.Tensor | None:
        self.observed.append(
            ("alloc", self.stream_stack[-1] if self.stream_stack else None)
        )
        return super().alloc(need_size)

    def free(self, free_index: torch.Tensor) -> None:
        self.observed.append(
            ("free", self.stream_stack[-1] if self.stream_stack else None)
        )
        super().free(free_index)


class FakeCudaEvent:
    def __init__(
        self,
        env: FakeCudaEnv,
        *,
        interprocess: bool = False,
    ) -> None:
        self.env = env
        self.interprocess = interprocess
        self.recorded_streams: list[SimpleNamespace] = []
        env.event_devices.append(env.current_device)
        env.created_events.append(self)

    def record(self, stream: SimpleNamespace) -> None:
        self.recorded_streams.append(stream)
        self.env.events_log.append(("record", stream.name))


class FakeCudaEnv:
    """Shared monkeypatchable CUDA device/stream/event recorder for CPU tests."""

    def __init__(self) -> None:
        self.current_device = "cuda:0"
        self.entered_devices: list[str] = []
        self.event_devices: list[str] = []
        self.created_events: list[FakeCudaEvent] = []
        self.events_log: list[tuple[str, str]] = []
        self.stream_stack: list[str] = []
        self.compute_stream = SimpleNamespace(
            name="compute",
            wait_event=lambda ev: self.events_log.append(("wait", "compute")),
        )
        self.alloc_stream = SimpleNamespace(
            name="alloc",
            wait_event=lambda ev: self.events_log.append(("wait", "alloc")),
        )
        self.active_stream: SimpleNamespace = self.compute_stream

    @contextlib.contextmanager
    def device_context(self, device: str) -> Iterator[None]:
        previous, self.current_device = self.current_device, device
        self.entered_devices.append(device)
        try:
            yield
        finally:
            self.current_device = previous

    @contextlib.contextmanager
    def stream_context(self, stream: SimpleNamespace) -> Iterator[None]:
        previous = self.active_stream
        self.active_stream = stream
        self.stream_stack.append("allocation_stream")
        try:
            yield
        finally:
            self.stream_stack.pop()
            self.active_stream = previous


def install_fake_cuda(monkeypatch: pytest.MonkeyPatch) -> FakeCudaEnv:
    env = FakeCudaEnv()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", env.device_context)
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda *, interprocess=False: FakeCudaEvent(env, interprocess=interprocess),
    )
    monkeypatch.setattr(
        torch.cuda,
        "Stream",
        lambda device=None: SimpleNamespace(
            name="alloc",
            device=device,
            wait_event=lambda ev: env.events_log.append(("wait", "alloc")),
        ),
    )
    monkeypatch.setattr(torch.cuda, "stream", env.stream_context)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: env.active_stream
    )
    return env


def make_paged_allocator(
    pages: int, page_size: int = 16
) -> PagedTokenToKVPoolAllocator:
    return PagedTokenToKVPoolAllocator(
        size=page_size * pages,
        page_size=page_size,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )


class FakeCudaSlots:
    """Minimal CUDA-like slot tensor double for verifying record_stream order on CPU."""

    is_cuda = True
    device = torch.device("cpu")

    def __init__(
        self, values: torch.Tensor, expected_stream: SimpleNamespace, log: list[str]
    ) -> None:
        self.values, self.expected_stream, self.log = values, expected_stream, log

    def record_stream(self, stream: SimpleNamespace) -> None:
        assert stream is self.expected_stream
        self.log.append("record_stream")

    def __getitem__(self, token_slice: slice) -> torch.Tensor:
        return self.values[token_slice]
