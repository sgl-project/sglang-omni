# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from array import array
from collections import deque
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as upstream_scheduler
from sglang.srt.managers.schedule_batch import (
    NextBatchPlan,
    Req,
    ReqKvInfo,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_schedule
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import SessionController
from sglang.srt.session.streaming_session import SessionSlot
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.admission import SessionCapacityExhaustedError
from sglang_omni.model_runner.base import PendingStep
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, SessionOperation, TimedChunk
from sglang_omni.scheduling import omni_scheduler
from sglang_omni.scheduling.omni_scheduler import (
    OmniScheduler,
    PendingDecode,
    SessionRetryPrefillAdder,
)
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionBridge,
    BridgeSession,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import ModelRunnerOutput, SchedulerOutput

CONTEXT_LENGTH = 64
RETAINED_TOKEN_COUNT = 8
UNIT_TOKEN_IDS = list(range(1, RETAINED_TOKEN_COUNT + 4))


class KVAllocator:
    device = "cpu"
    page_size = 1
    available_tokens = 1 << 20

    def free(self, free_index: torch.Tensor) -> None:
        pass

    def available_size(self) -> int:
        return self.available_tokens


class AdmissionScheduler:
    enable_plain_fallback = False
    session_prefill_row_cache = None
    get_num_allocatable_reqs = OmniScheduler.get_num_allocatable_reqs

    def __init__(
        self,
        *,
        request_rows: int,
        continuation_aware_admission: bool = True,
        kv_pool_tokens: int | None = None,
        device: str = "cpu",
        page_size: int = 1,
    ) -> None:
        self.continuation_aware_admission = continuation_aware_admission
        self.req_to_token_pool = ReqToTokenPool(
            size=request_rows,
            max_context_len=CONTEXT_LENGTH,
            device=device,
            enable_memory_saver=False,
        )
        if kv_pool_tokens is None:
            self.token_to_kv_pool_allocator = KVAllocator()
        else:
            kv_cache = MHATokenToKVPool(
                size=kv_pool_tokens,
                page_size=page_size,
                dtype=torch.float16,
                head_num=1,
                head_dim=8,
                layer_num=1,
                device=device,
                enable_memory_saver=False,
            )
            if page_size == 1:
                self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                    size=kv_pool_tokens,
                    dtype=torch.float16,
                    device=device,
                    kvcache=kv_cache,
                    need_sort=False,
                )
            else:
                self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                    size=kv_pool_tokens,
                    page_size=page_size,
                    dtype=torch.float16,
                    device=device,
                    kvcache=kv_cache,
                    need_sort=False,
                )
        self.tree_cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=page_size,
                enable_kv_cache_events=False,
                eviction_policy="lru",
                tree_components=(ComponentType.FULL,),
            )
        )
        self.session_controller = SessionController(self.tree_cache)
        self.max_running_requests = request_rows
        self.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
        self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        self.waiting_queue: list[Req] = []
        self.session_bridge = ARSessionBridge(self, adapter=None)

    def open_session(self, session_id: str, *, holds_row: bool) -> None:
        assert self.session_bridge.open_streaming_session(session_id)
        self.session_bridge.sessions[session_id] = BridgeSession(
            session_identity=SessionIdentity(session_id)
        )
        if holds_row:
            (row,) = self.req_to_token_pool.alloc_rows(1)
            if isinstance(
                self.token_to_kv_pool_allocator,
                (TokenToKVPoolAllocator, PagedTokenToKVPoolAllocator),
            ):
                retained_positions = self.token_to_kv_pool_allocator.alloc(
                    RETAINED_TOKEN_COUNT
                )
                assert retained_positions is not None
                self.req_to_token_pool.write(
                    (row, slice(0, RETAINED_TOKEN_COUNT)), retained_positions
                )
            else:
                pass
            self.tree_cache.session.slots[session_id] = SessionSlot(
                kv=ReqKvInfo(
                    req_pool_idx=row,
                    kv_committed_len=RETAINED_TOKEN_COUNT,
                    kv_allocated_len=RETAINED_TOKEN_COUNT,
                )
            )
        else:
            pass

    def queue_request(self, request_id: str, session_id: str | None = None) -> Req:
        request = Req(
            rid=request_id,
            origin_input_text="",
            origin_input_ids=UNIT_TOKEN_IDS,
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        if session_id is not None:
            request.session = self.session_controller.get(session_id)
        else:
            pass
        self.waiting_queue.append(request)
        return request

    def admit_prefill_batch(self) -> list[str]:
        admitted_requests: list[Req] = []
        for request in self.waiting_queue:
            if len(admitted_requests) >= self.get_num_allocatable_reqs(
                0,
                None,
                running_batch=self.running_batch,
                num_held_rows=sum(
                    admitted.kv.holds_kv for admitted in admitted_requests
                ),
            ):
                break
            else:
                self.tree_cache.match_prefix(
                    MatchPrefixParams(
                        key=RadixKey(array("q", request.origin_input_ids[:-1])),
                        req=request,
                    )
                )
                admitted_requests.append(request)
        assert self.req_to_token_pool.alloc(admitted_requests) is not None
        return [request.rid for request in admitted_requests]

    def free_rows(self) -> int:
        return self.req_to_token_pool.available_size()

    def append_unit(
        self,
        request_id: str,
        session_id: str,
        *,
        is_enqueued: bool,
        prompt_tokens: int = 3,
        max_new_tokens: int = 1,
    ) -> Req:
        request = self.queue_request(request_id, session_id)
        request.origin_input_ids = array("q", range(prompt_tokens))
        request.sampling_params.max_new_tokens = max_new_tokens
        unit = self.session_bridge.accept(
            StagePayload(request_id, OmniRequest(None), {}),
            SessionOperation(
                "append",
                SessionIdentity(session_id),
                ("talker",),
                TimedChunk("text", 0, 0, 0, {}),
            ),
        )
        unit.session_request = request
        unit.is_enqueued = is_enqueued
        return request


@pytest.fixture(autouse=True)
def per_batch_limit(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    parallel = SimpleNamespace(pp_max_micro_batch_size=64)
    monkeypatch.setattr(omni_scheduler, "get_parallel", lambda: parallel)
    monkeypatch.setattr(upstream_scheduler, "get_parallel", lambda: parallel)
    return parallel


@pytest.fixture
def native_runtime_context() -> Iterator[None]:
    with get_context().override_server_args(page_size=1):
        yield


@pytest.mark.parametrize("continuation_aware_admission", [True, False])
def test_every_open_session_unit_is_admitted_in_one_batch_with_one_free_row(
    continuation_aware_admission: bool,
) -> None:
    session_count = 6
    scheduler = AdmissionScheduler(
        request_rows=session_count + 1,
        continuation_aware_admission=continuation_aware_admission,
    )
    held_rows: dict[str, int] = {}
    for index in range(session_count):
        scheduler.open_session(f"session-{index}", holds_row=True)
        held_rows[f"unit-{index}"] = scheduler.tree_cache.session.slots[
            f"session-{index}"
        ].kv.req_pool_idx
    units = [
        scheduler.queue_request(f"unit-{index}", f"session-{index}")
        for index in range(session_count)
    ]
    assert scheduler.free_rows() == 1

    assert scheduler.admit_prefill_batch() == [unit.rid for unit in units]
    assert {unit.rid: unit.kv.req_pool_idx for unit in units} == held_rows
    assert scheduler.free_rows() == 1


def test_continuation_aware_admission_admits_every_unit_with_no_free_row() -> None:
    session_count = 6
    scheduler = AdmissionScheduler(request_rows=session_count)
    held_rows: dict[str, int] = {}
    for index in range(session_count):
        scheduler.open_session(f"session-{index}", holds_row=True)
        held_rows[f"unit-{index}"] = scheduler.tree_cache.session.slots[
            f"session-{index}"
        ].kv.req_pool_idx
    units = [
        scheduler.queue_request(f"unit-{index}", f"session-{index}")
        for index in range(session_count)
    ]
    assert scheduler.free_rows() == 0

    assert scheduler.admit_prefill_batch() == [unit.rid for unit in units]
    assert {unit.rid: unit.kv.req_pool_idx for unit in units} == held_rows
    assert scheduler.free_rows() == 0


def test_without_continuation_aware_admission_no_free_row_admits_nothing() -> None:
    session_count = 6
    scheduler = AdmissionScheduler(
        request_rows=session_count, continuation_aware_admission=False
    )
    for index in range(session_count):
        scheduler.open_session(f"session-{index}", holds_row=True)
    units = [
        scheduler.queue_request(f"unit-{index}", f"session-{index}")
        for index in range(session_count)
    ]
    assert scheduler.free_rows() == 0

    assert scheduler.admit_prefill_batch() == []
    assert scheduler.waiting_queue == units
    assert scheduler.free_rows() == 0


def test_first_units_and_plain_requests_are_limited_by_free_rows() -> None:
    scheduler = AdmissionScheduler(request_rows=5)
    for session_id in ("held-a", "held-b", "held-c"):
        scheduler.open_session(session_id, holds_row=True)
    scheduler.open_session("new", holds_row=False)
    scheduler.queue_request("unit-a", "held-a")
    scheduler.queue_request("first-unit", "new")
    scheduler.queue_request("unit-b", "held-b")
    scheduler.queue_request("plain-0")
    scheduler.queue_request("plain-1")
    scheduler.queue_request("unit-c", "held-c")
    assert scheduler.free_rows() == 2

    assert scheduler.admit_prefill_batch() == [
        "unit-a",
        "first-unit",
        "unit-b",
        "plain-0",
    ]
    assert scheduler.free_rows() == 0


def native_prefill_scheduler(pool_state: AdmissionScheduler) -> OmniScheduler:
    scheduler = object.__new__(OmniScheduler)
    scheduler.__dict__.update(pool_state.__dict__)
    scheduler.session_bridge.bridge_scheduler = scheduler
    scheduler.is_retrying_session_prefill = False
    scheduler.prefill_coalesce_requests = 0
    scheduler.chunked_req = None
    scheduler.prefill_delayer = None
    scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
    scheduler.enable_priority_preemption = False
    scheduler.is_hybrid_swa = False
    scheduler.min_free_slots_delayer = None
    scheduler.policy = SchedulePolicy("fcfs", scheduler.tree_cache, False, False, False)
    scheduler.processed_tokens_counter = 0
    scheduler.chunked_prefill_size = None
    scheduler.dynamic_chunk_sizer = None
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(), prefill_aware_swa=False
        )
    )
    scheduler.page_size = scheduler.token_to_kv_pool_allocator.page_size
    scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.7)
    scheduler.max_prefill_tokens = 64
    scheduler.is_mixed_chunk = False
    scheduler.priority_scheduling_preemption_threshold = 0
    scheduler.max_prefill_bs = 0
    scheduler.dllm_config = None
    scheduler.enable_lora = False
    scheduler.enable_hicache_storage = False
    scheduler.enable_lmcache = False
    scheduler.hisparse_coordinator = None
    scheduler.truncation_align_size = None
    scheduler.model_config = SimpleNamespace(vocab_size=128, is_encoder_decoder=False)
    scheduler.enable_overlap = False
    scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_unified_cache_external_linker = False
    scheduler.enable_priority_scheduling = False
    scheduler.load_inquirer = SimpleNamespace(
        _get_num_pending_tokens=lambda **kwargs: 0
    )
    return scheduler


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("priority_scheduling", [False, True])
def test_native_prefill_bypasses_first_turn_when_only_retained_rows_remain(
    native_runtime_context: None, priority_scheduling: bool
) -> None:
    """A blocked first turn must not prevent a later session from reusing its row."""
    pool_state = AdmissionScheduler(request_rows=4, kv_pool_tokens=128, device="cuda")
    for session_id in ("held-a", "held-b"):
        pool_state.open_session(session_id, holds_row=True)
    pool_state.open_session("fresh", holds_row=False)
    fresh = pool_state.append_unit("fresh", "fresh", is_enqueued=True)
    assert pool_state.session_bridge.check_session_capacity("fresh") is None

    for index in range(2):
        (row,) = pool_state.req_to_token_pool.alloc_rows(1)
        positions = pool_state.token_to_kv_pool_allocator.alloc(1)
        assert positions is not None
        pool_state.req_to_token_pool.write((row, slice(0, 1)), positions)
        request = Req(
            rid=f"running-{index}",
            origin_input_text="",
            origin_input_ids=[1],
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        request.kv = ReqKvInfo(req_pool_idx=row, kv_committed_len=1, kv_allocated_len=1)
        pool_state.running_batch.reqs.append(request)

    first = pool_state.append_unit(
        "first", "held-a", is_enqueued=True, prompt_tokens=11
    )
    second = pool_state.append_unit(
        "second", "held-b", is_enqueued=True, prompt_tokens=11
    )
    for continuation in (first, second):
        continuation.sampling_params.normalize(None)
        assert (
            pool_state.session_bridge.check_session_capacity(continuation.rid) is None
        )
    assert pool_state.free_rows() == 0
    assert pool_state.waiting_queue == [fresh, first, second]

    scheduler = native_prefill_scheduler(pool_state)
    if priority_scheduling:
        fresh.priority, first.priority, second.priority = 0, 10, 1
        scheduler.enable_priority_scheduling = True
        scheduler.policy = SchedulePolicy(
            "fcfs", scheduler.tree_cache, False, True, True
        )
        expected = [second, first]
    else:
        expected = [first, second]
    plan = scheduler.get_new_batch_prefill(scheduler.running_batch)

    assert plan.batch_to_run is not None
    assert plan.batch_to_run.reqs == expected
    assert scheduler.waiting_queue == [fresh]
    assert (
        scheduler.session_bridge.units_by_request_id[fresh.rid].capacity_wait_started_at
        is not None
    )
    assert all(
        scheduler.session_bridge.units_by_request_id[req.rid].capacity_wait_started_at
        is None
        for req in (first, second)
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("continuation_aware_admission", [True, False])
def test_native_prefill_probes_single_untried_waiter_after_full_batch(
    native_runtime_context: None,
    continuation_aware_admission: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full batch must not strand an eligible session that was not tried."""
    with get_context().override_server_args(page_size=4):
        pool = AdmissionScheduler(
            request_rows=4,
            kv_pool_tokens=24,
            device="cuda",
            page_size=4,
            continuation_aware_admission=continuation_aware_admission,
        )
        (row,) = pool.req_to_token_pool.alloc_rows(1)
        positions = pool.token_to_kv_pool_allocator.alloc(4)
        assert positions is not None
        pool.req_to_token_pool.write((row, slice(0, 4)), positions)
        peer = Req(
            rid="running",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2, 3, 4]),
            sampling_params=SamplingParams(max_new_tokens=4),
        )
        peer.sampling_params.normalize(None)
        peer.kv = ReqKvInfo(req_pool_idx=row, kv_committed_len=4, kv_allocated_len=4)
        pool.running_batch.reqs.append(peer)
        for session_id in ("first", "second"):
            pool.open_session(session_id, holds_row=False)
        first = pool.append_unit(
            "first-req", "first", is_enqueued=True, prompt_tokens=5, max_new_tokens=4
        )
        second = pool.append_unit(
            "second-req", "second", is_enqueued=True, prompt_tokens=1, max_new_tokens=1
        )
        for request in (first, second):
            request.sampling_params.normalize(None)
        scheduler = native_prefill_scheduler(pool)
        scheduler.new_token_ratio_tracker.current = 1.0
        scheduler.session_admission_timeout_seconds = 30.0
        assert (
            scheduler.session_bridge.check_session_capacity(
                second.rid,
                running_request_ids=frozenset({peer.rid}),
                new_token_ratio=1.0,
            )
            is None
        )

        plan = scheduler.get_new_batch_prefill(scheduler.running_batch)
        assert plan.batch_to_run.reqs == [first]
        assert scheduler.waiting_queue == [second]
        assert plan.running_batch.batch_is_full
        assert scheduler.req_to_token_pool.available_size() == 2

        first.output_ids.append(1)
        plan.running_batch.reqs.append(first)
        plan = scheduler.get_new_batch_prefill(plan.running_batch)
        assert plan.batch_to_run is None
        assert plan.running_batch.batch_is_full
        unit = scheduler.session_bridge.units_by_request_id[second.rid]
        assert unit.capacity_wait_started_at is not None
        started_at = unit.capacity_wait_started_at

        attempted = []
        native_add = PrefillAdder.add_one_req

        def record_add(adder, request, has_chunked_req, truncation_align_size):
            attempted.append(request.rid)
            return native_add(adder, request, has_chunked_req, truncation_align_size)

        monkeypatch.setattr(PrefillAdder, "add_one_req", record_add)
        plan = scheduler.get_new_batch_prefill(plan.running_batch)
        assert plan.batch_to_run is None
        assert attempted == [second.rid]
        assert unit.capacity_wait_started_at == started_at

        scheduler.new_token_ratio_tracker.current = 0.0
        plan = scheduler.get_new_batch_prefill(plan.running_batch)
        assert plan.batch_to_run.reqs == [second]
        assert unit.capacity_wait_started_at == started_at


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("first_is_session", [False, True])
def test_native_prefill_bypasses_turn_that_does_not_fit_the_kv_budget(
    first_is_session: bool,
) -> None:
    """A native KV budget refusal must not block a smaller held-row continuation."""
    with get_context().override_server_args(page_size=4):
        pool_state = AdmissionScheduler(
            request_rows=4, kv_pool_tokens=24, device="cuda", page_size=4
        )
        pool_state.open_session("held", holds_row=True)
        if first_is_session:
            pool_state.open_session("fresh", holds_row=False)
            fresh = pool_state.append_unit(
                "fresh", "fresh", is_enqueued=True, prompt_tokens=9, max_new_tokens=3
            )
        else:
            fresh = pool_state.queue_request("fresh")
            fresh.sampling_params.max_new_tokens = 3
        fresh.origin_input_ids = array("q", range(50, 59))
        fresh.sampling_params.normalize(None)
        if first_is_session:
            assert pool_state.session_bridge.check_session_capacity("fresh") is None
        else:
            pass
        continuation = pool_state.append_unit(
            "continuation", "held", is_enqueued=True, prompt_tokens=9
        )
        continuation.origin_input_ids = array("q", range(1, 10))
        continuation.sampling_params.normalize(None)
        assert pool_state.session_bridge.check_session_capacity("continuation") is None
        assert pool_state.token_to_kv_pool_allocator.available_size() == 16
        assert pool_state.free_rows() == 3

        scheduler = native_prefill_scheduler(pool_state)
        plan = scheduler.get_new_batch_prefill(scheduler.running_batch)

    assert plan.batch_to_run is not None
    assert plan.batch_to_run.reqs == [continuation]
    assert scheduler.waiting_queue == [fresh]
    if first_is_session:
        assert (
            scheduler.session_bridge.units_by_request_id[
                fresh.rid
            ].capacity_wait_started_at
            is not None
        )
    assert (
        scheduler.session_bridge.units_by_request_id[
            continuation.rid
        ].capacity_wait_started_at
        is None
    )


def test_retry_prefill_discards_rejected_ignore_eos_budget(
    native_runtime_context: None,
) -> None:
    """A rejected fixed-length request must not reserve KV against later candidates."""
    pool_state = AdmissionScheduler(request_rows=4, kv_pool_tokens=16)
    pool_state.tree_cache.disable = True
    rejected_ids = []

    class RecordingRetryPrefillAdder(SessionRetryPrefillAdder):
        def on_capacity_wait(self, request_id: str) -> None:
            rejected_ids.append(request_id)

    adder = RecordingRetryPrefillAdder(
        page_size=1,
        tree_cache=pool_state.tree_cache,
        token_to_kv_pool_allocator=pool_state.token_to_kv_pool_allocator,
        running_batch=pool_state.running_batch,
        new_token_ratio=0.7,
        rem_input_tokens=64,
        rem_chunk_tokens=None,
    )
    rejected = Req(
        rid="too-long",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=SamplingParams(max_new_tokens=12, ignore_eos=True),
    )
    accepted = Req(
        rid="fits",
        origin_input_text="",
        origin_input_ids=array("q", [3, 4]),
        sampling_params=SamplingParams(max_new_tokens=4, ignore_eos=True),
    )
    for request in (rejected, accepted):
        request.sampling_params.normalize(None)
        request.init_next_round_input(pool_state.tree_cache)

    assert adder.add_one_req(rejected, False, None) == AddReqResult.CONTINUE
    assert adder.add_one_req(accepted, False, None) == AddReqResult.CONTINUE
    assert adder.can_run_list == [accepted]
    assert rejected_ids == [rejected.rid]


@pytest.mark.parametrize("raise_on_retry", [False, True])
def test_native_prefill_retry_restores_queued_and_requeued_requests(
    monkeypatch: pytest.MonkeyPatch, raise_on_retry: bool
) -> None:
    """Native requeues survive the retry once, including when selection raises."""
    scheduler = object.__new__(OmniScheduler)
    first = Req("first", "", [1], SamplingParams(max_new_tokens=1))
    continuation = Req("continuation", "", [2], SamplingParams(max_new_tokens=1))
    continuation.session = SimpleNamespace(streaming=True)
    requeued = Req("requeued", "", [3], SamplingParams(max_new_tokens=1))
    scheduler.waiting_queue = [first, continuation]
    scheduler.session_bridge = SimpleNamespace()
    scheduler.continuation_aware_admission = True
    scheduler.is_retrying_session_prefill = False
    scheduler.session_prefill_row_cache = None
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
    running_batch = SimpleNamespace(batch_is_full=False)
    calls = []

    def native_prefill(scheduler: OmniScheduler, batch: ScheduleBatch) -> NextBatchPlan:
        calls.append(list(scheduler.waiting_queue))
        if len(calls) == 2:
            assert issubclass(
                scheduler.get_prefill_adder_class(), SessionRetryPrefillAdder
            )
            scheduler.waiting_queue.append(requeued)
            if raise_on_retry:
                raise RuntimeError("native prefill failed")
            else:
                pass
        else:
            pass
        return NextBatchPlan(batch_to_run=None, running_batch=batch)

    monkeypatch.setattr(
        upstream_scheduler.Scheduler, "get_new_batch_prefill", native_prefill
    )
    if raise_on_retry:
        with pytest.raises(RuntimeError, match="native prefill failed"):
            scheduler.get_native_batch_prefill(running_batch)
    else:
        assert scheduler.get_native_batch_prefill(running_batch).batch_to_run is None
    assert calls == [[first, continuation], [first, continuation]]
    assert scheduler.waiting_queue == [first, continuation, requeued]
    assert not scheduler.is_retrying_session_prefill
    assert scheduler.session_prefill_row_cache is None


@pytest.mark.parametrize(
    "has_session, continuation_aware", [(False, True), (True, False)]
)
def test_native_prefill_does_not_retry_without_session_admission(
    monkeypatch: pytest.MonkeyPatch, has_session: bool, continuation_aware: bool
) -> None:
    """Ordinary stages and disabled continuation admission keep native selection."""
    scheduler = object.__new__(OmniScheduler)
    first = Req("first", "", [1], SamplingParams(max_new_tokens=1))
    second = Req("second", "", [2], SamplingParams(max_new_tokens=1))
    if has_session:
        second.session = SimpleNamespace(streaming=True)
    else:
        pass
    scheduler.waiting_queue = [first, second]
    scheduler.session_bridge = SimpleNamespace()
    scheduler.continuation_aware_admission = continuation_aware
    scheduler.is_retrying_session_prefill = False
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
    running_batch = SimpleNamespace(batch_is_full=False)
    calls = []

    def native_prefill(scheduler: OmniScheduler, batch: ScheduleBatch) -> NextBatchPlan:
        calls.append(list(scheduler.waiting_queue))
        return NextBatchPlan(batch_to_run=None, running_batch=batch)

    monkeypatch.setattr(
        upstream_scheduler.Scheduler, "get_new_batch_prefill", native_prefill
    )
    assert scheduler.get_native_batch_prefill(running_batch).batch_to_run is None
    assert calls == [[first, second]]
    assert scheduler.waiting_queue == [first, second]


def test_retry_row_count_cache_tracks_capacity_and_queue_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One retry pass reuses its row count only while its inputs stay unchanged."""
    scheduler = AdmissionScheduler(request_rows=4)
    scheduler.open_session("held", holds_row=True)
    scheduler.open_session("fresh", holds_row=False)
    scheduler.queue_request("fresh", "fresh")
    scheduler.queue_request("continuation", "held")
    bridge = scheduler.session_bridge
    count_rows = bridge.count_row_reusing_requests
    counted: list[int] = []

    def count(waiting_queue: list[Req], free_rows: int) -> int:
        counted.append(free_rows)
        return count_rows(waiting_queue, free_rows)

    monkeypatch.setattr(bridge, "count_row_reusing_requests", count)
    scheduler.session_prefill_row_cache = (-1, -1, 0)
    assert (
        scheduler.get_num_allocatable_reqs(0, running_batch=scheduler.running_batch)
        == 4
    )
    assert (
        scheduler.get_num_allocatable_reqs(0, running_batch=scheduler.running_batch)
        == 4
    )
    assert counted == [3]

    assert scheduler.req_to_token_pool.alloc_rows(1) is not None
    assert (
        scheduler.get_num_allocatable_reqs(0, running_batch=scheduler.running_batch)
        == 3
    )
    scheduler.queue_request("ordinary")
    assert (
        scheduler.get_num_allocatable_reqs(0, running_batch=scheduler.running_batch)
        == 3
    )
    assert counted == [3, 2, 2]


def test_session_units_stay_within_the_per_batch_limit(
    per_batch_limit: SimpleNamespace,
) -> None:
    per_batch_limit.pp_max_micro_batch_size = 6
    scheduler = AdmissionScheduler(request_rows=9)
    for index in range(8):
        scheduler.open_session(f"session-{index}", holds_row=True)
        scheduler.queue_request(f"unit-{index}", f"session-{index}")

    assert (
        scheduler.get_num_allocatable_reqs(
            2, None, running_batch=scheduler.running_batch
        )
        == 4
    )


@pytest.mark.parametrize("peer_is_enqueued", [False, True])
def test_capacity_waiters_do_not_promise_request_rows(peer_is_enqueued: bool) -> None:
    """A refused fresh unit must not keep the next eligible unit from taking a row."""
    scheduler = AdmissionScheduler(request_rows=2)
    for session_id in ("candidate", "peer"):
        scheduler.open_session(session_id, holds_row=False)
    scheduler.append_unit("candidate", "candidate", is_enqueued=False)
    scheduler.append_unit("peer", "peer", is_enqueued=peer_is_enqueued)

    refusal = scheduler.session_bridge.check_session_capacity("candidate")

    assert (refusal is None) == (not peer_is_enqueued)
    if refusal is not None:
        assert "request-to-token" in refusal
    else:
        pass


@pytest.mark.parametrize("peer_is_enqueued", [False, True])
def test_capacity_waiters_do_not_reserve_future_kv(peer_is_enqueued: bool) -> None:
    """Unadmitted work cannot consume a later candidate's available KV budget."""
    scheduler = AdmissionScheduler(request_rows=4)
    for session_id in ("candidate", "peer"):
        scheduler.open_session(session_id, holds_row=False)
    scheduler.append_unit("candidate", "candidate", is_enqueued=False)
    scheduler.append_unit("peer", "peer", is_enqueued=peer_is_enqueued)
    scheduler.token_to_kv_pool_allocator.available_tokens = 4

    assert (scheduler.session_bridge.check_session_capacity("candidate") is None) == (
        not peer_is_enqueued
    )


@pytest.mark.parametrize("prefix_matched", [False, True])
@pytest.mark.parametrize("available_tokens", [7, 8])
def test_queued_peer_retained_history_is_credited_before_and_after_prefix_match(
    prefix_matched: bool, available_tokens: int
) -> None:
    """A queued continuation must not reserve its already retained history twice."""
    scheduler = AdmissionScheduler(request_rows=4)
    scheduler.open_session("candidate", holds_row=False)
    scheduler.open_session("peer", holds_row=True)
    scheduler.append_unit("candidate", "candidate", is_enqueued=False)
    peer = scheduler.append_unit("peer", "peer", is_enqueued=True, prompt_tokens=11)
    if prefix_matched:
        scheduler.tree_cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", peer.origin_input_ids[:-1])), req=peer
            )
        )
    else:
        pass
    scheduler.token_to_kv_pool_allocator.available_tokens = available_tokens

    assert (scheduler.session_bridge.check_session_capacity("candidate") is None) == (
        available_tokens == 8
    )


@pytest.mark.parametrize("available_tokens", [12, 16])
def test_page_rounding_charges_each_request_separately(available_tokens: int) -> None:
    """Two five-token requests need four four-token pages, not ten token slots."""
    scheduler = AdmissionScheduler(request_rows=4)
    for session_id in ("candidate", "peer"):
        scheduler.open_session(session_id, holds_row=False)
        scheduler.append_unit(
            session_id, session_id, is_enqueued=session_id == "peer", max_new_tokens=2
        )
    scheduler.token_to_kv_pool_allocator.page_size = 4
    scheduler.token_to_kv_pool_allocator.available_tokens = available_tokens

    assert (scheduler.session_bridge.check_session_capacity("candidate") is None) == (
        available_tokens == 16
    )


@pytest.mark.parametrize(
    "is_running, ignore_eos, available_tokens, is_admitted",
    [
        (False, False, 9, False),
        (True, False, 8, False),
        (True, False, 9, True),
        (True, True, 9, False),
    ],
)
def test_only_running_peers_use_the_native_remaining_output_estimate(
    is_running: bool, ignore_eos: bool, available_tokens: int, is_admitted: bool
) -> None:
    """A running early-EOS peer frees reservation headroom without lowering its cap."""
    scheduler = AdmissionScheduler(request_rows=4)
    scheduler.open_session("candidate", holds_row=False)
    scheduler.open_session("peer", holds_row=True)
    scheduler.append_unit("candidate", "candidate", is_enqueued=False)
    peer = scheduler.append_unit(
        "peer", "peer", is_enqueued=True, prompt_tokens=11, max_new_tokens=16
    )
    peer.output_ids.extend([20, 21])
    peer.sampling_params.ignore_eos = ignore_eos
    scheduler.tree_cache.session.slots["peer"].kv.kv_allocated_len = 12
    scheduler.token_to_kv_pool_allocator.available_tokens = available_tokens

    refusal = scheduler.session_bridge.check_session_capacity(
        "candidate",
        running_request_ids=frozenset({"peer"}) if is_running else frozenset(),
        new_token_ratio=0.25,
    )

    assert (refusal is None) == is_admitted
    assert peer.sampling_params.max_new_tokens == 16


@pytest.mark.parametrize(
    "retain_completed_kv, is_enqueued, is_retained",
    [(False, True, False), (True, True, True), (False, False, True)],
)
def test_completed_discarding_adapter_returns_its_physical_kv_and_row(
    retain_completed_kv: bool,
    is_enqueued: bool,
    is_retained: bool,
    native_runtime_context: None,
) -> None:
    """A completed reset-on-next-reply model must not pin the last reply's KV idle."""
    scheduler = AdmissionScheduler(request_rows=2, kv_pool_tokens=RETAINED_TOKEN_COUNT)
    adapter = ARSessionAdapter()
    adapter.retain_completed_kv = retain_completed_kv
    scheduler.session_bridge.adapter = adapter
    scheduler.open_session("peer", holds_row=True)
    scheduler.append_unit("peer", "peer", is_enqueued=is_enqueued)
    native_session = scheduler.session_controller.get("peer")
    assert scheduler.free_rows() == 1

    scheduler.session_bridge.complete("peer")

    assert scheduler.free_rows() == (1 if is_retained else 2)
    assert scheduler.token_to_kv_pool_allocator.available_size() == (
        0 if is_retained else RETAINED_TOKEN_COUNT
    )
    assert ("peer" in scheduler.tree_cache.session_records()) == is_retained
    assert scheduler.session_controller.get("peer") is native_session
    assert scheduler.session_bridge.sessions["peer"].unit is None


@pytest.mark.parametrize("new_token_ratio, selected_count", [(1.0, 0), (0.25, 1)])
def test_native_prefill_selects_an_extra_reply_after_running_reservation_estimation(
    new_token_ratio: float, selected_count: int, native_runtime_context: None
) -> None:
    """Running estimation admits an additional reply through both admission gates."""
    scheduler = AdmissionScheduler(request_rows=4, kv_pool_tokens=21)
    scheduler.open_session("peer", holds_row=True)
    scheduler.open_session("candidate", holds_row=False)
    peer = scheduler.append_unit(
        "peer", "peer", is_enqueued=True, prompt_tokens=7, max_new_tokens=16
    )
    peer.kv = scheduler.tree_cache.session.slots["peer"].kv
    peer.output_ids.append(99)
    scheduler.running_batch.reqs.append(peer)
    candidate = scheduler.append_unit(
        "candidate", "candidate", is_enqueued=False, max_new_tokens=4
    )
    candidate.init_next_round_input(scheduler.tree_cache)
    adder = PrefillAdder(
        page_size=1,
        tree_cache=scheduler.tree_cache,
        token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
        running_batch=scheduler.running_batch,
        new_token_ratio=new_token_ratio,
        rem_input_tokens=CONTEXT_LENGTH,
        rem_chunk_tokens=None,
    )

    refusal = scheduler.session_bridge.check_session_capacity(
        "candidate",
        running_request_ids=frozenset({"peer"}),
        new_token_ratio=new_token_ratio,
    )
    adder.add_one_req(candidate, has_chunked_req=False, truncation_align_size=None)

    assert (refusal is None) == bool(selected_count)
    assert len(adder.can_run_list) == selected_count
    assert scheduler.req_to_token_pool.alloc(adder.can_run_list) is not None
    assert scheduler.free_rows() == 3 - selected_count
    assert peer.sampling_params.max_new_tokens == 16
    assert candidate.sampling_params.max_new_tokens == 4


def make_pressured_scheduler(kv_pool_tokens: int) -> OmniScheduler:
    pool_state = AdmissionScheduler(request_rows=4, kv_pool_tokens=kv_pool_tokens)
    for session_id in ("survivor", "victim"):
        pool_state.open_session(session_id, holds_row=True)
        request = pool_state.append_unit(
            session_id, session_id, is_enqueued=True, prompt_tokens=7, max_new_tokens=16
        )
        request.kv = pool_state.tree_cache.session.slots[session_id].kv
        request.skip_radix_cache_insert = True
        request._omni_terminal_claimed = (
            False  # noqa: leading-underscore  # scheduler lifecycle field
        )
        request.omni_data = SGLangARRequestData(req=request)
        request.sampling_params.normalize(None)
        request.output_ids.extend([99, 100] if session_id == "survivor" else [99])
        pool_state.running_batch.reqs.append(request)
    pool_state.waiting_queue.clear()
    scheduler = object.__new__(OmniScheduler)
    scheduler.__dict__.update(pool_state.__dict__)
    scheduler.session_bridge.bridge_scheduler = scheduler
    scheduler.session_bridge.adapter = ARSessionAdapter()
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.enable_overlap = False
    scheduler.enable_skip_finishing_decode = False
    scheduler.enable_plain_fallback = False
    scheduler.enable_priority_scheduling = True
    scheduler.schedule_low_priority_values_first = True
    scheduler.abort_on_priority_when_disabled = False
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.max_queued_requests = None
    scheduler.enable_hicache_storage = scheduler.enable_lmcache = False
    scheduler.processed_tokens_counter = 0
    scheduler.new_token_ratio_tracker = NewTokenRatioTracker.from_config()
    scheduler.is_entry_rank = True
    scheduler.tp_size = 1
    scheduler.request_admission_lock = threading.RLock()
    scheduler.aborted_request_ids = set()
    scheduler.aborted_request_id_order = deque()
    scheduler.pending_request_builds = {}
    scheduler.pending_request_admissions = {}
    scheduler.pending_session_admissions = {}
    scheduler.backlogged_request_build_payloads = deque()
    scheduler.pending_stream_ingress = {"victim": {}}
    scheduler.deferred_request_payloads = {}
    scheduler.dirty_deferred_request_ids = set()
    scheduler.first_emit_done = set()
    scheduler.prefill_start_done = set()
    scheduler.prefill_end_done = set()
    scheduler.abort_callback = None
    scheduler.inbox = queue.Queue()
    scheduler.outbox = queue.Queue()
    scheduler.cur_batch = scheduler.last_batch = None
    scheduler.chunked_req = None
    scheduler.decode_batch_states = None
    scheduler.previous_pending_decode = scheduler.async_pending = None
    scheduler.result_queue = deque()
    batch = scheduler.running_batch
    batch.device = "cpu"
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.token_to_kv_pool_allocator = scheduler.token_to_kv_pool_allocator
    batch.req_to_token_pool = scheduler.req_to_token_pool
    batch.tree_cache = scheduler.tree_cache
    batch.spec_algorithm = SpeculativeAlgorithm.NONE
    batch.hisparse_coordinator = None
    batch.req_pool_indices = torch.tensor(
        [request.kv.req_pool_idx for request in batch.reqs]
    )
    batch.req_pool_indices_cpu = batch.req_pool_indices.clone()
    batch.seq_lens = torch.tensor([8, 8])
    batch.seq_lens_cpu = batch.seq_lens.clone()
    batch.orig_seq_lens = batch.seq_lens.clone()
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(batch, vocab_size=128)
    return scheduler


@pytest.fixture
def pressured_scheduler(native_runtime_context: None) -> OmniScheduler:
    return make_pressured_scheduler(kv_pool_tokens=16)


def test_real_decode_shortage_releases_one_session_and_preserves_the_survivor(
    pressured_scheduler: OmniScheduler,
) -> None:
    """Actual next-step shortage frees the chosen session's resources exactly once."""
    scheduler = pressured_scheduler
    batch = scheduler.running_batch
    assert not batch.check_decode_mem()

    assert scheduler.relieve_session_capacity(batch)

    assert [request.rid for request in batch.reqs] == ["survivor"]
    assert batch.check_decode_mem()
    assert scheduler.req_to_token_pool.available_size() == 3
    assert scheduler.token_to_kv_pool_allocator.available_size() == 8
    assert set(scheduler.tree_cache.session_records()) == {"survivor"}
    assert set(scheduler.session_bridge.units_by_request_id) == {"survivor"}
    assert scheduler.session_controller.get("victim") is None
    assert "victim" not in scheduler.pending_stream_ingress
    error = scheduler.outbox.get_nowait()
    assert error.request_id == "victim"
    assert isinstance(error.data, SessionCapacityExhaustedError)
    assert scheduler.relieve_session_capacity(batch)
    assert scheduler.outbox.empty()
    survivor_next_position = scheduler.token_to_kv_pool_allocator.alloc(1)
    assert survivor_next_position is not None
    survivor = scheduler.tree_cache.session_records()["survivor"]
    survivor_positions = scheduler.req_to_token_pool.req_to_token[
        survivor.req_pool_idx, :RETAINED_TOKEN_COUNT
    ]
    assert not torch.isin(survivor_next_position, survivor_positions).any()
    assert scheduler.req_to_token_pool.alloc_rows(1) is not None


def test_session_capacity_kill_reestimates_running_output_demand(
    pressured_scheduler: OmniScheduler,
) -> None:
    """A capacity kill must revise the decayed admission ratio for survivors."""
    scheduler = pressured_scheduler
    batch = scheduler.running_batch
    survivor = batch.reqs[0]
    scheduler.new_token_ratio_tracker.current = 0.001
    expected = NewTokenRatioTracker.estimate_new_token_ratio_after_retract([survivor])
    assert expected > scheduler.new_token_ratio_tracker.current

    assert scheduler.relieve_session_capacity(batch)

    assert batch.reqs == [survivor]
    assert scheduler.new_token_ratio_tracker.current == pytest.approx(expected)
    bridge = scheduler.session_bridge
    identity = SessionIdentity("candidate")
    bridge.apply_operation(
        StagePayload("candidate-open", OmniRequest(None), {}),
        SessionOperation("open", identity, ("talker",)),
    )
    unit = bridge.accept(
        StagePayload("candidate", OmniRequest(None), {}),
        SessionOperation(
            "append", identity, ("talker",), TimedChunk("text", 0, 0, 0, {})
        ),
    )
    unit.session_request = Req(
        "candidate", "", array("q", [1, 2, 3]), SamplingParams(max_new_tokens=1)
    )
    assert (
        bridge.check_session_capacity(
            "candidate",
            running_request_ids=frozenset({"survivor"}),
            new_token_ratio=0.1,
        )
        is None
    )
    assert (
        bridge.check_session_capacity(
            "candidate",
            running_request_ids=frozenset({"survivor"}),
            new_token_ratio=scheduler.new_token_ratio_tracker.current,
        )
        == "session KV capacity exhausted"
    )


def test_session_capacity_kill_does_not_lower_survivor_output_reservation(
    pressured_scheduler: OmniScheduler,
) -> None:
    """A non-recoverable session kill cannot relax the survivors' reservation."""
    scheduler = pressured_scheduler
    batch = scheduler.running_batch
    survivor = batch.reqs[0]
    survivor.sampling_params.max_new_tokens = 1024
    scheduler.new_token_ratio_tracker.current = 0.7
    assert NewTokenRatioTracker.estimate_new_token_ratio_after_retract([survivor]) < 0.7

    assert scheduler.relieve_session_capacity(batch)

    assert batch.reqs == [survivor]
    assert scheduler.new_token_ratio_tracker.current == pytest.approx(0.7)


def test_last_session_capacity_kill_resets_output_demand_estimate(
    pressured_scheduler: OmniScheduler,
) -> None:
    """After the last running session is killed, later work starts from the idle estimate."""
    scheduler = pressured_scheduler
    batch = scheduler.running_batch
    idle_request, victim = batch.reqs
    scheduler.session_bridge.release_append_unit(idle_request.rid)
    batch.filter_batch(reqs_to_exclude={idle_request})
    assert batch.reqs == [victim]
    assert not batch.check_decode_mem()
    scheduler.new_token_ratio_tracker.current = 0.001

    assert scheduler.relieve_session_capacity(batch)

    assert batch.is_empty()
    assert (
        scheduler.new_token_ratio_tracker.current
        == scheduler.new_token_ratio_tracker.init
    )


@pytest.mark.parametrize("is_entry_rank", [False, True])
def test_session_capacity_kill_closes_bridge_and_pending_read(
    pressured_scheduler: OmniScheduler,
    monkeypatch: pytest.MonkeyPatch,
    is_entry_rank: bool,
) -> None:
    """A capacity kill releases the session and closes the read on every rank."""
    scheduler = pressured_scheduler
    scheduler.is_entry_rank = is_entry_rank
    scheduler.tp_size = 1 if is_entry_rank else 2
    bridge = scheduler.session_bridge
    closed: list[SessionIdentity] = []
    monkeypatch.setattr(bridge.adapter, "close", closed.append)
    read = StagePayload("victim-read", OmniRequest(None), {})
    bridge.sessions["victim"].output_read = read

    assert scheduler.relieve_session_capacity(scheduler.running_batch)

    assert "victim" not in bridge.sessions
    assert closed == [SessionIdentity("victim")]
    if is_entry_rank:
        error = scheduler.outbox.get_nowait()
        assert error.request_id == "victim"
        assert isinstance(error.data, SessionCapacityExhaustedError)
        read_error = scheduler.outbox.get_nowait()
        assert read_error.request_id == "victim-read"
        assert read_error.type == "error"
        assert isinstance(read_error.data, SessionCapacityExhaustedError)
        assert str(read_error.data) == str(error.data)
    else:
        result = scheduler.outbox.get_nowait()
        assert result.request_id == "victim-read"
        assert result.type == "result"
        assert result.data.data == {"closed": True}
    assert scheduler.outbox.empty()


def test_mixed_decode_pressure_requeues_the_lower_priority_ordinary_request(
    pressured_scheduler: OmniScheduler,
) -> None:
    """A lower-priority ordinary row must retract before a higher-priority session dies."""
    scheduler = pressured_scheduler
    batch = scheduler.running_batch
    survivor, ordinary = batch.reqs
    survivor.priority = 0
    ordinary.priority = 10
    bridge = scheduler.session_bridge
    bridge.release_append_unit(ordinary.rid)
    scheduler.tree_cache.session.slots.pop(ordinary.rid)
    ordinary.session = None
    bridge.release_streaming_session(ordinary.rid)
    bridge.sessions.pop(ordinary.rid)
    assert not batch.check_decode_mem()

    with get_schedule().override(
        retraction_policy="priority", schedule_low_priority_values_first=True
    ):
        assert scheduler.update_running_batch(batch) is batch

    assert batch.reqs == [survivor]
    assert scheduler.waiting_queue == [ordinary]
    assert ordinary.is_retracted and not ordinary.finished()
    assert set(scheduler.tree_cache.session_records()) == {"survivor"}
    assert set(bridge.units_by_request_id) == {"survivor"}
    assert scheduler.req_to_token_pool.available_size() == 3
    assert scheduler.token_to_kv_pool_allocator.available_size() == 7
    assert survivor.kv.kv_allocated_len == 9
    assert scheduler.outbox.empty()


def test_session_kill_leaves_final_ordinary_oom_to_native_retraction(
    native_runtime_context: None,
) -> None:
    """The last ordinary request keeps native OOM behavior after a session dies."""
    with get_context().override_server_args(
        page_size=1,
        speculative_algorithm="EAGLE",
        speculative_num_steps=5,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=5,
    ):
        scheduler = make_pressured_scheduler(kv_pool_tokens=24)
        AdmissionScheduler.open_session(scheduler, "idle", holds_row=True)
        batch = scheduler.running_batch
        ordinary, victim = batch.reqs
        ordinary.priority = 0
        ordinary.sampling_params.max_new_tokens = 4
        victim.priority = 10
        bridge = scheduler.session_bridge
        bridge.release_append_unit(ordinary.rid)
        scheduler.tree_cache.session.slots.pop(ordinary.rid)
        ordinary.session = None
        bridge.release_streaming_session(ordinary.rid)
        bridge.sessions.pop(ordinary.rid)
        batch.spec_algorithm = SpeculativeAlgorithm.EAGLE
        with get_schedule().override(
            retraction_policy="priority", schedule_low_priority_values_first=True
        ):
            assert scheduler.relieve_session_capacity(batch)

        assert batch.reqs == [ordinary]
        assert scheduler.waiting_queue == []
        assert not batch.check_decode_mem()
        retracted, _, aborted = batch.retract_decode()
        assert retracted == []
        assert aborted == [ordinary]
        assert ordinary.to_finish is not None
        assert batch.is_empty()


class DeviceCompletion:
    def __init__(self) -> None:
        self.is_ready = False

    def query(self) -> bool:
        return self.is_ready

    def synchronize(self) -> None:
        self.is_ready = True


class CompletedReplyAdapter(ARSessionAdapter):
    retain_completed_kv: bool = False

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        assert request_data.stage_payload is not None
        return request_data.stage_payload


class CompletedTokenRunner:
    def execute_resolve(self, pending_step: PendingStep) -> ModelRunnerOutput:
        assert pending_step.event.query()
        return ModelRunnerOutput(outputs={}, next_token_ids=torch.tensor([101]))

    def on_request_finished(
        self, request_id: str, request_data: SGLangARRequestData
    ) -> None:
        pass


@pytest.mark.parametrize(
    "enable_overlap, skip_known_finishing",
    [(False, False), (True, False), (True, True)],
    ids=["async-pending", "overlap-drain", "overlap-finishing"],
)
def test_pending_decode_completion_frees_capacity_before_another_session_is_failed(
    pressured_scheduler: OmniScheduler,
    enable_overlap: bool,
    skip_known_finishing: bool,
) -> None:
    """Drain a launched completion before choosing a capacity victim from live sessions."""
    scheduler = pressured_scheduler
    scheduler.enable_overlap = enable_overlap
    scheduler.enable_skip_finishing_decode = skip_known_finishing
    batch = scheduler.running_batch
    batch.enable_overlap = enable_overlap
    survivor, completing = batch.reqs
    completing.sampling_params.max_new_tokens = 2
    completing.omni_data = SGLangARRequestData(
        req=completing,
        stage_payload=StagePayload(completing.rid, OmniRequest(None), {}),
    )
    scheduler.session_bridge.adapter = CompletedReplyAdapter()
    scheduler.model_runner = CompletedTokenRunner()
    scheduler.draft_worker = None
    scheduler.stream_output_builder = None
    scheduler.request_finished_callback = None
    scheduler.completed_request_ids = {}
    scheduler.async_prefill = False
    model_config = SimpleNamespace(
        is_encoder_decoder=False,
        think_end_ids=None,
        self_opened_think_start_ids=None,
    )
    beam_coordinator = BeamCoordinator(
        model_config=model_config,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        dllm_enabled=False,
        max_req_len=CONTEXT_LENGTH,
        req_to_token_pool=scheduler.req_to_token_pool,
        token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
        tree_cache=scheduler.tree_cache,
        future_map=None,
    )
    result_processor = SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_overlap=enable_overlap,
        enable_overlap_mlx=False,
        model_config=model_config,
        token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
        tree_cache=scheduler.tree_cache,
        hisparse_coordinator=None,
        req_to_token_pool=scheduler.req_to_token_pool,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(
            num_generated_tokens=0,
            forward_ct_decode=0,
            report_decode_stats=lambda can_run_cuda_graph, running_batch, num_generated_tokens: None,
        ),
        draft_worker=None,
        model_worker=None,
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(stream_output=scheduler.stream_output),
        beam_coordinator=beam_coordinator,
        abort_request=scheduler.abort,
    )
    scheduler.process_batch_result = result_processor.process_batch_result_decode
    pending_batch = ScheduleBatch(
        reqs=[completing],
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        device="cpu",
    )
    scheduler_output = SchedulerOutput(requests=[], batch_data=pending_batch)
    completion_event = DeviceCompletion()
    pending_step = PendingStep(
        event=completion_event,
        launch_buf=None,
        scheduler_output=scheduler_output,
        forward_batch=None,
        schedule_batch=pending_batch,
        batch_result=OmniScheduler.make_batch_result(
            ModelRunnerOutput(outputs={}, next_token_ids=torch.tensor([101]))
        ),
    )
    if enable_overlap:
        pending_step.batch_result.copy_done = completion_event
        scheduler.result_queue.append((pending_batch, pending_step.batch_result))
    else:
        scheduler.async_pending = PendingDecode(
            batch=pending_batch,
            scheduler_output=scheduler_output,
            device_step=pending_step,
        )
    assert not batch.check_decode_mem()

    batch.batch_is_full = True
    if skip_known_finishing:
        assert scheduler.update_running_batch(batch) is None
        assert batch.reqs == [survivor]
        assert not survivor.finished() and not completing.finished()
        assert not completion_event.query()
        assert scheduler.token_to_kv_pool_allocator.available_size() == 0
        assert scheduler.result_queue
        assert not scheduler.aborted_request_ids
        assert scheduler.outbox.empty()
        scheduler.resolve_pending_async()
    else:
        pass
    assert scheduler.update_running_batch(batch) is batch

    assert completion_event.query()
    assert scheduler.async_pending is None
    assert not scheduler.result_queue
    assert batch.reqs == [survivor]
    assert not batch.batch_is_full
    assert not scheduler.aborted_request_ids
    assert set(scheduler.session_bridge.units_by_request_id) == {"survivor"}
    assert set(scheduler.tree_cache.session_records()) == {"survivor"}
    assert scheduler.token_to_kv_pool_allocator.available_size() == 7
    assert survivor.kv.kv_allocated_len == 9
    result = scheduler.outbox.get_nowait()
    assert result.request_id == completing.rid and result.type == "result"
    assert scheduler.outbox.empty()
