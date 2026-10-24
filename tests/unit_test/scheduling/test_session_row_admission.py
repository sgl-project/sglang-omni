# SPDX-License-Identifier: Apache-2.0

from array import array
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers import scheduler as upstream_scheduler
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import SessionController
from sglang.srt.session.streaming_session import SessionSlot

from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling import omni_scheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionBridge,
    BridgeSession,
    SessionUnit,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

CONTEXT_LENGTH = 64
RETAINED_TOKEN_COUNT = 8
UNIT_TOKEN_IDS = list(range(1, RETAINED_TOKEN_COUNT + 4))


class KVAllocator:
    device = "cpu"
    page_size = 1

    def __init__(self) -> None:
        self.free_tokens = 1 << 20

    def free(self, free_index: torch.Tensor) -> None:
        pass

    def available_size(self) -> int:
        return self.free_tokens


class AdmissionScheduler:
    get_num_allocatable_reqs = OmniScheduler.get_num_allocatable_reqs

    def __init__(self, *, request_rows: int) -> None:
        self.req_to_token_pool = ReqToTokenPool(
            size=request_rows,
            max_context_len=CONTEXT_LENGTH,
            device="cpu",
            enable_memory_saver=False,
        )
        self.token_to_kv_pool_allocator = KVAllocator()
        self.tree_cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=1,
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
        self.session_bridge = ARSessionBridge(self, adapter=ARSessionAdapter())
        self.model_config = SimpleNamespace(vocab_size=32000)
        self.drains = 0

    def synchronize_launched_decode(self) -> None:
        self.drains += 1

    def resolve_pending_async(self) -> None:
        pass

    def build_unit(self, session_id: str) -> SessionUnit:
        self.open_session(session_id, holds_row=False)
        identity = SessionIdentity(session_id)
        unit = SessionUnit(
            request_id=f"unit-{session_id}",
            session_identity=identity,
            stages=("lm",),
            chunk=TimedChunk("audio", 0, 80, 0, b""),
        )
        bridge = self.session_bridge
        bridge.sessions[session_id] = BridgeSession(
            session_identity=identity, unit=unit
        )
        bridge.units_by_request_id[unit.request_id] = unit
        request = Req(
            rid=unit.request_id,
            origin_input_text="",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        payload = StagePayload(unit.request_id, OmniRequest(None), {})
        bridge.create_session_request(
            payload,
            SGLangARRequestData(req=request, stage_payload=payload),
        )
        return unit

    def open_session(self, session_id: str, *, holds_row: bool) -> None:
        assert self.session_bridge.open_streaming_session(session_id)
        if holds_row:
            (row,) = self.req_to_token_pool.alloc_rows(1)
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
                0, None, running_batch=self.running_batch
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


@pytest.fixture(autouse=True)
def per_batch_limit(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    parallel = SimpleNamespace(pp_max_micro_batch_size=64)
    monkeypatch.setattr(omni_scheduler, "get_parallel", lambda: parallel)
    monkeypatch.setattr(upstream_scheduler, "get_parallel", lambda: parallel)
    return parallel


def test_every_open_session_unit_is_admitted_in_one_batch_with_one_free_row() -> None:
    session_count = 6
    scheduler = AdmissionScheduler(request_rows=session_count + 1)
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


@pytest.mark.parametrize("batched", [False, True])
def test_append_admission_keeps_the_reserved_row_and_collects_once_per_batch(
    monkeypatch, batched
):
    """Fresh appends must not consume the row reserved for retained sessions."""
    scheduler = AdmissionScheduler(request_rows=4)
    bridge = scheduler.session_bridge
    original_records = scheduler.tree_cache.session_records
    record_reads = []

    def records():
        record_reads.append(True)
        return original_records()

    monkeypatch.setattr(scheduler.tree_cache, "session_records", records)
    results = []
    with bridge.admission_batch() if batched else nullcontext():
        for index in range(4):
            unit = scheduler.build_unit(str(index))
            results.append(bridge.check_session_capacity(unit.request_id))
    assert results[:3] == [None, None, None]
    assert "request-to-token slots exhausted" in results[3]
    assert scheduler.drains == (1 if batched else 4)
    assert len(record_reads) == (1 if batched else 4)


@pytest.mark.parametrize("free_tokens", [7, 8, 9])
def test_admission_checks_the_combined_kv_demand_at_the_exact_boundary(free_tokens):
    """Two individually valid appends can exceed their shared KV budget."""
    scheduler = AdmissionScheduler(request_rows=4)
    scheduler.token_to_kv_pool_allocator.free_tokens = free_tokens
    bridge = scheduler.session_bridge
    with bridge.admission_batch():
        first = scheduler.build_unit("a")
        assert bridge.check_session_capacity(first.request_id) is None
        second = scheduler.build_unit("b")
        message = bridge.check_session_capacity(second.request_id)
    assert message == ("session KV capacity exhausted" if free_tokens == 7 else None)


def test_rechecking_a_unit_updates_its_shared_kv_reservation() -> None:
    """A changed token budget must replace the unit's previous reservation."""
    scheduler = AdmissionScheduler(request_rows=4)
    scheduler.token_to_kv_pool_allocator.free_tokens = 8
    bridge = scheduler.session_bridge
    with bridge.admission_batch():
        first = scheduler.build_unit("a")
        assert bridge.check_session_capacity(first.request_id) is None
        second = scheduler.build_unit("b")
        assert bridge.check_session_capacity(second.request_id) is None
        second.session_request.sampling_params.max_new_tokens = 2
        assert (
            bridge.check_session_capacity(second.request_id)
            == "session KV capacity exhausted"
        )
        first.session_request.sampling_params.max_new_tokens = 0
        assert bridge.check_session_capacity(first.request_id) is None
        assert bridge.check_session_capacity(second.request_id) is None


def test_pending_results_are_collected_before_capacity_is_snapshotted(monkeypatch):
    """A unit completed by the drain must not remain in the next admission budget."""
    scheduler = AdmissionScheduler(request_rows=3)
    scheduler.token_to_kv_pool_allocator.free_tokens = 4
    bridge = scheduler.session_bridge
    first = scheduler.build_unit("a")
    assert bridge.check_session_capacity(first.request_id) is None
    monkeypatch.setattr(
        scheduler,
        "resolve_pending_async",
        lambda: bridge.complete(first.request_id),
    )
    with bridge.admission_batch():
        second = scheduler.build_unit("b")
        assert bridge.check_session_capacity(second.request_id) is None


@pytest.mark.parametrize("release", ["release_append_unit", "complete"])
def test_a_removed_unit_no_longer_reserves_capacity_in_the_admission_batch(release):
    """A cancelled or completed unit must not leave a stale KV reservation."""
    scheduler = AdmissionScheduler(request_rows=4)
    scheduler.token_to_kv_pool_allocator.free_tokens = 8
    bridge = scheduler.session_bridge
    with bridge.admission_batch():
        first = scheduler.build_unit("a")
        assert bridge.check_session_capacity(first.request_id) is None
        second = scheduler.build_unit("b")
        assert bridge.check_session_capacity(second.request_id) is None
        getattr(bridge, release)(first.request_id)
        third = scheduler.build_unit("c")
        assert bridge.check_session_capacity(third.request_id) is None
    scheduler.token_to_kv_pool_allocator.free_tokens = 7
    assert (
        bridge.check_session_capacity(third.request_id)
        == "session KV capacity exhausted"
    )
