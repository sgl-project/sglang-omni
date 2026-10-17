# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers import scheduler as upstream_scheduler
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import SessionController
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession

from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling import omni_scheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionBridge,
    SessionUnit,
)

CONTEXT_LENGTH = 64
RETAINED_TOKEN_COUNT = 8
UNIT_TOKEN_IDS = list(range(1, RETAINED_TOKEN_COUNT + 4))
PAGE_SIZE = 16


class KVAllocator:
    device = "cpu"
    free_token_count = 1 << 20

    def __init__(self, page_size: int) -> None:
        self.page_size = page_size

    def free(self, free_index: torch.Tensor) -> None:
        pass

    def available_size(self) -> int:
        return self.free_token_count


class AdmissionScheduler:
    get_num_allocatable_reqs = OmniScheduler.get_num_allocatable_reqs

    def __init__(self, *, request_rows: int, page_size: int = 1) -> None:
        self.req_to_token_pool = ReqToTokenPool(
            size=request_rows,
            max_context_len=CONTEXT_LENGTH,
            device="cpu",
            enable_memory_saver=False,
        )
        self.token_to_kv_pool_allocator = KVAllocator(page_size)
        self.tree_cache = StreamingSession(
            RadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    page_size=page_size,
                    enable_kv_cache_events=False,
                    eviction_policy="lru",
                )
            )
        )
        self.session_controller = SessionController(self.tree_cache)
        self.max_running_requests = request_rows
        self.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
        self.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
        self.waiting_queue: list[Req] = []
        self.session_bridge = ARSessionBridge(self, adapter=None)

    def open_session(
        self,
        session_id: str,
        *,
        holds_row: bool,
        retained_tokens: int = RETAINED_TOKEN_COUNT,
    ) -> None:
        assert self.session_bridge.open_streaming_session(session_id)
        if holds_row:
            (row,) = self.req_to_token_pool.alloc_rows(1)
            self.tree_cache.slots[session_id] = SessionSlot(
                kv=ReqKvInfo(
                    req_pool_idx=row,
                    kv_committed_len=retained_tokens,
                    kv_allocated_len=retained_tokens,
                )
            )
        else:
            pass

    def queue_request(
        self,
        request_id: str,
        session_id: str | None = None,
        token_ids: list[int] | None = None,
    ) -> Req:
        request = Req(
            rid=request_id,
            origin_input_text="",
            origin_input_ids=UNIT_TOKEN_IDS if token_ids is None else token_ids,
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
                        key=RadixKey(request.origin_input_ids[:-1]), req=request
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
        held_rows[f"unit-{index}"] = scheduler.tree_cache.slots[
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


def open_pending_units(
    scheduler: AdmissionScheduler, *, retained_tokens: int, token_ids: list[int]
) -> ARSessionBridge:
    """Two sessions whose slots hold retained_tokens of history, each with a queued unit of token_ids."""
    bridge = scheduler.session_bridge
    for session_id in ("session-a", "session-b"):
        scheduler.open_session(
            session_id, holds_row=True, retained_tokens=retained_tokens
        )
        bridge.units_by_request_id[f"unit-{session_id}"] = SessionUnit(
            request_id=f"unit-{session_id}",
            session_identity=SessionIdentity(session_id),
            stages=("thinker",),
            chunk=TimedChunk("audio", 0, 20, 0, b"x"),
            session_request=scheduler.queue_request(
                f"unit-{session_id}", session_id, token_ids
            ),
        )
    return bridge


@pytest.mark.parametrize(
    ("spare_tokens", "capacity_message"),
    [(0, None), (-1, "session KV capacity exhausted")],
    ids=["fits", "one-token-short"],
)
def test_kv_check_counts_a_pending_unit_once_against_its_retained_history(
    spare_tokens: int, capacity_message: str | None
) -> None:
    """Two pending units need only their tokens past the history their slots already hold."""
    scheduler = AdmissionScheduler(request_rows=4)
    bridge = open_pending_units(
        scheduler, retained_tokens=RETAINED_TOKEN_COUNT, token_ids=UNIT_TOKEN_IDS
    )
    tokens_past_history = len(UNIT_TOKEN_IDS) + 1 - RETAINED_TOKEN_COUNT
    scheduler.token_to_kv_pool_allocator.free_token_count = (
        2 * tokens_past_history + spare_tokens
    )

    message = bridge.check_session_capacity("unit-session-a")

    assert message == capacity_message, "the other unit's retained history counts once"


@pytest.mark.parametrize(
    ("retained_tokens", "unit_token_count", "free_pages", "capacity_message"),
    [
        (PAGE_SIZE, PAGE_SIZE + 1, 1, "session KV capacity exhausted"),
        (PAGE_SIZE - 4, PAGE_SIZE - 3, 0, None),
    ],
    ids=["each-unit-needs-a-new-page", "units-stay-in-their-pages"],
)
def test_kv_check_reserves_whole_pages_past_the_pages_a_unit_holds(
    retained_tokens: int,
    unit_token_count: int,
    free_pages: int,
    capacity_message: str | None,
) -> None:
    """With 16-token pages, growth inside a held page needs no slot and growth past it needs a whole page."""
    scheduler = AdmissionScheduler(request_rows=4, page_size=PAGE_SIZE)
    bridge = open_pending_units(
        scheduler,
        retained_tokens=retained_tokens,
        token_ids=list(range(1, unit_token_count + 1)),
    )
    scheduler.token_to_kv_pool_allocator.free_token_count = free_pages * PAGE_SIZE

    message = bridge.check_session_capacity("unit-session-a")

    assert (
        message == capacity_message
    ), "each unit reserves the pages past those it holds"
