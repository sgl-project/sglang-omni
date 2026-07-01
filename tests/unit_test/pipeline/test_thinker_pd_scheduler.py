from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from sglang_omni.scheduling import omni_scheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


class _FakeReq:
    def __init__(self, rid: str, *, finished: bool = False):
        self.rid = rid
        self._finished = finished
        self._omni_data = SimpleNamespace(request_id=rid)
        self.req_pool_idx = None
        self.mamba_pool_idx = None

    def finished(self) -> bool:
        return self._finished


class _FakeMode:
    def __init__(self, *, extend: bool):
        self._extend = extend

    def is_extend(self) -> bool:
        return self._extend


class _FakeBatch:
    def __init__(self, reqs, *, extend: bool = True, batch_is_full: bool = False):
        self.reqs = list(reqs)
        self.forward_mode = _FakeMode(extend=extend)
        self.batch_is_full = batch_is_full
        self.chunked_req = None
        self.is_prefill_only = False

    def batch_size(self) -> int:
        return len(self.reqs)

    def is_empty(self) -> bool:
        return not self.reqs

    def filter_batch(self, chunked_req_to_exclude=None, keep_indices=None, **_kwargs):
        if keep_indices is not None:
            self.reqs = [self.reqs[i] for i in keep_indices]
            return
        excluded = set(chunked_req_to_exclude or [])
        self.reqs = [
            req for req in self.reqs if not req.finished() and req not in excluded
        ]

    def merge_batch(self, other) -> None:
        self.reqs.extend(other.reqs)


class _FakePool:
    def __init__(self, available: int):
        self._available = available

    def available_size(self) -> int:
        return self._available


def _scheduler() -> OmniScheduler:
    scheduler = object.__new__(OmniScheduler)
    scheduler._pd_ready_decode_batches = deque()
    scheduler._pd_ready_decode_limit = 4
    scheduler._pd_prefill_admission_active = False
    scheduler._enable_thinker_pd = True
    scheduler.chunked_req = None
    scheduler._chunked_req_scheduled_last_iter = False
    scheduler.running_batch = _FakeBatch([], extend=False, batch_is_full=True)
    scheduler.last_batch = None
    scheduler.is_mixed_chunk = False
    scheduler.max_running_requests = 4
    scheduler.req_to_token_pool = _FakePool(available=16)
    return scheduler


def test_pd_stashes_prefill_batch_and_admits_to_decode(monkeypatch):
    events = []
    monkeypatch.setattr(
        omni_scheduler,
        "_emit_event",
        lambda **kwargs: events.append(kwargs),
    )
    scheduler = _scheduler()
    reqs = [_FakeReq("r1"), _FakeReq("r2")]
    scheduler.last_batch = _FakeBatch(reqs, extend=True)

    assert OmniScheduler._pd_stash_last_prefill_batch(scheduler) is True

    assert scheduler.running_batch.batch_is_full is False
    assert scheduler.last_batch is None
    assert OmniScheduler._pd_ready_decode_req_count(scheduler) == 2
    assert [event["event_name"] for event in events] == [
        "scheduler_pd_ready_decode_enter",
        "scheduler_pd_ready_decode_enter",
    ]

    assert OmniScheduler._pd_stash_last_prefill_batch(scheduler) is False
    assert OmniScheduler._pd_ready_decode_req_count(scheduler) == 2

    OmniScheduler._pd_admit_ready_decode_batches(scheduler)

    assert scheduler.running_batch.reqs == reqs
    assert OmniScheduler._pd_ready_decode_req_count(scheduler) == 0
    assert [event["event_name"] for event in events[-2:]] == [
        "scheduler_pd_decode_admit",
        "scheduler_pd_decode_admit",
    ]


def test_pd_does_not_overfill_decode_slots(monkeypatch):
    monkeypatch.setattr(omni_scheduler, "_emit_event", lambda **_kwargs: None)
    scheduler = _scheduler()
    scheduler.running_batch = _FakeBatch(
        [_FakeReq("active-1"), _FakeReq("active-2")],
        extend=False,
    )
    scheduler._pd_ready_decode_batches.append(
        _FakeBatch([_FakeReq("ready-1"), _FakeReq("ready-2"), _FakeReq("ready-3")])
    )

    OmniScheduler._pd_admit_ready_decode_batches(scheduler)

    assert [req.rid for req in scheduler.running_batch.reqs] == [
        "active-1",
        "active-2",
    ]
    assert OmniScheduler._pd_ready_decode_req_count(scheduler) == 3


def test_pd_allocatable_reqs_uses_ready_decode_room(monkeypatch):
    scheduler = _scheduler()
    scheduler._pd_ready_decode_batches.append(
        _FakeBatch([_FakeReq("ready-1"), _FakeReq("ready-2")])
    )
    scheduler._pd_prefill_admission_active = True

    assert OmniScheduler.get_num_allocatable_reqs(scheduler, running_bs=4) == 2

    scheduler._pd_prefill_admission_active = False
    monkeypatch.setattr(
        omni_scheduler._Upstream,
        "get_num_allocatable_reqs",
        lambda _self, _running_bs: 7,
    )
    assert OmniScheduler.get_num_allocatable_reqs(scheduler, running_bs=4) == 7


def test_pd_allocatable_reqs_never_builds_oversized_ready_decode_batch():
    scheduler = _scheduler()
    scheduler._pd_ready_decode_limit = 32
    scheduler.max_running_requests = 4
    scheduler.req_to_token_pool = _FakePool(available=32)
    scheduler._pd_prefill_admission_active = True

    assert OmniScheduler.get_num_allocatable_reqs(scheduler, running_bs=0) == 4


def test_pd_prefill_admission_temporarily_disables_mixed_chunk(monkeypatch):
    scheduler = _scheduler()
    scheduler.is_mixed_chunk = True
    seen = []

    def fake_get_new_batch_prefill(self):
        seen.append((self._pd_prefill_admission_active, self.is_mixed_chunk))
        return "new-batch"

    monkeypatch.setattr(
        omni_scheduler._Upstream,
        "get_new_batch_prefill",
        fake_get_new_batch_prefill,
    )

    assert OmniScheduler._pd_get_new_batch_prefill(scheduler) == "new-batch"
    assert seen == [(True, False)]
    assert scheduler.is_mixed_chunk is True
    assert scheduler._pd_prefill_admission_active is False


def test_pd_ready_decode_requests_participate_in_abort_lookup():
    scheduler = _scheduler()
    ready = _FakeReq("ready")
    scheduler._pd_ready_decode_batches.append(_FakeBatch([ready]))
    scheduler.waiting_queue = []
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._async_pending = None

    assert OmniScheduler._find_request_data(scheduler, "ready") is ready._omni_data
    assert OmniScheduler._active_request_ids(scheduler) == ["ready"]

    OmniScheduler._remove_from_pd_ready_decode(scheduler, "ready")

    assert OmniScheduler._pd_ready_decode_req_count(scheduler) == 0
