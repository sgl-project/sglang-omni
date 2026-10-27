# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the prefill admission-coalescing gate.

The gate holds prefill until ``prefill_coalesce_requests`` are waiting or the
oldest queued request has waited ``prefill_coalesce_wait_ms``. The deadline is
keyed on each request's enqueue time (``_coalesce_enqueue_t``), so partial
upstream admission or an aborted request never restarts the window for the
requests left behind. Chunked prefill in flight and an empty queue pass
straight through. Tested against a stub scheduler with the upstream call
patched to a sentinel.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace, TracebackType
from unittest import mock

import pytest

pytest.importorskip("sglang")

from sglang.srt.managers.schedule_batch import NextBatchPlan  # noqa: E402

from sglang_omni.scheduling import omni_scheduler  # noqa: E402
from sglang_omni.scheduling.omni_scheduler import OmniScheduler  # noqa: E402

UPSTREAM_BATCH = object()


def make_req(enqueue_t: float | None):
    if enqueue_t is None:
        return SimpleNamespace()
    return SimpleNamespace(_coalesce_enqueue_t=enqueue_t)


class StubScheduler:
    """The attribute surface get_new_batch_prefill touches."""

    def __init__(
        self,
        *,
        coalesce_requests: int,
        wait_ms: float = 60.0,
        coalesce_when_idle: bool = False,
        requires_pending_builds: bool = False,
        coalesce_after_builds_during_decode: bool = False,
    ) -> None:
        self.prefill_coalesce_requests = coalesce_requests
        self.prefill_coalesce_wait_s = wait_ms / 1e3
        self.prefill_coalesce_when_idle = coalesce_when_idle
        self.prefill_coalesce_requires_pending_builds = requires_pending_builds
        self.prefill_coalesce_after_builds_during_decode = (
            coalesce_after_builds_during_decode
        )
        self.async_prefill = False
        self.chunked_req = None
        self.waiting_queue: list = []
        self.running_batch = SimpleNamespace(is_empty=lambda: False)
        self.request_admission_lock = threading.RLock()
        self.pending_request_builds: dict = {}
        self.pending_request_admissions: dict = {}
        self.backlogged_request_build_payloads: list = []

    def prefill_coalesce_waiting_requests(
        self, running_batch: SimpleNamespace | None
    ) -> list[SimpleNamespace] | None:
        return OmniScheduler.prefill_coalesce_waiting_requests(self, running_batch)

    def prefill_coalesce_holds(
        self, waiting: list[SimpleNamespace], now_s: float
    ) -> bool:
        return OmniScheduler.prefill_coalesce_holds(self, waiting, now_s)

    def get_new_batch_prefill(self):
        # Upstream takes running_batch in and hands back a NextBatchPlan;
        # unwrap it so the assertions below stay about the gate decision.
        plan = OmniScheduler.get_new_batch_prefill(self, self.running_batch)
        return plan.batch_to_run


@pytest.fixture()
def upstream():
    with mock.patch.object(
        omni_scheduler._Upstream,  # noqa: leading-underscore  # production name
        "get_new_batch_prefill",
        return_value=NextBatchPlan(batch_to_run=UPSTREAM_BATCH, running_batch=None),
    ) as patched:
        yield patched


@pytest.fixture()
def clock():
    with mock.patch.object(omni_scheduler.time, "perf_counter") as patched:
        patched.return_value = 100.0
        yield patched


def test_disabled_gate_passes_through(upstream):
    sched = StubScheduler(coalesce_requests=0)
    sched.waiting_queue = [make_req(0.0)]
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH


def test_chunked_prefill_bypasses_gate(upstream):
    sched = StubScheduler(coalesce_requests=8)
    sched.waiting_queue = [make_req(0.0)]
    sched.chunked_req = object()
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH


def test_empty_queue_passes_through(upstream):
    sched = StubScheduler(coalesce_requests=8)
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH


def test_full_batch_passes_through_immediately(upstream):
    sched = StubScheduler(coalesce_requests=4)
    sched.waiting_queue = [make_req(100.0)] * 4
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH


def test_small_queue_is_held_until_oldest_expires(upstream, clock):
    sched = StubScheduler(coalesce_requests=8, wait_ms=60.0)
    sched.waiting_queue = [make_req(100.0), make_req(100.01)]

    clock.return_value = 100.03  # oldest has waited 30ms of the 60ms window
    assert sched.get_new_batch_prefill() is None

    clock.return_value = 100.07  # oldest past the deadline
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH
    upstream.assert_called_once()


def test_idle_loop_bypasses_gate(upstream):
    # Note: (maydomine) no decode in flight: holding amortizes nothing, only
    # costs TTFB.
    sched = StubScheduler(coalesce_requests=8)
    sched.waiting_queue = [make_req(0.0)]
    sched.running_batch = None
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH

    sched.running_batch = SimpleNamespace(is_empty=lambda: True)
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH


def test_idle_loop_can_coalesce_when_explicitly_enabled(upstream, clock):
    sched = StubScheduler(
        coalesce_requests=8,
        wait_ms=10.0,
        coalesce_when_idle=True,
    )
    sched.running_batch = None
    sched.waiting_queue = [make_req(100.0)]

    clock.return_value = 100.005
    assert sched.get_new_batch_prefill() is None

    clock.return_value = 100.011
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH


@pytest.mark.parametrize("source", ["pending", "admission", "backlog"])
def test_pending_build_work_holds_small_prefill(upstream, clock, source):
    sched = StubScheduler(
        coalesce_requests=8,
        wait_ms=6.0,
        coalesce_when_idle=True,
        requires_pending_builds=True,
    )
    sched.running_batch = None
    sched.waiting_queue = [make_req(100.0)]
    if source == "pending":
        sched.pending_request_builds["building"] = object()
    elif source == "admission":
        sched.pending_request_admissions["admitting"] = object()
    else:
        sched.backlogged_request_build_payloads.append(object())

    clock.return_value = 100.005
    assert sched.get_new_batch_prefill() is None


def test_pending_build_gate_releases_when_build_work_drains(upstream, clock):
    sched = StubScheduler(
        coalesce_requests=8,
        wait_ms=6.0,
        coalesce_when_idle=True,
        requires_pending_builds=True,
    )
    sched.running_batch = None
    sched.waiting_queue = [make_req(100.0)]
    sched.pending_request_builds["building"] = object()

    clock.return_value = 100.001
    assert sched.get_new_batch_prefill() is None

    sched.pending_request_builds.clear()
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH


def test_pending_build_gate_releases_at_target(upstream, clock):
    sched = StubScheduler(
        coalesce_requests=8,
        wait_ms=6.0,
        coalesce_when_idle=True,
        requires_pending_builds=True,
    )
    sched.running_batch = None
    sched.waiting_queue = [make_req(100.0)] * 8
    sched.pending_request_builds["building"] = object()

    clock.return_value = 100.001
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH


def test_pending_build_gate_releases_at_deadline(upstream, clock):
    sched = StubScheduler(
        coalesce_requests=8,
        wait_ms=6.0,
        coalesce_when_idle=True,
        requires_pending_builds=True,
    )
    sched.running_batch = None
    sched.waiting_queue = [make_req(100.0)]
    sched.pending_request_builds["building"] = object()

    clock.return_value = 100.006
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH


def test_pending_build_gate_does_not_wait_without_build_work(upstream, clock):
    sched = StubScheduler(
        coalesce_requests=8,
        wait_ms=6.0,
        coalesce_when_idle=True,
        requires_pending_builds=True,
    )
    sched.running_batch = None
    sched.waiting_queue = [make_req(100.0)]

    clock.return_value = 100.001
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH


def test_decode_can_coalesce_after_build_work_drains(upstream, clock):
    sched = StubScheduler(
        coalesce_requests=8,
        wait_ms=6.0,
        coalesce_when_idle=True,
        requires_pending_builds=True,
        coalesce_after_builds_during_decode=True,
    )
    sched.waiting_queue = [make_req(100.0)]

    clock.return_value = 100.001
    assert sched.get_new_batch_prefill() is None


def test_idle_decode_still_releases_after_build_work_drains(upstream, clock):
    sched = StubScheduler(
        coalesce_requests=8,
        wait_ms=6.0,
        coalesce_when_idle=True,
        requires_pending_builds=True,
        coalesce_after_builds_during_decode=True,
    )
    sched.running_batch = None
    sched.waiting_queue = [make_req(100.0)]

    clock.return_value = 100.001
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH


def test_real_partial_admission_cycle_releases_leftovers_immediately(clock):
    # Note: (Jiaxin Deng) real admit cycle: upstream pops only the head of an
    # expired wave; the leftover keeps its stamp and releases on the next pass.
    def admit_head(self, running_batch):
        self.waiting_queue.pop(0)
        return NextBatchPlan(batch_to_run=UPSTREAM_BATCH, running_batch=running_batch)

    sched = StubScheduler(coalesce_requests=8, wait_ms=60.0)
    clock.return_value = 100.07
    leftover = make_req(100.005)
    sched.waiting_queue = [make_req(100.0), leftover]
    with mock.patch.object(
        omni_scheduler._Upstream,  # noqa: leading-underscore  # production name
        "get_new_batch_prefill",
        autospec=True,
        side_effect=admit_head,
    ) as patched:
        assert sched.get_new_batch_prefill() is UPSTREAM_BATCH
        assert sched.waiting_queue == [leftover]
        assert sched.get_new_batch_prefill() is UPSTREAM_BATCH
        assert patched.call_count == 2
        assert sched.waiting_queue == []


def test_newcomer_after_queue_drain_waits_its_own_window(upstream, clock):
    # Note: (Jiaxin Deng) once the arming request leaves the queue (abort's
    # queue filtering), a stamp-on-miss newcomer ages from its own arrival.
    sched = StubScheduler(coalesce_requests=8, wait_ms=60.0)
    clock.return_value = 100.0
    armer = make_req(None)
    sched.waiting_queue = [armer]
    assert sched.get_new_batch_prefill() is None
    assert (
        armer._coalesce_enqueue_t == 100.0
    )  # noqa: leading-underscore  # production name

    clock.return_value = 100.059  # abort just before the armer's deadline
    sched.waiting_queue.remove(armer)
    newcomer = make_req(None)
    sched.waiting_queue.append(newcomer)
    assert sched.get_new_batch_prefill() is None
    assert (
        newcomer._coalesce_enqueue_t == 100.059
    )  # noqa: leading-underscore  # production name

    clock.return_value = 100.07  # past the armer's window, inside the newcomer's
    assert sched.get_new_batch_prefill() is None

    clock.return_value = 100.12  # newcomer's own deadline expires
    assert sched.get_new_batch_prefill() is UPSTREAM_BATCH


class CountingLock:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.acquire_count = 0

    def __enter__(self) -> bool:
        self.acquire_count += 1
        return self.lock.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.lock.__exit__(exc_type, exc, traceback)


class ClockAdvancingLock(CountingLock):
    def __init__(self, clock: mock.MagicMock, now_after_acquire_s: float) -> None:
        super().__init__()
        self.clock = clock
        self.now_after_acquire_s = now_after_acquire_s

    def __enter__(self) -> bool:
        self.clock.return_value = self.now_after_acquire_s
        return super().__enter__()


def gate_state(
    *,
    coalesce_requests: int,
    has_chunked_request: bool,
    coalesce_when_idle: bool,
    decode_is_idle: bool,
    requires_pending_builds: bool,
    has_build_work: bool,
    coalesce_after_builds_during_decode: bool,
    enqueue_times: tuple[float | None, ...],
) -> StubScheduler:
    sched = StubScheduler(
        coalesce_requests=coalesce_requests,
        wait_ms=60.0,
        coalesce_when_idle=coalesce_when_idle,
        requires_pending_builds=requires_pending_builds,
        coalesce_after_builds_during_decode=coalesce_after_builds_during_decode,
    )
    sched.chunked_req = object() if has_chunked_request else None
    sched.running_batch = SimpleNamespace(is_empty=lambda: decode_is_idle)
    sched.request_admission_lock = CountingLock()
    sched.pending_request_builds = {"build": object()} if has_build_work else {}
    sched.waiting_queue = [make_req(enqueue_t) for enqueue_t in enqueue_times]
    return sched


def enqueue_stamps(sched: StubScheduler) -> list[float | None]:
    return [
        getattr(
            req, "_coalesce_enqueue_t", None
        )  # noqa: leading-underscore  # production name
        for req in sched.waiting_queue
    ]


@pytest.mark.parametrize("enqueue_times", [(99.99, 99.995, None)])
@pytest.mark.parametrize("requires_pending_builds", [False, True])
def test_the_hold_predicate_stamps_nothing_and_agrees_with_the_gate(
    upstream: mock.MagicMock,
    clock: mock.MagicMock,
    enqueue_times: tuple[float | None, ...],
    requires_pending_builds: bool,
) -> None:
    sched = gate_state(
        coalesce_requests=8,
        has_chunked_request=False,
        coalesce_when_idle=False,
        decode_is_idle=False,
        requires_pending_builds=requires_pending_builds,
        has_build_work=True,
        coalesce_after_builds_during_decode=False,
        enqueue_times=enqueue_times,
    )

    waiting = sched.prefill_coalesce_waiting_requests(sched.running_batch)
    assert waiting is not None
    is_held = OmniScheduler.prefill_coalesce_holds(sched, waiting, clock.return_value)

    assert enqueue_stamps(sched) == list(enqueue_times)
    assert is_held == (sched.get_new_batch_prefill() is None)


class DecodingStubScheduler(StubScheduler):
    def __init__(self, *, request_admission_lock: CountingLock) -> None:
        super().__init__(coalesce_requests=8, requires_pending_builds=True)
        self.is_mixed_chunk = False
        self.dllm_config = None
        self._prefill_decode_interval_remaining = (
            0  # noqa: leading-underscore  # production name
        )
        self.enable_priority_preemption = False
        self.is_hybrid_swa = False
        self.enable_hisparse = False
        self.min_free_slots_delayer = None
        self.running_batch = SimpleNamespace(
            is_empty=lambda: False, reqs=[object()], batch_is_full=False
        )
        self.request_admission_lock = request_admission_lock
        self.pending_request_builds = {"build": object()}
        self.waiting_queue = [make_req(100.0)]

    def get_num_allocatable_reqs(
        self, running_request_count: int, running_batch: SimpleNamespace
    ) -> int:
        return 4


def test_the_prefill_expectation_reads_the_clock_before_the_pending_builds(
    clock: mock.MagicMock,
) -> None:
    clock.return_value = 100.059999
    lock = ClockAdvancingLock(clock, now_after_acquire_s=100.060001)
    sched = DecodingStubScheduler(request_admission_lock=lock)

    is_expected = OmniScheduler.is_prefill_expected(sched)

    assert lock.acquire_count == 1
    assert clock.return_value == 100.060001
    assert is_expected is False
    assert enqueue_stamps(sched) == [100.0]


@pytest.mark.parametrize("async_prefill", [False, True])
def test_build_admission_preserves_fifo_unless_async_prefill_is_enabled(
    async_prefill: bool,
) -> None:
    """A later completed build may bypass the first only on the opt-in path."""
    first = Future()
    second = Future()
    second.set_result("second-result")
    scheduler = SimpleNamespace(
        async_prefill=async_prefill,
        request_admission_lock=threading.RLock(),
        pending_request_builds={
            "first": ("first-payload", False, first),
            "second": ("second-payload", False, second),
        },
        aborted_request_ids=set(),
        admit_or_defer_built_request=mock.Mock(),
    )
    OmniScheduler.drain_request_build_results(scheduler)
    assert scheduler.admit_or_defer_built_request.call_count == int(async_prefill)
    if async_prefill:
        assert (
            scheduler.admit_or_defer_built_request.call_args.args[0] == "second-payload"
        )
    first.set_result("first-result")
    OmniScheduler.drain_request_build_results(scheduler)
    expected = (
        ["second-payload", "first-payload"]
        if async_prefill
        else ["first-payload", "second-payload"]
    )
    assert [
        call.args[0] for call in scheduler.admit_or_defer_built_request.call_args_list
    ] == expected
    assert not scheduler.pending_request_builds


@pytest.mark.parametrize(
    "batch_full,available_slots", [(True, 4), (False, 0), (False, 4)]
)
def test_expired_coalescing_does_not_repeat_wait_when_admission_is_blocked(
    upstream: mock.MagicMock,
    clock: mock.MagicMock,
    batch_full: bool,
    available_slots: int,
) -> None:
    """A full running batch must keep decoding without waiting for request builds."""
    clock.return_value = 101.0
    scheduler = DecodingStubScheduler(request_admission_lock=threading.RLock())
    scheduler.async_prefill = True
    scheduler.running_batch.batch_is_full = batch_full
    scheduler.get_num_allocatable_reqs = lambda *args, **kwargs: available_slots
    scheduler.is_prefill_expected = lambda: OmniScheduler.is_prefill_expected(scheduler)
    scheduler.admit_imminent_request_builds = mock.Mock()
    for _ in range(3):
        assert scheduler.get_new_batch_prefill() is UPSTREAM_BATCH
    assert scheduler.admit_imminent_request_builds.call_count == (
        3 if not batch_full and available_slots else 0
    )


@pytest.mark.parametrize("async_prefill", [False, True])
def test_drain_admits_a_build_that_finishes_during_another_admission(
    async_prefill: bool,
) -> None:
    """Opt-out admission must retain same-iteration completion collection."""
    first = Future()
    second = Future()
    first.set_result("first-result")
    admitted = []

    def admit(payload, pending_stream_done, result, *, request_admission_lock_held):
        assert not pending_stream_done
        assert request_admission_lock_held
        admitted.append((payload, result))
        if payload == "first-payload":
            second.set_result("second-result")

    scheduler = SimpleNamespace(
        async_prefill=async_prefill,
        request_admission_lock=threading.RLock(),
        pending_request_builds={
            "first": ("first-payload", False, first),
            "second": ("second-payload", False, second),
        },
        aborted_request_ids=set(),
        admit_or_defer_built_request=admit,
    )
    OmniScheduler.drain_request_build_results(scheduler)
    assert admitted == [
        ("first-payload", "first-result"),
        ("second-payload", "second-result"),
    ]
    assert not scheduler.pending_request_builds
