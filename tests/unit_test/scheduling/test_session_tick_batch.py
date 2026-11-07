# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import time
from functools import partial
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("sglang")

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    LM_STEP_MS_KEY,
    NS_PER_MS,
    SESSION_METADATA_KEY,
    Cohort,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
)
from sglang_omni.scheduling import omni_scheduler
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import (
    LM_STEP_EMA_WEIGHT,
    HeldPromptUnit,
    OmniScheduler,
    PendingDecode,
)
from sglang_omni.scheduling.session import SessionScheduler
from sglang_omni.scheduling.tick import TickCohorts
from tests.unit_test.scheduling.test_session_batch import RecordingHooks, opens
from tests.unit_test.scheduling.test_session_gather_window import (
    ScriptedInbox,
    run_next_batch,
)

PERIOD_S = 0.08
NS_PER_SECOND = 1_000_000_000


def tick_message(session_id: str, tick: Cohort | None) -> IncomingMessage:
    request_id = f"{session_id}0"
    session_operation = SessionOperation(
        operation="append",
        session_identity=SessionIdentity(session_id),
        stages=("speech",),
        chunk=TimedChunk("audio", 0, 80, 0, b"x"),
        tick=tick,
    )
    request = OmniRequest(
        None, metadata={SESSION_METADATA_KEY: session_operation.to_dict()}
    )
    return IncomingMessage(
        request_id, "new_request", StagePayload(request_id, request, {})
    )


def slot(index: int, size: int) -> Cohort:
    return Cohort(
        index=index, size=size, due_ns=round((index + 1) * PERIOD_S * NS_PER_SECOND)
    )


def scripted_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    arrivals: list[tuple[str, float, Cohort | None]],
) -> tuple[SessionScheduler, ScriptedInbox]:
    scheduler = SessionScheduler(RecordingHooks())
    inbox = ScriptedInbox()
    inbox.now_seconds = PERIOD_S
    for queued in opens(*[session_id for session_id, _, _ in arrivals]):
        scheduler.register_operation(queued)
        scheduler.compute(queued.data)
    for session_id, seconds, tick in arrivals:
        queued = tick_message(session_id, tick)
        scheduler.register_operation(queued)
        inbox.arrivals.append((seconds, queued))
    scheduler.inbox = inbox
    monkeypatch.setattr(time, "monotonic", lambda: inbox.now_seconds)
    monkeypatch.setattr(
        time, "monotonic_ns", lambda: round(inbox.now_seconds * NS_PER_SECOND)
    )
    return scheduler, inbox


def test_a_ticks_appends_run_in_one_batch_once_the_last_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, inbox = scripted_scheduler(
        monkeypatch,
        [("a", 0.08, slot(1, 3)), ("b", 0.09, slot(1, 3)), ("c", 0.12, slot(1, 3))],
    )
    assert run_next_batch(scheduler, inbox) == ["a0", "b0", "c0"]
    assert inbox.now_seconds == pytest.approx(0.12)


def test_a_ticks_batch_runs_when_the_tick_is_over_without_the_late_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, inbox = scripted_scheduler(
        monkeypatch,
        [("a", 0.08, slot(1, 3)), ("b", 0.09, slot(1, 3)), ("c", 0.2, slot(1, 3))],
    )
    assert run_next_batch(scheduler, inbox) == ["a0", "b0"]
    assert inbox.now_seconds == pytest.approx(2 * PERIOD_S)
    assert run_next_batch(scheduler, inbox) == ["c0"]


def test_appends_off_the_grid_are_not_held(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler, inbox = scripted_scheduler(
        monkeypatch, [("a", 0.08, None), ("b", 0.09, None)]
    )
    assert run_next_batch(scheduler, inbox) == ["a0"]
    assert inbox.now_seconds == pytest.approx(0.08)
    assert inbox.timeouts == []


def test_a_wait_that_runs_out_between_two_clock_reads_is_not_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StubScheduler()
    due_ns = 10 * NS_PER_SECOND
    stub.tick_cohorts.arrive(Cohort(index=1, size=2, due_ns=due_ns))
    # The hold is decided one microsecond before the tick ends and the wait computed after it.
    clock = iter([due_ns - 1000, due_ns + 1000])
    monkeypatch.setattr(time, "monotonic_ns", lambda: next(clock))
    timeouts: list[float] = []

    def get(timeout: float) -> IncomingMessage:
        timeouts.append(timeout)
        raise queue.Empty

    stub.inbox.get = get
    assert OmniScheduler.wait_for_tick_cohort(stub)
    assert timeouts == [0.0]


def test_a_stage_wait_that_runs_out_between_two_clock_reads_is_not_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, inbox = scripted_scheduler(
        monkeypatch, [("a", 0.08, slot(1, 2)), ("b", 0.3, slot(1, 2))]
    )
    due_ns = slot(1, 2).due_ns
    reads = iter([due_ns - 1000, due_ns + 1000, due_ns + 2000, due_ns + 3000])
    monkeypatch.setattr(time, "monotonic_ns", lambda: next(reads))
    inbox.now_seconds = 0.08
    batch = scheduler.collect_batch(inbox.get_nowait())
    assert [message.request_id for message in batch] == ["a0"]
    assert all(timeout >= 0 for timeout in inbox.timeouts)


def test_cohorts_hold_until_whole_or_over() -> None:
    cohorts = TickCohorts()
    cohorts.arrive(slot(1, 2))
    assert cohorts.hold_until_ns(slot(1, 2).due_ns - 1) == slot(1, 2).due_ns
    assert cohorts.is_collecting(slot(1, 2).due_ns - 1)
    cohorts.arrive(slot(1, 2))
    assert cohorts.hold_until_ns(slot(1, 2).due_ns - 1) is None
    assert cohorts.is_collecting(slot(1, 2).due_ns - 1)
    cohorts.ran([slot(1, 2)])
    assert not cohorts.is_collecting(slot(1, 2).due_ns - 1)

    cohorts.arrive(slot(2, 2))
    assert cohorts.hold_until_ns(slot(2, 2).due_ns) is None
    assert not cohorts.is_collecting(slot(2, 2).due_ns)


def test_a_late_row_of_a_tick_whose_batch_ran_is_not_waited_for() -> None:
    cohorts = TickCohorts()
    cohorts.arrive(slot(1, 3))
    cohorts.arrive(slot(1, 3))
    cohorts.ran([slot(1, 3)])
    cohorts.arrive(slot(1, 3))
    assert cohorts.hold_until_ns(slot(1, 3).due_ns - 1) is None
    assert not cohorts.is_collecting(slot(1, 3).due_ns - 1)
    cohorts.arrive(slot(2, 1))
    assert cohorts.hold_until_ns(slot(1, 3).due_ns) is None
    assert cohorts.is_collecting(slot(2, 1).due_ns - 1)


class StubScheduler:
    def __init__(self) -> None:
        self.tick_cohorts = TickCohorts()
        self.is_on_tick_grid = False
        self.held_prompt_units: dict = {}
        self.tp_size = 1
        self.is_entry_rank = True
        self.inbox: queue.Queue = queue.Queue()
        self.idle_wait_message = None
        self.units: dict[str, SimpleNamespace] = {}
        self.session_bridge = SimpleNamespace(
            units_by_request_id=self.units, stepped_by_request_id={}
        )
        self.separate_session_openings = False
        self.last_prefill_continued_sessions = False
        self.admitted: list[str] = []
        self.lm_step_ms: float | None = None

    def active_session_unit(self, request_id: str):
        return self.units.get(request_id)

    def admit_or_defer_built_request(self, payload, pending_stream_done, req_data):
        self.admitted.append(payload.request_id)


def held_prompt(stub: StubScheduler, request_id: str, release_ns: int = 0) -> None:
    payload = StagePayload(request_id, OmniRequest(None), {})
    stub.units[request_id] = SimpleNamespace(tick=None)
    stub.held_prompt_units[request_id] = HeldPromptUnit(
        payload=payload,
        pending_stream_done=False,
        req_data=object(),
        release_ns=release_ns,
    )


def test_lm_gate_waits_for_the_rest_of_the_tick_then_steps() -> None:
    stub = StubScheduler()
    now_ns = time.monotonic_ns()
    tick = Cohort(index=1, size=2, due_ns=now_ns + round(0.05 * NS_PER_SECOND))
    stub.tick_cohorts.arrive(tick)
    stub.inbox.put(IncomingMessage("b0", "new_request", None))
    assert OmniScheduler.wait_for_tick_cohort(stub)
    assert (
        stub.idle_wait_message is not None and stub.idle_wait_message.request_id == "b0"
    )
    stub.tick_cohorts.arrive(tick)
    assert not OmniScheduler.wait_for_tick_cohort(stub)


def test_lm_gate_stops_waiting_when_the_tick_is_over() -> None:
    stub = StubScheduler()
    now_ns = time.monotonic_ns()
    stub.tick_cohorts.arrive(
        Cohort(index=1, size=2, due_ns=now_ns + round(0.02 * NS_PER_SECOND))
    )
    waits = 0
    while OmniScheduler.wait_for_tick_cohort(stub):
        waits += 1
    assert waits >= 1
    assert time.monotonic_ns() >= now_ns + round(0.02 * NS_PER_SECOND) - 10**6


def test_prompt_unit_waits_for_the_ticks_batch_to_run() -> None:
    stub = StubScheduler()
    tick = Cohort(
        index=1, size=1, due_ns=time.monotonic_ns() + round(10 * NS_PER_SECOND)
    )
    stub.tick_cohorts.arrive(tick)
    stub.units["u0"] = SimpleNamespace(tick=tick)
    held_prompt(stub, "p0", tick.due_ns)
    OmniScheduler.release_held_prompt_units(stub)
    assert stub.admitted == []

    OmniScheduler.note_tick_batch(
        stub, SimpleNamespace(reqs=[SimpleNamespace(rid="u0")])
    )
    OmniScheduler.release_held_prompt_units(stub)
    assert stub.admitted == ["p0"]
    assert stub.held_prompt_units == {}


def test_prompt_unit_waits_at_most_until_the_next_tick() -> None:
    stub = StubScheduler()
    stub.tick_cohorts.arrive(
        Cohort(index=1, size=2, due_ns=time.monotonic_ns() + round(10 * NS_PER_SECOND))
    )
    held_prompt(stub, "p0", time.monotonic_ns())
    OmniScheduler.release_held_prompt_units(stub)
    assert stub.admitted == ["p0"]


def test_on_the_grid_a_pending_decode_runs_before_the_next_extend() -> None:
    seen: list[list] = []

    def upstream(self, running_batch, last_batch):
        seen.append(list(self.waiting_queue))
        return SimpleNamespace(batch_to_run="batch", running_batch=running_batch)

    stub = StubScheduler()
    stub.running_batch = SimpleNamespace(is_empty=lambda: True, batch_is_full=False)
    stub.chunked_req = None
    stub.waiting_queue = ["extend"]
    extend = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True),
        reqs=[SimpleNamespace(finished=lambda: False)],
    )
    stub.last_batch = extend
    with mock.patch.object(
        omni_scheduler._Upstream,  # noqa: leading-underscore  # production name
        "get_next_batch_to_run",
        upstream,
    ):
        stub.is_on_tick_grid = False
        assert OmniScheduler.get_next_batch_to_run(stub) == "batch"
        stub.is_on_tick_grid = True
        assert OmniScheduler.get_next_batch_to_run(stub) == "batch"
        extend.reqs[0].finished = lambda: True
        assert OmniScheduler.get_next_batch_to_run(stub) == "batch"
    assert seen == [["extend"], [], ["extend"]]
    assert stub.waiting_queue == ["extend"]


def test_prompt_unit_of_a_closed_session_is_dropped() -> None:
    stub = StubScheduler()
    held_prompt(stub, "p0")
    del stub.units["p0"]
    OmniScheduler.release_held_prompt_units(stub)
    assert stub.admitted == [] and stub.held_prompt_units == {}


def test_step_cost_is_measured_from_the_launch_and_averaged() -> None:
    stub = StubScheduler()
    OmniScheduler.note_step_cost(stub, True, 0, 20 * NS_PER_MS)
    assert stub.lm_step_ms == 20
    OmniScheduler.note_step_cost(stub, True, 0, 60 * NS_PER_MS)
    assert stub.lm_step_ms == pytest.approx(20 + LM_STEP_EMA_WEIGHT * 40)
    result = OmniScheduler.with_step_cost(
        stub, StagePayload("u0", OmniRequest(None), {})
    )
    assert result.request.metadata[LM_STEP_MS_KEY] == stub.lm_step_ms


def test_a_prompt_prefill_step_is_not_a_grid_step_cost() -> None:
    stub = StubScheduler()
    OmniScheduler.note_step_cost(stub, False, 0, 30 * NS_PER_MS)
    assert stub.lm_step_ms is None


def test_a_drained_lookahead_step_counts_from_its_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StubScheduler()
    stub.enable_overlap = False
    stub.previous_pending_decode = None
    stub.async_pending = PendingDecode(
        batch=SimpleNamespace(reqs=[]),
        scheduler_output=object(),
        device_step=object(),
        launch_ns=0,
        is_tick_step=True,
    )
    stub.wait_async_device = lambda batch, device_step: None
    stub.process_owned_async = lambda pending: None
    stub.note_step_cost = partial(OmniScheduler.note_step_cost, stub)
    monkeypatch.setattr(time, "monotonic_ns", lambda: 30 * NS_PER_MS)
    OmniScheduler.resolve_pending_async(stub)
    assert stub.async_pending is None
    assert stub.lm_step_ms == 30


def test_a_stepped_rows_waiting_unit_is_the_row_of_its_tick_batch() -> None:
    stub = StubScheduler()
    first, second = slot(1, 1), slot(2, 1)
    stub.tick_cohorts.arrive(first)
    stub.tick_cohorts.arrive(second)
    stub.units["r0"] = SimpleNamespace(tick=first)
    waiting = SimpleNamespace(unit=SimpleNamespace(tick=second))
    stub.session_bridge.stepped_by_request_id["r0"] = SimpleNamespace(
        stepped=SimpleNamespace(units=[waiting])
    )
    OmniScheduler.note_tick_batch(
        stub, SimpleNamespace(reqs=[SimpleNamespace(rid="r0")])
    )
    assert list(stub.tick_cohorts.cohorts) == [1]
