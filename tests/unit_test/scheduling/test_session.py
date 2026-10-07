# SPDX-License-Identifier: Apache-2.0
"""Session value validation and stage state ownership without worker processes."""

import queue
import threading
from collections import deque
from dataclasses import asdict, dataclass
from types import MethodType, SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import msgpack
import pytest

from sglang_omni.admission import ContextExhaustedError, QueueFullError
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    OutputChunk,
    ResourceUsage,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
    wire_size,
)
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.session import (
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionBridge,
    BridgeSession,
    SessionUnit,
)
from tests.unit_test.fixtures.session_pipeline import (
    compute_registered,
    operation_metadata,
)


@dataclass
class RecordedState:
    session_id: str
    byte_count: int = 0


class Hooks(SessionHooks):
    def __init__(self, name: str, events: queue.Queue[tuple[object, ...]]) -> None:
        self.name = name
        self.events = events
        self.states: dict[SessionIdentity, RecordedState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.events.put(("open", self.name, session_identity.id))
        self.states[session_identity] = RecordedState(session_identity.id)

    def close(self, session_identity: SessionIdentity) -> None:
        state = self.states.pop(session_identity)
        self.events.put(("close", self.name, state.session_id))


def test_open_usage_failure_releases_state():

    class BrokenUsage(Hooks):
        def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
            raise RuntimeError("usage failed")

    events = queue.Queue()
    scheduler = SessionScheduler(BrokenUsage("source", events))
    request = OmniRequest(
        None, metadata=operation_metadata("open", SessionIdentity("one"))
    )
    with pytest.raises(RuntimeError, match="usage failed"):
        compute_registered(scheduler, StagePayload("one-open", request, {}))
    assert events.get_nowait()[0] == "open"
    assert events.get_nowait()[0] == "close"
    with pytest.raises(RuntimeError, match="usage failed"):
        compute_registered(scheduler, StagePayload("one-open-again", request, {}))


def test_state_budget_is_per_session():

    class SizedHooks(Hooks):
        def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
            self.states[session_identity] = RecordedState(
                session_identity.id, byte_count=2
            )

        def append(
            self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
        ) -> StagePayload:
            self.states[context.session_identity].byte_count += 1
            return payload

        def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
            return ResourceUsage(bytes=self.states[session_identity].byte_count)

    events = queue.Queue()
    scheduler = SessionScheduler(
        SizedHooks("source", events), max_state_bytes_per_session=3
    )

    def invoke(sid, operation):
        request = OmniRequest(
            None,
            metadata=operation_metadata(
                operation, SessionIdentity(sid), TimedChunk("audio", 0, 20, 0, b"x")
            ),
        )
        return compute_registered(scheduler, StagePayload(sid + operation, request, {}))

    invoke("one", "open")
    invoke("two", "open")
    invoke("one", "append")
    with pytest.raises(QueueFullError):
        invoke("one", "append")
    # note (Junnan Li): The failed session is released at once, so a unit queued behind it cannot run.
    with pytest.raises(ValueError, match="unknown session"):
        invoke("one", "append")
    # note (Junnan Li): Session two stays within its own budget while session one outgrows it.
    invoke("two", "append")
    scheduler.stop()
    closed = sorted(events.get_nowait()[2] for _ in range(events.qsize()))
    assert closed == ["one", "two"]


def test_malformed_operation_fails_inside_the_request_boundary():

    scheduler = SessionScheduler(Hooks("source", queue.Queue()))
    worker = threading.Thread(target=scheduler.start)
    worker.start()
    try:
        request = OmniRequest(None, metadata={"omni_session": {"operation": "append"}})
        scheduler.inbox.put(
            IncomingMessage("bad", "new_request", StagePayload("bad", request, {}))
        )
        output = scheduler.outbox.get(timeout=5)
        assert output.request_id == "bad"
        assert output.type == "error"
        assert isinstance(output.data, ValueError)
        request = OmniRequest(
            None, metadata=operation_metadata("open", SessionIdentity("ok"))
        )
        scheduler.inbox.put(
            IncomingMessage("open", "new_request", StagePayload("open", request, {}))
        )
        assert scheduler.outbox.get(timeout=5).type == "result"
    finally:
        scheduler.stop()
        worker.join(timeout=5)


@pytest.mark.parametrize("configured", [False, True])
def test_ordinary_request_uses_handler_or_reports_scoped_error(configured):

    def compute(payload: StagePayload) -> StagePayload:
        payload.data = {"ordinary": True}
        return payload

    kwargs = {"compute_fn": compute} if configured else {}
    scheduler = SessionScheduler(Hooks("source", queue.Queue()), **kwargs)
    worker = threading.Thread(target=scheduler.start)
    worker.start()
    try:
        payload = StagePayload("ordinary", OmniRequest(None), {})
        scheduler.inbox.put(IncomingMessage("ordinary", "new_request", payload))
        output = scheduler.outbox.get(timeout=5)
        assert output.request_id == "ordinary"
        if configured:
            assert output.type == "result"
            assert output.data.data == {"ordinary": True}
        else:
            assert output.type == "error"
            assert isinstance(output.data, ValueError)
            assert "ordinary requests" in str(output.data)
        request = OmniRequest(
            None, metadata=operation_metadata("open", SessionIdentity("after-ordinary"))
        )
        scheduler.inbox.put(
            IncomingMessage("open", "new_request", StagePayload("open", request, {}))
        )
        assert scheduler.outbox.get(timeout=5).type == "result"
    finally:
        scheduler.stop()
        worker.join(timeout=5)
    assert not worker.is_alive()


@pytest.mark.parametrize("size", [0, 255, 256, 65535, 65536])
def test_binary_chunk_wire_size_matches_msgpack(size, monkeypatch):

    chunk = TimedChunk("audio", 0, 80, 0, b"x" * size, format="pcm16")
    output = OutputChunk(
        SessionIdentity("session"),
        0,
        0,
        **{key: value for key, value in asdict(chunk).items() if key != "seq"},
    )
    pack = msgpack.packb
    values = [asdict(chunk), asdict(output)]
    expected = [len(pack(value, use_bin_type=True)) for value in values]
    encoded_payloads = []

    def record(value, **kwargs):
        encoded_payloads.append(len(value["payload"]))
        return pack(value, **kwargs)

    monkeypatch.setattr(msgpack, "packb", record)
    assert [wire_size(value) for value in values] == expected
    assert all(size == 0 for size in encoded_payloads)


class BlockingHooks(Hooks):
    def __init__(self) -> None:
        super().__init__("source", queue.Queue())
        self.entered = threading.Event()
        self.release = threading.Event()

    def append(
        self,
        chunk: TimedChunk,
        payload: StagePayload,
        context: SessionContext,
    ) -> StagePayload:
        self.events.put(("append", payload.request_id))
        if payload.request_id == "first":
            self.entered.set()
            self.release.wait(5)
        return payload


def session_stage_payload(
    request_id: str, operation: Literal["open", "append", "close"]
) -> StagePayload:
    return StagePayload(
        request_id,
        OmniRequest(
            None,
            metadata=operation_metadata(
                operation,
                SessionIdentity("session"),
                TimedChunk("audio", 0, 20, 0, b"x"),
            ),
        ),
        {},
    )


def test_session_operations_run_in_arrival_order_even_when_one_is_aborted():
    hooks = BlockingHooks()
    scheduler = SessionScheduler(hooks, max_concurrency=3)
    compute_registered(scheduler, session_stage_payload("open", "open"))
    payloads = [
        session_stage_payload(request_id, "append")
        for request_id in ("first", "second", "third")
    ]
    for payload in payloads:
        scheduler.inbox.put(IncomingMessage(payload.request_id, "new_request", payload))
    messages = [scheduler.inbox.get_nowait() for _ in payloads]
    threads = [threading.Thread(target=scheduler.compute, args=(messages[0].data,))]
    threads[0].start()
    assert hooks.entered.wait(5)
    scheduler.abort("second")
    assert scheduler.consume_if_aborted("second")
    threads.append(threading.Thread(target=scheduler.compute, args=(messages[2].data,)))
    threads[1].start()
    threads[1].join(0.2)
    assert threads[1].is_alive(), "third operation ran before the first finished"
    hooks.release.set()
    for thread in threads:
        thread.join(5)
    assert not any(thread.is_alive() for thread in threads)
    order = []
    while not hooks.events.empty():
        event = hooks.events.get_nowait()
        if event[0] == "append":
            order.append(event[1])
    assert order == ["first", "third"]
    assert not scheduler.cursors_by_session and not scheduler.arrivals_by_request_id


def test_later_operation_does_not_start_before_the_session_lock() -> None:
    hooks = BlockingHooks()
    scheduler = SessionScheduler(hooks, max_concurrency=3)
    started: list[tuple[str, Literal["open", "append", "close"]]] = []
    compute_session = scheduler.compute_session

    def record_compute_session(
        payload: StagePayload, session_operation: SessionOperation
    ) -> StagePayload:
        started.append((payload.request_id, session_operation.operation))
        return compute_session(payload, session_operation)

    scheduler.compute_session = record_compute_session
    compute_registered(scheduler, session_stage_payload("open", "open"))
    payloads = [
        session_stage_payload(request_id, "append")
        for request_id in ("first", "second", "third")
    ]
    for payload in payloads:
        scheduler.inbox.put(IncomingMessage(payload.request_id, "new_request", payload))
    messages = [scheduler.inbox.get_nowait() for _ in payloads]
    first = threading.Thread(target=scheduler.compute, args=(messages[0].data,))
    first.start()
    assert hooks.entered.wait(5)
    scheduler.abort("second")
    assert scheduler.consume_if_aborted("second")
    third = threading.Thread(target=scheduler.compute, args=(messages[2].data,))
    third.start()
    third.join(0.2)
    assert started == [("open", "open"), ("first", "append")]
    hooks.release.set()
    for thread in (first, third):
        thread.join(5)
    assert started == [("open", "open"), ("first", "append"), ("third", "append")]


def test_close_runs_after_its_request_is_aborted():

    events = queue.Queue()
    scheduler = SessionScheduler(Hooks("source", events))

    def message(rid, operation):
        request = OmniRequest(
            None, metadata=operation_metadata(operation, SessionIdentity("s"))
        )
        return IncomingMessage(rid, "new_request", StagePayload(rid, request, {}))

    scheduler.inbox.put(message("open", "open"))
    scheduler.inbox.put(message("late", "close"))
    scheduler.abort("late")
    worker = threading.Thread(target=scheduler.start)
    worker.start()
    try:
        assert events.get(timeout=5)[0] == "open"
        assert events.get(timeout=5)[0] == "close"
    finally:
        scheduler.stop()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert not scheduler.open_sessions and not scheduler.arrivals_by_request_id


def test_operation_finished_by_abort_before_running_does_not_wait():

    class AppendHooks(Hooks):
        def append(
            self,
            chunk: TimedChunk,
            payload: StagePayload,
            context: SessionContext,
        ) -> StagePayload:
            state = self.states[context.session_identity]
            self.events.put(("append", self.name, state.session_id))
            return payload

    events = queue.Queue()
    scheduler = SessionScheduler(AppendHooks("source", events), max_concurrency=2)

    compute_registered(scheduler, session_stage_payload("open", "open"))
    payload = session_stage_payload("late", "append")
    scheduler.inbox.put(IncomingMessage("late", "new_request", payload))
    message = scheduler.inbox.get_nowait()
    # Note (Junnan Li): A request-level abort consumed the ticket first; the command still runs.
    scheduler.abort("late")
    assert scheduler.consume_if_aborted("late")
    errors = []

    def run():
        try:
            scheduler.compute(message.data)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), "command waited for a number that was already served"
    assert not errors, errors
    seen = []
    while not events.empty():
        seen.append(events.get_nowait()[0])
    assert seen == ["open", "append"]
    assert not scheduler.cursors_by_session and not scheduler.arrivals_by_request_id


def test_context_exhaustion_releases_append_unit() -> None:
    request_id = "append-1"
    identity = SessionIdentity("session-1")
    request = SimpleNamespace(
        rid=request_id,
        origin_input_ids=[1] * 8191,
        origin_input_ids_unpadded=[1] * 8191,
        sampling_params=SimpleNamespace(max_new_tokens=1),
    )
    unit = SessionUnit(
        request_id=request_id,
        session_identity=identity,
        stages=("thinker",),
        chunk=TimedChunk("audio", 0, 1000, 0, b""),
        session_request=request,
    )
    scheduler = Mock(spec=OmniScheduler)
    scheduler.scheduler_thread_id = None
    scheduler.request_admission_lock = threading.RLock()
    scheduler.deferred_request_payloads = {}
    scheduler.pending_request_builds = {}
    scheduler.pending_request_admissions = {}
    scheduler.pending_stream_ingress = {}
    scheduler.aborted_request_ids = set()
    scheduler.aborted_request_id_order = deque()
    scheduler.dirty_deferred_request_ids = set()
    scheduler.first_emit_done = set()
    scheduler.prefill_start_done = set()
    scheduler.prefill_end_done = set()
    scheduler.backlogged_request_build_payloads = deque()
    scheduler.waiting_queue = []
    scheduler.mark_running_request_aborted.return_value = False
    scheduler.mark_request_finished_immediately.return_value = []
    scheduler.init_req_max_new_tokens = Mock()
    scheduler.max_req_input_len = 8190
    scheduler.server_args = SimpleNamespace(context_length=8192)
    scheduler.active_session_unit = MethodType(
        OmniScheduler.active_session_unit, scheduler
    )
    scheduler.normalize_req_token_arrays = OmniScheduler.normalize_req_token_arrays
    scheduler.prepare_request_limits = MethodType(
        OmniScheduler.prepare_request_limits, scheduler
    )
    scheduler.abort = MethodType(OmniScheduler.abort, scheduler)
    scheduler.session_controller = Mock()
    bridge = ARSessionBridge(scheduler, Mock())
    bridge.sessions[identity.id] = BridgeSession(session_identity=identity, unit=unit)
    bridge.units_by_request_id[request_id] = unit
    bridge.create_session_request = Mock()
    scheduler.session_bridge = bridge

    OmniScheduler.enqueue_built_request(
        scheduler,
        SimpleNamespace(request_id=request_id),
        False,
        SimpleNamespace(req=request, enforce_request_limits=True),
    )

    error = scheduler.emit_request_error.call_args.args[1]
    assert isinstance(error, ContextExhaustedError)
    assert str(error).startswith("context_exhausted:")
    assert "8192" in str(error)
    assert not bridge.units_by_request_id
    assert bridge.sessions[identity.id].unit is None
    scheduler.session_controller.get.return_value.abort_req.assert_called_once()
