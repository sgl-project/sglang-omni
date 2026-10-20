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
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

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
    ARSessionAdapter,
    ARSessionBridge,
    BridgeSession,
    SessionUnit,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from tests.unit_test.fixtures.session_outputs import control_chunk
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


def operation_payload(
    request_id: str,
    operation: str,
    session_identity: SessionIdentity,
    chunk: TimedChunk | None = None,
    *,
    should_preempt: bool = False,
) -> StagePayload:
    metadata = operation_metadata(
        operation, session_identity, chunk, should_preempt=should_preempt
    )
    return StagePayload(request_id, OmniRequest(None, metadata=metadata), {})


def test_a_read_attaches_on_arrival_streams_emitted_chunks_and_is_answered_by_the_close():
    emitters = {}

    class EmittingHooks(Hooks):
        def attach_output_stream(self, session_identity: SessionIdentity, emit) -> None:
            emitters[session_identity] = emit

    scheduler = SessionScheduler(EmittingHooks("source", queue.Queue()))
    identity = SessionIdentity("one")
    chunk = control_chunk("speech.stopped")
    compute_registered(scheduler, operation_payload("open", "open", identity))
    with pytest.raises(RuntimeError, match="no output reader"):
        emitters[identity](chunk)
    read = operation_payload("read", "read", identity)
    scheduler.inbox.put(IncomingMessage("read", "new_request", read))
    assert scheduler.inbox.empty() and not scheduler.arrivals_by_request_id
    emitters[identity](chunk)
    streamed = scheduler.outbox.get_nowait()
    assert (streamed.request_id, streamed.type) == ("read", "stream")
    assert TimedChunk.from_dict(streamed.data) == chunk
    with pytest.raises(ValueError, match="already has an output reader"):
        compute_registered(scheduler, operation_payload("again", "read", identity))
    compute_registered(scheduler, operation_payload("close", "close", identity))
    answered = scheduler.outbox.get_nowait()
    assert (answered.request_id, answered.type) == ("read", "result")
    assert answered.data.data == {"closed": True}
    with pytest.raises(RuntimeError, match="no output reader"):
        emitters[identity](chunk)


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


@pytest.mark.parametrize("size", [255, 256, 65535, 65536])
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


class PreemptibleHooks(Hooks):
    """An append that runs until its cancel event is set; controls are recorded."""

    def __init__(self) -> None:
        super().__init__("source", queue.Queue())
        self.entered = threading.Event()

    def append(
        self,
        chunk: TimedChunk,
        payload: StagePayload,
        context: SessionContext,
    ) -> StagePayload:
        self.entered.set()
        was_cancelled = context.cancelled.wait(5)
        self.events.put(("append", payload.request_id, was_cancelled))
        return payload

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        self.events.put(("control", session_identity.id, event.payload["type"]))


def control_stage_payload(request_id: str, *, should_preempt: bool) -> StagePayload:
    return operation_payload(
        request_id,
        "control",
        SessionIdentity("session"),
        control_chunk("interrupt", turn=1),
        should_preempt=should_preempt,
    )


def start_compute(
    scheduler: SessionScheduler, payload: StagePayload
) -> threading.Thread:
    """Register one operation and compute it on its own thread."""
    scheduler.inbox.put(IncomingMessage(payload.request_id, "new_request", payload))
    thread = threading.Thread(
        target=scheduler.compute, args=(scheduler.inbox.get_nowait().data,)
    )
    thread.start()
    return thread


def test_control_reaches_the_hooks_in_arrival_order_and_preempt_cancels_the_in_flight_append():
    hooks = PreemptibleHooks()
    scheduler = SessionScheduler(hooks, max_concurrency=2)
    compute_registered(scheduler, session_stage_payload("open", "open"))
    unit = start_compute(scheduler, session_stage_payload("unit", "append"))
    assert hooks.entered.wait(5)
    stop = start_compute(scheduler, control_stage_payload("stop", should_preempt=True))
    for thread in (unit, stop):
        thread.join(5)
    assert not any(thread.is_alive() for thread in (unit, stop))
    recorded = [hooks.events.get_nowait() for _ in range(hooks.events.qsize())]
    assert recorded[1:] == [
        ("append", "unit", True),
        ("control", "session", "interrupt"),
    ]
    assert not scheduler.append_cancel_events
    quiet = compute_registered(
        scheduler, control_stage_payload("quiet", should_preempt=False)
    )
    assert quiet.data == {"controlled": True}
    assert hooks.events.get_nowait() == ("control", "session", "interrupt")


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
    scheduler.model_config = SimpleNamespace(context_len=8192)
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


def test_an_ar_stage_holds_the_read_takes_a_control_and_answers_the_read_at_close():
    identity = SessionIdentity("session-1")
    scheduler = Mock(spec=OmniScheduler)
    scheduler.max_running_requests = 4
    scheduler.session_controller = Mock()
    scheduler.session_controller.open.return_value = SimpleNamespace(success=True)
    scheduler.session_controller.get.return_value = None
    scheduler.tree_cache = SimpleNamespace(slots={})
    scheduler.outbox = queue.Queue()
    scheduler.aborted_request_ids = set()
    adapter = Mock()
    scheduler.session_bridge = ARSessionBridge(scheduler, adapter)
    event = control_chunk("interrupt")

    def route(request_id: str, operation: str, **fields) -> list[tuple[str, dict]]:
        payload = operation_payload(request_id, operation, identity, **fields)
        assert not OmniScheduler.route_session_input(scheduler, payload)
        answered = []
        while not scheduler.outbox.empty():
            message = scheduler.outbox.get_nowait()
            answered.append((message.request_id, message.data.data))
        return answered

    assert route("open", "open") == [("open", {"opened": True})]
    assert route("read", "read") == []
    assert route("control", "control", chunk=event) == [
        ("control", {"controlled": True})
    ]
    adapter.control.assert_called_once_with(identity, event)
    assert route("preempt", "control", chunk=event, should_preempt=True) == [
        ("preempt", {"controlled": True})
    ]
    assert route("close", "close") == [
        ("read", {"closed": True}),
        ("close", {"closed": True}),
    ]


@pytest.mark.parametrize("should_preempt", [False, True])
def test_ar_session_control_preserves_session(should_preempt: bool) -> None:
    identity = SessionIdentity("voicechat")
    event = TimedChunk("control", 0, 0, 0, {"responses": []}, format="tool_response")
    scheduler = Mock(spec=OmniScheduler)
    scheduler.aborted_request_ids = set()
    scheduler.outbox = queue.Queue()
    scheduler.session_controller = Mock()
    adapter = ARSessionAdapter()
    adapter.control = Mock(wraps=adapter.control)
    bridge = ARSessionBridge(scheduler, adapter)
    scheduler.session_bridge = bridge
    session = BridgeSession(session_identity=identity)
    bridge.sessions[identity.id] = session
    bridge.accept(
        StagePayload("append", OmniRequest(None), {}),
        SessionOperation("append", identity, ("thinker",), event),
    )
    payload = StagePayload(
        "control",
        OmniRequest(
            None,
            metadata=operation_metadata(
                "control", identity, event, should_preempt=should_preempt
            ),
        ),
        {},
    )
    assert not OmniScheduler.route_session_input(scheduler, payload)
    assert scheduler.outbox.get_nowait().data.data == {"controlled": True}
    adapter.control.assert_called_once_with(identity, event)
    assert bridge.sessions[identity.id] is session
    assert (session.unit is None) == should_preempt
    assert scheduler.abort.call_count == int(should_preempt)
    for invalid_identity in (
        SessionIdentity("missing"),
        SessionIdentity(identity.id, 2),
    ):
        with pytest.raises(ValueError, match="unknown|stale"):
            bridge.apply_operation(
                payload,
                SessionOperation("control", invalid_identity, ("thinker",), event),
            )


def test_ar_bridge_binds_the_native_request_and_hands_an_aborted_unit_to_the_adapter() -> (
    None
):
    identity = SessionIdentity("native")
    adapter = ARSessionAdapter()
    adapter.bind_request = Mock()
    order: list[str] = []
    adapter.on_abort = Mock(side_effect=lambda identity, unit: order.append("model"))
    scheduler = Mock()
    scheduler.model_config.vocab_size = 100
    scheduler.max_req_len = 100
    native = scheduler.session_controller.get.return_value
    native.session_id = identity.id
    native.req_nodes = {}
    native.abort_req.side_effect = lambda: order.append("native")
    native_request = Mock(origin_input_ids=[10, 20, 30], to_finish=None)
    native.create_req.return_value = native_request
    bridge = ARSessionBridge(scheduler, adapter)
    bridge.sessions[identity.id] = BridgeSession(session_identity=identity)
    payload = StagePayload("unit", OmniRequest(None), {})
    unit = bridge.accept(
        payload,
        SessionOperation(
            "append", identity, ("thinker",), TimedChunk("text", 0, 0, 0, {})
        ),
    )
    request = Req("unit", "", [30], SamplingParams(max_new_tokens=1), vocab_size=100)
    request.tokenizer = Mock()
    request_data = SGLangARRequestData(req=request)
    bridge.create_session_request(payload, request_data)
    assert request_data.req is native_request
    assert unit.native_offset == 2 and unit.prompt_length == 1
    adapter.bind_request.assert_called_once_with(identity, request_data, unit)
    bridge.release_append_unit("unit")
    adapter.on_abort.assert_called_once_with(identity, unit)
    assert order == ["model", "native"]
    assert bridge.sessions[identity.id].unit is None
