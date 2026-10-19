"""State kept for one long-lived pipeline connection.

Note (chenyang): These definitions could be hard to understand.

A stage is one segment of the pipeline that a unit passes through. Just as
we defined for Qwen3 Omni, thinker, talker, and the completion head are
stages. A stage receives a request, runs that request through its scheduler,
and routes the result to the next stage. When the session starts, the
coordinator picks one fixed worker for each stage and stores those owners on
the session. After that, every unit of the session runs only on those
workers.

A session is one long-lived connection. A session ID identifies it. State
stays across the many short requests/units of that connection, so a later
unit reuses what earlier units left behind: perception state, AR state such as
the KV cache and token history, and encoder or codec state. When a unit
finishes, that unit's own resources are released. The session state stays.
Closing the session releases that state and returns the session's capacity.

An operation is one action sent for a session: open, append, or close.
append is the unit above, and it follows the stages from one to the next.
open and close each target one stage. open visits the stages from upstream
to downstream and creates that stage's session state. close visits them from
downstream to upstream and releases the state. On one stage, open, append,
and close share one arrival order.

A cursor is one stage's record of the operations that have arrived for one
session. It keeps them in arrival order, and it keeps the place that may
start next. Each stage has its own cursor for that session. An operation
that enters the inbox takes the next place. The stage runs it only when
that place is due. Chunks emitted during append go back to the caller
through the coordinator.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Literal, Protocol

from sglang_omni.admission import QueueFullError
from sglang_omni.profiler.duplex_events import (
    capture_session_stage_start,
    emit_session_stage_started,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    ResourceUsage,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
    find_session_operation,
)
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

DEFAULT_MAX_OPEN_SESSIONS = 64
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_STATE_BYTES_PER_SESSION = 1 << 30


class ChunkEmitter(Protocol):
    def __call__(self, chunk: TimedChunk) -> None: ...


class OperationRegistrar(Protocol):
    def __call__(self, message: IncomingMessage) -> None: ...


class StageCompute(Protocol):
    def __call__(self, payload: StagePayload) -> StagePayload: ...


@dataclass(kw_only=True)
class SessionContext:
    session_identity: SessionIdentity
    cancelled: threading.Event
    emit: ChunkEmitter


class SessionHooks:
    """A stage's callbacks for one session: open, append, close, and usage."""

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        raise NotImplementedError

    def append(
        self,
        chunk: TimedChunk,
        payload: StagePayload,
        context: SessionContext,
    ) -> StagePayload:
        raise NotImplementedError

    def close(self, session_identity: SessionIdentity) -> None:
        raise NotImplementedError

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return ResourceUsage()

    def warm_up_serving_thread(self) -> None:
        """Build per-thread state, such as cuDNN convolution plans, on the scheduler thread before the stage reports ready."""


@dataclass(frozen=True, kw_only=True)
class SessionAppend:
    chunk: TimedChunk
    payload: StagePayload
    context: SessionContext


class BatchedSessionHooks(SessionHooks):
    """Hooks that run one unit of each ready session in one call.

    gather_window_ms is how long an idle stage waits for more appends when
    fewer sessions are ready than open; 0 runs with what is ready.
    """

    gather_window_ms: float = 0.0

    def append_batch(self, appends: list[SessionAppend]) -> list[StagePayload]:
        """Run one unit for each of several distinct sessions, in order."""
        raise NotImplementedError

    def gathers_peers(self, payload: StagePayload, chunk: TimedChunk) -> bool:
        """Whether waiting for other sessions' appends can batch this one's work."""
        return True


@dataclass(kw_only=True)
class StageSession:
    is_open: bool = False
    is_failed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(kw_only=True)
class AppendOperationState:
    session_identity: SessionIdentity
    cancelled: threading.Event
    is_executing: bool = False


@dataclass(frozen=True, kw_only=True)
class OperationArrival:
    """Arrival position of one accepted operation inside its session."""

    session_identity: SessionIdentity
    operation: Literal["open", "append", "close"]
    sequence: int


@dataclass(kw_only=True)
class SessionOperationCursor:
    """Arrival cursor whose failure fence can outlive a not-yet-open session."""

    next_sequence: int = 0
    runnable_sequence: int = 0
    completed_sequences: set[int] = field(default_factory=set)
    is_failed: bool = False


class SessionInbox(queue.Queue[IncomingMessage]):
    def __init__(self, register: OperationRegistrar) -> None:
        super().__init__()
        self.register = register

    def put(
        self,
        message: IncomingMessage,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        if message.type == "new_request":
            self.register(message)
        else:
            pass
        super().put(message, block, timeout)


class SessionScheduler(SimpleScheduler[StagePayload, StagePayload]):
    """Opt-in scheduler for persistent hooks, with bounded stage admission."""

    def __init__(
        self,
        session_hooks: SessionHooks,
        *,
        compute_fn: StageCompute | None = None,
        max_open_sessions: int = DEFAULT_MAX_OPEN_SESSIONS,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_state_bytes_per_session: int = DEFAULT_MAX_STATE_BYTES_PER_SESSION,
    ) -> None:
        self.session_hooks = session_hooks
        self.request_compute = compute_fn
        self.max_open_sessions = max_open_sessions
        self.max_state_bytes_per_session = max_state_bytes_per_session
        self.open_sessions: dict[SessionIdentity, StageSession] = {}
        self.append_operation_states: dict[str, AppendOperationState] = {}
        self.session_table_lock = threading.Lock()
        self.is_shutting_down = False
        self.arrivals_by_request_id: dict[str, OperationArrival] = {}
        self.cursors_by_session: dict[SessionIdentity, SessionOperationCursor] = {}
        self.operation_finished: threading.Condition = threading.Condition(
            self.session_table_lock
        )
        if isinstance(session_hooks, BatchedSessionHooks):
            # Note (Junnan Li): One hook call runs every session whose append is ready, so the batch replaces the concurrent workers.
            batch_compute_fn = self.compute_batch
            max_batch_size = max_open_sessions
            max_concurrency = 1
            gather_window_seconds = session_hooks.gather_window_ms / 1000
        else:
            batch_compute_fn = None
            max_batch_size = 1
            # note (Junnan Li): Each call runs one operation, so there is nothing to gather.
            gather_window_seconds = 0.0
        super().__init__(
            self.compute,
            batch_compute_fn=batch_compute_fn,
            max_batch_size=max_batch_size,
            max_concurrency=max_concurrency,
            abort_callback=self.cancel_operation,
            shutdown_callback=self.release_sessions_on_scheduler_stop,
        )
        self.inbox = SessionInbox(self.register_operation)
        self.gather_window_seconds = gather_window_seconds
        # note (Junnan Li): Operations that arrived while the previous call ran are batched as they are; only a call that starts from an idle stage waits.
        self.is_backlogged = False
        self.batch_gathers_peers = True

    def warm_up_serving_thread(self) -> None:
        self.session_hooks.warm_up_serving_thread()

    def register_operation(self, message: IncomingMessage) -> None:
        try:
            session_operation = find_session_operation(message.data.request.metadata)
        except ValueError:
            # Note (Junnan Li): put() runs on the stage loop; compute reports the malformed operation.
            # TODO (chenyang): This error handling is a bit rough here.
            return
        if session_operation is None:
            return
        else:
            session_identity = session_operation.session_identity
            with self.session_table_lock:
                cursor = self.cursors_by_session.setdefault(
                    session_identity, SessionOperationCursor()
                )
                self.arrivals_by_request_id[message.request_id] = OperationArrival(
                    session_identity=session_identity,
                    operation=session_operation.operation,
                    sequence=cursor.next_sequence,
                )
                cursor.next_sequence += 1

    def finish_operation(self, request_id: str) -> None:
        with self.operation_finished:
            self.finish_operation_locked(request_id)

    def finish_operation_locked(self, request_id: str) -> None:
        """Retire an arrival while operation_finished holds session_table_lock."""
        arrival = self.arrivals_by_request_id.pop(request_id, None)
        if arrival is None:
            return
        else:
            cursor = self.cursors_by_session.get(arrival.session_identity)
            if cursor is None:
                return
            else:
                # Note (Junnan Li): An aborted operation can finish before its predecessors ran.
                cursor.completed_sequences.add(arrival.sequence)
                while cursor.runnable_sequence in cursor.completed_sequences:
                    cursor.completed_sequences.discard(cursor.runnable_sequence)
                    cursor.runnable_sequence += 1
                if cursor.runnable_sequence == cursor.next_sequence:
                    self.cursors_by_session.pop(arrival.session_identity)
                else:
                    pass
                self.operation_finished.notify_all()

    def mark_session_failed_locked(self, session_identity: SessionIdentity) -> None:
        """Keep the failure fence on the object that outlives this operation."""
        session = self.open_sessions.get(session_identity)
        if session is not None:
            if not session.is_failed:
                session.is_failed = True
                self.operation_finished.notify_all()
            else:
                pass
        else:
            cursor = self.cursors_by_session.get(session_identity)
            if cursor is not None and not cursor.is_failed:
                cursor.is_failed = True
                self.operation_finished.notify_all()
            else:
                pass

    def fence_aborted_append_locked(self, request_id: str) -> bool:
        """Fence an accepted append until local settlement removes its arrival.

        The active operation record is the conservative commit boundary. A stage
        commit is not treated as end-to-end success while settlement is pending;
        an abort observed during that interval always fences the session. Once
        settlement removes both records, a later abort is stale for this stage.
        """
        append_state = self.append_operation_states.get(request_id)
        if append_state is not None:
            append_state.cancelled.set()
            self.mark_session_failed_locked(append_state.session_identity)
            return not append_state.is_executing
        else:
            arrival = self.arrivals_by_request_id.get(request_id)
            if arrival is not None and arrival.operation == "append":
                self.mark_session_failed_locked(arrival.session_identity)
                return True
            else:
                return False

    def mark_session_failed(self, session_identity: SessionIdentity) -> None:
        with self.operation_finished:
            self.mark_session_failed_locked(session_identity)

    def begin_append_execution(
        self, request_id: str, session_identity: SessionIdentity
    ) -> bool:
        with self.operation_finished:
            append_state = self.append_operation_states.get(request_id)
            if append_state is None:
                return False
            else:
                pass
            session = self.open_sessions.get(session_identity)
            if append_state.cancelled.is_set() or session is None or session.is_failed:
                self.mark_session_failed_locked(session_identity)
                return False
            else:
                append_state.is_executing = True
                return True

    def validate_append_completion(self, request_id: str) -> None:
        """Validate append completion while keeping its fence active."""
        with self.operation_finished:
            append_state = self.append_operation_states.get(request_id)
            shutdown_owns_cleanup = False
            if append_state is None:
                raise RuntimeError("append operation is no longer active")
            else:
                session = self.open_sessions.get(append_state.session_identity)
                shutdown_owns_cleanup = (
                    self.is_shutting_down
                    and session is not None
                    and not session.is_failed
                )
            # Note (chenye): Shutdown already owns the terminal cleanup, so an
            # active hook that finishes normally must not turn stop into a request error.
            if append_state.cancelled.is_set() and not shutdown_owns_cleanup:
                self.mark_session_failed_locked(append_state.session_identity)
                raise RuntimeError("session append was cancelled")
            else:
                pass

    def consume_if_aborted(self, request_id: str) -> bool:
        aborted = super().consume_if_aborted(request_id)
        should_finish = False
        with self.operation_finished:
            arrival = self.arrivals_by_request_id.get(request_id)
            is_close_operation = arrival is not None and arrival.operation == "close"
            is_open_operation = arrival is not None and arrival.operation == "open"
            append_state = self.append_operation_states.get(request_id)
            is_append_operation = append_state is not None or (
                arrival is not None and arrival.operation == "append"
            )
            if aborted and is_append_operation:
                should_finish = self.fence_aborted_append_locked(request_id)
            elif aborted and is_open_operation:
                self.mark_session_failed_locked(arrival.session_identity)
                should_finish = True
            else:
                pass
            if aborted and not is_close_operation and should_finish:
                self.finish_operation_locked(request_id)
            else:
                pass
        if aborted and is_close_operation:
            # Note (Junnan Li): A timed-out close is request-aborted; skipping it would leak the state.
            return False
        else:
            return aborted

    def compute(self, payload: StagePayload) -> StagePayload:
        session_operation = find_session_operation(payload.request.metadata)
        if session_operation is None:
            if self.request_compute is None:
                raise ValueError("this stage has no compute_fn for ordinary requests")
            else:
                return self.request_compute(payload)
        else:
            session_identity = session_operation.session_identity
            operation_failed = False
            session_was_failed = False
            try:
                with self.operation_finished:
                    # Note (Junnan Li): A request-level abort may already have consumed the arrival.
                    arrival = self.arrivals_by_request_id.get(payload.request_id)
                    if arrival is not None:
                        session_identity = arrival.session_identity
                        sequence = arrival.sequence
                        self.operation_finished.wait_for(
                            lambda: (
                                (
                                    cursor := self.cursors_by_session.get(
                                        session_identity
                                    )
                                )
                                is None
                                or cursor.runnable_sequence >= sequence
                            )
                        )
                    else:
                        session_identity = session_operation.session_identity
                    session = self.open_sessions.get(session_identity)
                    session_was_failed = session is not None and session.is_failed
                return self.compute_session(payload, session_operation)
            except BaseException:
                operation_failed = True
                raise
            finally:
                self.settle_operation(
                    payload.request_id,
                    session_identity,
                    session_operation.operation,
                    operation_failed,
                    session_was_failed,
                )

    def settle_operation(
        self,
        request_id: str,
        session_identity: SessionIdentity,
        operation: Literal["open", "append", "close"],
        operation_failed: bool = False,
        session_was_failed: bool = False,
    ) -> None:
        # Note (chenye): An active append remains fenced through local settlement,
        # so a stage commit cannot be mistaken for end-to-end success.
        with self.session_table_lock:
            session = self.open_sessions.get(session_identity)
        if session is not None:
            with session.lock:
                with self.operation_finished:
                    append_was_started = request_id in self.append_operation_states
                    arrival = self.arrivals_by_request_id.get(request_id)
                    # note(chenye): A later append that only observes a terminal session
                    # does not own cleanup.
                    append_requires_cleanup = operation == "append" and (
                        append_was_started
                        or not session_was_failed
                        or (arrival is None and not self.is_shutting_down)
                    )
                    should_close = (
                        self.open_sessions.get(session_identity) is session
                        and (self.is_shutting_down or session.is_failed)
                        and (
                            (operation == "open" and not operation_failed)
                            or append_requires_cleanup
                        )
                    )
                    self.finish_operation_locked(request_id)
                    self.append_operation_states.pop(request_id, None)
                if should_close:
                    self.close_session(session_identity, session)
                else:
                    pass
        else:
            with self.operation_finished:
                self.finish_operation_locked(request_id)
                self.append_operation_states.pop(request_id, None)

    def run_batch(
        self, batch: list[IncomingMessage], loop: asyncio.AbstractEventLoop
    ) -> None:
        """Emit each operation's own result or error, so a failed unit fails only itself."""
        if self.batch_fn is None:
            super().run_batch(batch, loop)
            return
        else:
            pass
        batch = [
            message
            for message in batch
            if not self.consume_if_aborted(message.request_id)
        ]
        try:
            results = self.compute_batch([message.data for message in batch])
        finally:
            # note (ruoyu): A batch that was not held cost no window, so work queued behind it is not a backlog.
            self.is_backlogged = self.batch_gathers_peers and not self.inbox.empty()
        for message, result in zip(batch, results, strict=True):
            if self.consume_if_aborted(message.request_id):
                continue
            elif isinstance(result, Exception):
                self.emit_error(message.request_id, result, self.outbox)
            else:
                self.emit_result(message.request_id, result, self.outbox)

    def collect_batch(self, first_msg: IncomingMessage) -> list[IncomingMessage]:
        """Take every ready operation; from an idle stage, wait up to the hooks' gather window for more appends.

        The wait ends at the window, once the batch holds an append of every
        open session, or when an operation other than an append is in the
        batch. Messages keep their inbox order, so the cursor contract above
        is untouched.
        """
        batch = super().collect_batch(first_msg)
        self.batch_gathers_peers = self.gather_window_seconds == 0 or any(
            self.gathers_peers(message) for message in batch
        )
        if self.gather_window_seconds == 0 or self.is_backlogged:
            return batch
        elif not self.batch_gathers_peers:
            return batch
        else:
            pass
        deadline = time.monotonic() + self.gather_window_seconds
        while len(batch) < self.max_batch_size:
            with self.session_table_lock:
                arrivals = [
                    self.arrivals_by_request_id.get(message.request_id)
                    for message in batch
                ]
                open_session_count = len(self.open_sessions)
            if any(
                arrival is None or arrival.operation != "append" for arrival in arrivals
            ):
                break
            elif (
                len({arrival.session_identity for arrival in arrivals})
                >= open_session_count
            ):
                break
            else:
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            else:
                pass
            try:
                message = self.inbox.get(timeout=remaining)
            except queue.Empty:
                break
            if message.type == "new_request":
                batch.append(message)
            else:
                self.pending_messages.append(message)
        return batch

    def gathers_peers(self, message: IncomingMessage) -> bool:
        """Whether holding the batch for more appends can batch this message's work."""
        with self.session_table_lock:
            arrival = self.arrivals_by_request_id.get(message.request_id)
        if arrival is None or arrival.operation != "append":
            return False
        else:
            # note (ruoyu): An accepted append parsed this metadata in register_operation, so it cannot raise here.
            session_operation = find_session_operation(message.data.request.metadata)
            assert isinstance(self.session_hooks, BatchedSessionHooks)
            assert session_operation is not None and session_operation.chunk is not None
            return self.session_hooks.gathers_peers(
                message.data, session_operation.chunk
            )

    def compute_batch(
        self, payloads: list[StagePayload]
    ) -> list[StagePayload | Exception]:
        """Run operations in arrival order per session, sharing hooks across sessions.

        Each round takes the first remaining operation of every session. The
        round's appends go to the hooks in one call; its opens, closes and
        ordinary requests run one at a time.
        """
        results: dict[str, StagePayload | Exception] = {}
        pending = list(payloads)
        while pending:
            # Note (Junnan Li): The inbox is FIFO and batches run one at a time, so earlier operations of a session finished before this batch or come earlier in it.
            round_sessions: set[SessionIdentity] = set()
            appends: list[tuple[StagePayload, SessionOperation]] = []
            alone: list[StagePayload] = []
            later: list[StagePayload] = []
            for payload in pending:
                with self.session_table_lock:
                    arrival = self.arrivals_by_request_id.get(payload.request_id)
                if arrival is None:
                    alone.append(payload)
                elif arrival.session_identity in round_sessions:
                    later.append(payload)
                else:
                    round_sessions.add(arrival.session_identity)
                    session_operation = SessionOperation.from_dict(
                        payload.request.metadata[SESSION_METADATA_KEY]
                    )
                    if session_operation.operation == "append":
                        appends.append((payload, session_operation))
                    else:
                        alone.append(payload)
            for payload in alone:
                try:
                    results[payload.request_id] = self.compute(payload)
                except Exception as exc:
                    results[payload.request_id] = exc
            if appends:
                results.update(self.compute_appends(appends))
            else:
                pass
            pending = later
        return [results[payload.request_id] for payload in payloads]

    def cancel_operation(self, request_id: str) -> None:
        with self.operation_finished:
            # Note (chenye): Direct compute can race the abort callback before ticket retirement.
            arrival = self.arrivals_by_request_id.get(request_id)
            if arrival is not None and arrival.operation == "open":
                self.mark_session_failed_locked(arrival.session_identity)
                self.finish_operation_locked(request_id)
            elif self.fence_aborted_append_locked(request_id):
                self.finish_operation_locked(request_id)
            else:
                pass

    def release_sessions_on_scheduler_stop(self) -> None:
        """Release every session still open when this stage's scheduler stops.

        Note (chenyang):

        release_sessions_on_scheduler_stop is used when a stage/scheduler stops.
        In this time, all the sessions that are still open should be closed.
        """
        with self.session_table_lock:
            self.is_shutting_down = True
            for append_state in self.append_operation_states.values():
                append_state.cancelled.set()
            open_sessions = list(self.open_sessions.items())
        errors: list[Exception] = []
        for session_identity, session in open_sessions:
            if session.lock.acquire(blocking=False):
                try:
                    self.close_session(session_identity, session)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    session.lock.release()
            else:
                pass
        with self.operation_finished:
            # note(chenye): Stop retires queued tickets and lets active hooks
            # keep ownership until their settlement closes the session.
            for request_id in tuple(self.arrivals_by_request_id):
                self.finish_operation_locked(request_id)
        if errors:
            raise RuntimeError("session shutdown cleanup failed") from errors[0]
        else:
            pass

    def close_session(
        self, session_identity: SessionIdentity, session: StageSession
    ) -> None:
        """Release one session's state on this stage.

        Note (chenyang):

        close_session is used when a session is closed. The other sessions
        on the stage/scheduler should not be affected.
        """
        if session.is_open:
            try:
                self.session_hooks.close(session_identity)
            except BaseException:
                with self.operation_finished:
                    self.mark_session_failed_locked(session_identity)
                raise
            else:
                session.is_open = False
        else:
            pass
        with self.session_table_lock:
            self.open_sessions.pop(session_identity, None)

    def check_usage(self, session_identity: SessionIdentity) -> None:
        """Fail only the session whose state outgrew its own budget.

        The check runs after the open or append that grew the state, so a
        session may exceed its budget by that one step before it fails; other
        sessions are not affected.
        """
        if (
            self.session_hooks.usage(session_identity).bytes
            > self.max_state_bytes_per_session
        ):
            raise QueueFullError()
        else:
            pass

    def open_session(
        self,
        request_id: str,
        session_identity: SessionIdentity,
        request: OmniRequest,
    ) -> None:
        session = StageSession()
        session.lock.acquire()
        with self.session_table_lock:
            cursor = self.cursors_by_session.get(session_identity)
            if self.is_shutting_down:
                session.lock.release()
                raise RuntimeError("session scheduler is stopping")
            elif self.is_aborted(request_id):
                session.lock.release()
                raise RuntimeError("session open was aborted before execution")
            elif cursor is not None and cursor.is_failed:
                session.lock.release()
                raise RuntimeError("session is terminal before open")
            elif session_identity in self.open_sessions:
                session.lock.release()
                raise ValueError("session already opened")
            elif len(self.open_sessions) >= self.max_open_sessions:
                session.lock.release()
                raise QueueFullError()
            else:
                self.open_sessions[session_identity] = session
        try:
            self.session_hooks.open(session_identity, request)
            session.is_open = True
            self.check_usage(session_identity)
            if self.is_shutting_down:
                raise RuntimeError("session scheduler is stopping")
            else:
                pass
        except BaseException:
            self.close_session(session_identity, session)
            raise
        finally:
            session.lock.release()

    def compute_session(
        self, payload: StagePayload, session_operation: SessionOperation
    ) -> StagePayload:
        session_identity = session_operation.session_identity
        operation = session_operation.operation
        if operation == "open":
            self.open_session(payload.request_id, session_identity, payload.request)
            payload.data = {"opened": True}
            return payload
        else:
            with self.session_table_lock:
                session = self.open_sessions.get(session_identity)
            if session is None:
                if operation == "close":
                    payload.data = {"closed": True}
                    return payload
                elif operation == "append":
                    self.mark_session_failed(session_identity)
                    raise ValueError("unknown session open index")
                else:
                    raise ValueError("unknown session open index")
            else:
                with session.lock:
                    if operation == "close":
                        self.close_session(session_identity, session)
                        payload.data = {"closed": True}
                        return payload
                    elif session.is_failed:
                        raise RuntimeError(
                            "session is terminal after an append failure"
                        )
                    elif self.is_shutting_down:
                        raise RuntimeError("session scheduler is stopping")
                    else:
                        try:
                            append = self.start_append(payload, session_operation)
                            if not self.begin_append_execution(
                                payload.request_id, session_identity
                            ):
                                raise RuntimeError(
                                    "session append was aborted before execution"
                                )
                            else:
                                pass
                            emit_session_stage_started(
                                request_id=payload.request_id,
                                session_identity=session_identity,
                                input_chunk=append.chunk,
                            )
                            updated_payload = self.session_hooks.append(
                                append.chunk, append.payload, append.context
                            )
                            self.check_usage(session_identity)
                            self.validate_append_completion(payload.request_id)
                            return updated_payload
                        except BaseException:
                            self.mark_session_failed(session_identity)
                            raise

    def start_append(
        self, payload: StagePayload, session_operation: SessionOperation
    ) -> SessionAppend:
        input_chunk = session_operation.chunk
        assert input_chunk is not None, "append operation carries no chunk"
        cancel_event = threading.Event()
        append_state = AppendOperationState(
            session_identity=session_operation.session_identity,
            cancelled=cancel_event,
        )
        with self.session_table_lock:
            self.append_operation_states[payload.request_id] = append_state
        if self.is_aborted(payload.request_id):
            cancel_event.set()
            self.mark_session_failed(session_operation.session_identity)
        else:
            pass

        def emit_chunk(chunk: TimedChunk) -> None:
            if not cancel_event.is_set():
                self.outbox.put(
                    OutgoingMessage(
                        request_id=payload.request_id,
                        type="stream",
                        data=chunk.to_dict(),
                        metadata={"modality": chunk.modality},
                    )
                )
            else:
                pass

        return SessionAppend(
            chunk=input_chunk,
            payload=payload,
            context=SessionContext(
                session_identity=session_operation.session_identity,
                cancelled=cancel_event,
                emit=emit_chunk,
            ),
        )

    def compute_appends(
        self, appends: list[tuple[StagePayload, SessionOperation]]
    ) -> dict[str, StagePayload | Exception]:
        """Run one unit of each of several distinct sessions in one hook call."""
        results: dict[str, StagePayload | Exception] = {}
        started: list[SessionAppend] = []
        session_was_failed: dict[str, bool] = {}

        def fail(
            request_id: str, session_identity: SessionIdentity, exc: Exception
        ) -> None:
            # note (Junnan Li): Fencing keeps a queued later unit off this unit's state.
            self.mark_session_failed(session_identity)
            results[request_id] = exc

        try:
            with ExitStack() as held_locks:
                for payload, session_operation in appends:
                    with self.session_table_lock:
                        session = self.open_sessions.get(
                            session_operation.session_identity
                        )
                    if session is None:
                        fail(
                            payload.request_id,
                            session_operation.session_identity,
                            ValueError("unknown session open index"),
                        )
                    else:
                        try:
                            held_locks.enter_context(session.lock)
                            session_was_failed[payload.request_id] = session.is_failed
                            if self.is_shutting_down:
                                fail(
                                    payload.request_id,
                                    session_operation.session_identity,
                                    RuntimeError("session scheduler is stopping"),
                                )
                            elif session.is_failed:
                                fail(
                                    payload.request_id,
                                    session_operation.session_identity,
                                    RuntimeError(
                                        "session is terminal after an append failure"
                                    ),
                                )
                            else:
                                started.append(
                                    self.start_append(payload, session_operation)
                                )
                        except Exception as exc:
                            fail(
                                payload.request_id,
                                session_operation.session_identity,
                                exc,
                            )
                runnable: list[SessionAppend] = []
                for append in started:
                    if self.begin_append_execution(
                        append.payload.request_id, append.context.session_identity
                    ):
                        runnable.append(append)
                    else:
                        fail(
                            append.payload.request_id,
                            append.context.session_identity,
                            RuntimeError("session append was aborted before execution"),
                        )
                if runnable:
                    try:
                        batch_observation = capture_session_stage_start()
                        if batch_observation is not None:
                            run_id, batch_start_ns = batch_observation
                            for append in runnable:
                                emit_session_stage_started(
                                    request_id=append.payload.request_id,
                                    session_identity=append.context.session_identity,
                                    input_chunk=append.chunk,
                                    timestamp_ns=batch_start_ns,
                                    expected_run_id=run_id,
                                )
                        else:
                            pass
                        updated_payloads = self.session_hooks.append_batch(runnable)
                        if len(updated_payloads) != len(runnable):
                            raise ValueError(
                                "append_batch returned an unexpected number of payloads"
                            )
                        else:
                            pass
                        for append, updated_payload in zip(
                            runnable,
                            updated_payloads,
                            strict=True,
                        ):
                            try:
                                self.check_usage(append.context.session_identity)
                                self.validate_append_completion(
                                    append.payload.request_id
                                )
                                results[append.payload.request_id] = updated_payload
                            except Exception as exc:
                                fail(
                                    append.payload.request_id,
                                    append.context.session_identity,
                                    exc,
                                )
                    except Exception as exc:
                        for append in runnable:
                            if append.payload.request_id not in results:
                                fail(
                                    append.payload.request_id,
                                    append.context.session_identity,
                                    exc,
                                )
                            else:
                                pass
                else:
                    pass
        except Exception as exc:
            for payload, session_operation in appends:
                if payload.request_id not in results:
                    fail(
                        payload.request_id,
                        session_operation.session_identity,
                        exc,
                    )
                else:
                    pass
        finally:
            for payload, session_operation in appends:
                try:
                    self.settle_operation(
                        payload.request_id,
                        session_operation.session_identity,
                        "append",
                        False,
                        session_was_failed.get(payload.request_id, False),
                    )
                except Exception as exc:
                    # Note (chenye): Continue settling every row; a failed close
                    # remains open and fail closed for a later cleanup attempt.
                    results[payload.request_id] = exc
        return results
