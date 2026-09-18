# SPDX-License-Identifier: Apache-2.0
"""Persistent state for session-aware pipeline stages."""
from __future__ import annotations

import queue
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from sglang_omni.admission import QueueFullError
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    ResourceUsage,
    SessionRef,
    TimedChunk,
    wire_size,
)
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


@dataclass
class SessionContext:
    ref: SessionRef
    cancelled: threading.Event
    emit: Callable[[TimedChunk], None]


class SessionHooks:
    """Hooks run serially per session; different sessions may run concurrently.

    Failed open must release allocations it has not returned. append must stop
    using state before returning, including on cancellation. close is idempotent.
    """

    def open(self, ref: SessionRef, request: OmniRequest) -> Any:
        raise NotImplementedError

    def append(
        self,
        state: Any,
        chunk: TimedChunk,
        payload: StagePayload,
        context: SessionContext,
    ) -> StagePayload:
        raise NotImplementedError

    def abort(self, state: Any, ref: SessionRef) -> None:
        """Retain a usable context or raise to require closing the session."""
        raise NotImplementedError("this stage cannot retain context after abort")

    def close(self, state: Any) -> None:
        raise NotImplementedError

    def usage(self, state: Any) -> ResourceUsage:
        return ResourceUsage()


@dataclass
class _StageSession:
    ref: SessionRef
    state: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    usage: ResourceUsage = field(default_factory=ResourceUsage)


@dataclass
class _Order:
    """Arrival order of one session's commands: seq `served` runs next."""

    issued: int = 0
    served: int = 0
    finished: set[int] = field(default_factory=set)


class _SessionInbox(queue.Queue):
    def __init__(self, register):
        super().__init__()
        self._register = register

    def put(self, message, block=True, timeout=None):
        if message.type == "new_request":
            self._register(message)
        super().put(message, block, timeout)


class SessionScheduler(SimpleScheduler):
    """Opt-in scheduler for persistent hooks, with bounded stage admission."""

    def __init__(
        self,
        hooks: SessionHooks,
        *,
        compute_fn: Callable[[StagePayload], StagePayload] | None = None,
        max_sessions: int = 64,
        max_concurrency: int = 4,
        max_state_bytes: int = 1 << 30,
    ):
        self.hooks = hooks
        self._ordinary_compute = compute_fn
        self.max_sessions = max_sessions
        self.max_state_bytes = max_state_bytes
        self._sessions: dict[tuple[str, int], _StageSession] = {}
        self._commands: dict[str, threading.Event] = {}
        self._session_lock = threading.Lock()
        self._closing = False
        self._tickets: dict[str, tuple[tuple[str, int], int]] = {}
        self._orders: dict[tuple[str, int], _Order] = {}
        self._served = threading.Condition(self._session_lock)
        super().__init__(
            self._compute,
            max_concurrency=max_concurrency,
            abort_callback=self._cancel_command,
            shutdown_callback=self._shutdown_sessions,
        )
        self.inbox = _SessionInbox(self._register_command)

    def _register_command(self, message) -> None:
        command = message.data.request.metadata.get(SESSION_METADATA_KEY)
        if command is None:
            return
        try:
            key = (command["ref"]["session_id"], command["ref"]["incarnation"])
        except (KeyError, TypeError):
            # Note (Junnan Li): put() runs on the stage loop; a malformed command
            # must fail in _compute, inside the request error boundary.
            return
        with self._session_lock:
            order = self._orders.setdefault(key, _Order())
            self._tickets[message.request_id] = (key, order.issued)
            order.issued += 1

    def _finish_command(self, request_id) -> None:
        with self._served:
            ticket = self._tickets.pop(request_id, None)
            if ticket is None:
                return
            key, seq = ticket
            order = self._orders.get(key)
            if order is None:
                return
            # Note (Junnan Li): An aborted command can finish before its predecessors ran.
            order.finished.add(seq)
            while order.served in order.finished:
                order.finished.discard(order.served)
                order.served += 1
            if order.served == order.issued:
                del self._orders[key]
            self._served.notify_all()

    def _consume_if_aborted(self, request_id):
        aborted = super()._consume_if_aborted(request_id)
        if aborted:
            self._finish_command(request_id)
        return aborted

    def _compute(self, payload: StagePayload) -> StagePayload:
        command = payload.request.metadata.get(SESSION_METADATA_KEY)
        if command is None:
            if self._ordinary_compute is None:
                raise ValueError("this stage has no compute_fn for ordinary requests")
            return self._ordinary_compute(payload)
        ref = command["ref"]
        key = (ref["session_id"], ref["incarnation"])
        try:
            with self._served:
                # Note (Junnan Li): A request-level abort can consume this command's number
                # before it runs (a stream message for the same request arrives first);
                # such a command has no ticket left and must not wait.
                ticket = self._tickets.get(payload.request_id)
                if ticket is not None:
                    key, seq = ticket
                    self._served.wait_for(
                        lambda: (order := self._orders.get(key)) is None
                        or order.served >= seq
                    )
            return self._compute_session(payload)
        finally:
            try:
                # Once the hook released its owner lock, either stop observes
                # that unlocked owner or this completion observes shutdown.
                with self._session_lock:
                    session = self._sessions.get(key) if self._closing else None
                if session is not None:
                    with session.lock:
                        self._close(key, session)
            finally:
                self._finish_command(payload.request_id)

    def _cancel_command(self, request_id: str) -> None:
        with self._session_lock:
            event = self._commands.get(request_id)
            if event is not None:
                event.set()

    def _shutdown_sessions(self) -> None:
        with self._session_lock:
            self._closing = True
            for event in self._commands.values():
                event.set()
            sessions = list(self._sessions.items())
        errors = []
        for key, session in sessions:
            if session.lock.acquire(blocking=False):
                try:
                    self._close(key, session)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    session.lock.release()
        if errors:
            raise RuntimeError("session shutdown cleanup failed") from errors[0]

    def _close(self, key, session) -> None:
        if session.state is not None:
            self.hooks.close(session.state)
            session.state = None
        with self._session_lock:
            self._sessions.pop(key, None)

    def _update_usage(self, session: _StageSession, *, admit: bool = True) -> None:
        usage = self.hooks.usage(session.state)
        with self._session_lock:
            session.usage = usage
            if (
                admit
                and sum(s.usage.bytes for s in self._sessions.values())
                > self.max_state_bytes
            ):
                raise QueueFullError()

    def _open_session(self, ref: SessionRef, request: OmniRequest) -> None:
        key = (ref.session_id, ref.incarnation)
        session = _StageSession(ref, None)
        session.lock.acquire()
        with self._session_lock:
            if self._closing:
                session.lock.release()
                raise RuntimeError("session scheduler is stopping")
            if key in self._sessions:
                session.lock.release()
                raise ValueError("session already opened")
            if len(self._sessions) >= self.max_sessions:
                session.lock.release()
                raise QueueFullError()
            self._sessions[key] = session
        try:
            session.state = self.hooks.open(ref, request)
            self._update_usage(session)
            if self._closing:
                raise RuntimeError("session scheduler is stopping")
        except BaseException:
            self._close(key, session)
            raise
        finally:
            session.lock.release()

    def _compute_session(self, payload: StagePayload) -> StagePayload:
        command = payload.request.metadata[SESSION_METADATA_KEY]
        ref = SessionRef(**command["ref"])
        key = (ref.session_id, ref.incarnation)
        op = command["op"]
        if op == "open":
            self._open_session(ref, payload.request)
            payload.data = {"opened": True}
            return payload

        with self._session_lock:
            session = self._sessions.get(key)
        if session is None:
            if op == "close":
                payload.data = {"closed": True}
                return payload
            raise ValueError("unknown session incarnation")
        with session.lock:
            if op == "close":
                self._close(key, session)
                payload.data = {"closed": True}
                return payload
            if self._closing:
                raise RuntimeError("session scheduler is stopping")
            if op == "abort":
                if ref.epoch != session.ref.epoch + 1:
                    raise ValueError("invalid abort epoch")
                self.hooks.abort(session.state, ref)
                session.ref = ref
                self._update_usage(session, admit=False)
                payload.data = {"aborted": True}
                return payload
            if ref != session.ref:
                raise ValueError("stale session epoch")
            event = threading.Event()
            with self._session_lock:
                self._commands[payload.request_id] = event
            with self._abort_lock:
                if payload.request_id in self._aborted:
                    event.set()

            emitted_count = emitted_bytes = 0

            def emit(chunk: TimedChunk) -> None:
                nonlocal emitted_count, emitted_bytes
                emitted_count += 1
                emitted_bytes += wire_size(asdict(chunk))
                limits = command["output_limits"]
                if emitted_count > limits["chunks"] or emitted_bytes > limits["bytes"]:
                    raise QueueFullError()
                if not event.is_set():
                    self.outbox.put(
                        OutgoingMessage(
                            request_id=payload.request_id,
                            type="stream",
                            data=asdict(chunk),
                            metadata={"modality": chunk.modality},
                        )
                    )

            try:
                result = self.hooks.append(
                    session.state,
                    TimedChunk(**command["chunk"]),
                    payload,
                    SessionContext(ref, event, emit),
                )
                self._update_usage(session)
                return result
            finally:
                with self._session_lock:
                    self._commands.pop(payload.request_id, None)
