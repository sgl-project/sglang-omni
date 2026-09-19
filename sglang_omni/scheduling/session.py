# SPDX-License-Identifier: Apache-2.0
"""Persistent state for session-aware pipeline stages."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from sglang_omni.admission import QueueFullError
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    ResourceUsage,
    SessionCommand,
    SessionRef,
    TimedChunk,
    find_session_command,
)
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
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
class StageSession:
    ref: SessionRef
    state: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    usage: ResourceUsage = field(default_factory=ResourceUsage)


@dataclass
class CommandOrder:
    """Arrival order of one session's commands: seq served runs next."""

    issued: int = 0
    served: int = 0
    finished: set[int] = field(default_factory=set)


class SessionInbox(queue.Queue):
    def __init__(self, register: Callable[[IncomingMessage], None]) -> None:
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
    ) -> None:
        self.hooks = hooks
        self.ordinary_compute = compute_fn
        self.max_sessions = max_sessions
        self.max_state_bytes = max_state_bytes
        self.sessions: dict[tuple[str, int], StageSession] = {}
        self.commands: dict[str, threading.Event] = {}
        self.session_lock = threading.Lock()
        self.closing = False
        self.tickets: dict[str, tuple[tuple[str, int], int]] = {}
        self.orders: dict[tuple[str, int], CommandOrder] = {}
        self.served = threading.Condition(self.session_lock)
        super().__init__(
            self.compute,
            max_concurrency=max_concurrency,
            abort_callback=self.cancel_command,
            shutdown_callback=self.shutdown_sessions,
        )
        self.inbox = SessionInbox(self.register_command)

    def register_command(self, message: IncomingMessage) -> None:
        try:
            command = find_session_command(message.data.request.metadata)
        except ValueError:
            # Note (Junnan Li): put() runs on the stage loop; compute reports the malformed command.
            return
        if command is None:
            return
        key = (command.ref.session_id, command.ref.incarnation)
        with self.session_lock:
            order = self.orders.setdefault(key, CommandOrder())
            self.tickets[message.request_id] = (key, order.issued)
            order.issued += 1

    def finish_command(self, request_id: str) -> None:
        with self.served:
            ticket = self.tickets.pop(request_id, None)
            if ticket is None:
                return
            key, seq = ticket
            order = self.orders.get(key)
            if order is None:
                return
            # Note (Junnan Li): An aborted command can finish before its predecessors ran.
            order.finished.add(seq)
            while order.served in order.finished:
                order.finished.discard(order.served)
                order.served += 1
            if order.served == order.issued:
                del self.orders[key]
            self.served.notify_all()

    def _consume_if_aborted(self, request_id: str) -> bool:
        aborted = super()._consume_if_aborted(request_id)
        if aborted:
            self.finish_command(request_id)
        return aborted

    def compute(self, payload: StagePayload) -> StagePayload:
        command = find_session_command(payload.request.metadata)
        if command is None:
            if self.ordinary_compute is None:
                raise ValueError("this stage has no compute_fn for ordinary requests")
            return self.ordinary_compute(payload)
        key = (command.ref.session_id, command.ref.incarnation)
        try:
            with self.served:
                # Note (Junnan Li): A request-level abort may already have consumed the ticket.
                ticket = self.tickets.get(payload.request_id)
                if ticket is not None:
                    key, seq = ticket
                    self.served.wait_for(
                        lambda: (order := self.orders.get(key)) is None
                        or order.served >= seq
                    )
            return self.compute_session(payload, command)
        finally:
            try:
                # Note (Junnan Li): stop skips a session whose hook is running; it is closed here.
                with self.session_lock:
                    session = self.sessions.get(key) if self.closing else None
                if session is not None:
                    with session.lock:
                        self.close_session(key, session)
            finally:
                self.finish_command(payload.request_id)

    def cancel_command(self, request_id: str) -> None:
        with self.session_lock:
            event = self.commands.get(request_id)
            if event is not None:
                event.set()

    def shutdown_sessions(self) -> None:
        with self.session_lock:
            self.closing = True
            for event in self.commands.values():
                event.set()
            sessions = list(self.sessions.items())
        errors: list[Exception] = []
        for key, session in sessions:
            if session.lock.acquire(blocking=False):
                try:
                    self.close_session(key, session)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    session.lock.release()
        if errors:
            raise RuntimeError("session shutdown cleanup failed") from errors[0]

    def close_session(self, key: tuple[str, int], session: StageSession) -> None:
        if session.state is not None:
            self.hooks.close(session.state)
            session.state = None
        with self.session_lock:
            self.sessions.pop(key, None)

    def update_usage(self, session: StageSession, *, admit: bool = True) -> None:
        usage = self.hooks.usage(session.state)
        with self.session_lock:
            session.usage = usage
            if (
                admit
                and sum(s.usage.bytes for s in self.sessions.values())
                > self.max_state_bytes
            ):
                raise QueueFullError()

    def open_session(self, ref: SessionRef, request: OmniRequest) -> None:
        key = (ref.session_id, ref.incarnation)
        session = StageSession(ref, None)
        session.lock.acquire()
        with self.session_lock:
            if self.closing:
                session.lock.release()
                raise RuntimeError("session scheduler is stopping")
            if key in self.sessions:
                session.lock.release()
                raise ValueError("session already opened")
            if len(self.sessions) >= self.max_sessions:
                session.lock.release()
                raise QueueFullError()
            self.sessions[key] = session
        try:
            session.state = self.hooks.open(ref, request)
            self.update_usage(session)
            if self.closing:
                raise RuntimeError("session scheduler is stopping")
        except BaseException:
            self.close_session(key, session)
            raise
        finally:
            session.lock.release()

    def compute_session(
        self, payload: StagePayload, command: SessionCommand
    ) -> StagePayload:
        ref = command.ref
        key = (ref.session_id, ref.incarnation)
        op = command.op
        if op == "open":
            self.open_session(ref, payload.request)
            payload.data = {"opened": True}
            return payload

        with self.session_lock:
            session = self.sessions.get(key)
        if session is None:
            if op == "close":
                payload.data = {"closed": True}
                return payload
            raise ValueError("unknown session incarnation")
        with session.lock:
            if op == "close":
                self.close_session(key, session)
                payload.data = {"closed": True}
                return payload
            if self.closing:
                raise RuntimeError("session scheduler is stopping")
            if op == "abort":
                if ref.epoch != session.ref.epoch + 1:
                    raise ValueError("invalid abort epoch")
                self.hooks.abort(session.state, ref)
                session.ref = ref
                self.update_usage(session, admit=False)
                payload.data = {"aborted": True}
                return payload
            if ref != session.ref:
                raise ValueError("stale session epoch")
            input_chunk = command.chunk
            assert input_chunk is not None, "append command carries no chunk"
            event = threading.Event()
            with self.session_lock:
                self.commands[payload.request_id] = event
            with self._abort_lock:
                if payload.request_id in self._aborted:
                    event.set()

            def emit(chunk: TimedChunk) -> None:
                if not event.is_set():
                    self.outbox.put(
                        OutgoingMessage(
                            request_id=payload.request_id,
                            type="stream",
                            data=chunk.to_dict(),
                            metadata={"modality": chunk.modality},
                        )
                    )

            try:
                result = self.hooks.append(
                    session.state,
                    input_chunk,
                    payload,
                    SessionContext(ref, event, emit),
                )
                self.update_usage(session)
                return result
            finally:
                with self.session_lock:
                    self.commands.pop(payload.request_id, None)
