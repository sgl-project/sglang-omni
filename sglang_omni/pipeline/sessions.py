# SPDX-License-Identifier: Apache-2.0
"""Coordinator-owned session sequencing over ordinary pipeline requests."""

from __future__ import annotations

import asyncio
import math
import uuid
from collections import deque
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

import msgpack

from sglang_omni.admission import QueueFullError
from sglang_omni.pipeline.replicas import assign_replica_bindings
from sglang_omni.proto import OmniRequest, StreamMessage
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    OutputChunk,
    SessionIdentity,
    SessionLimits,
    SessionOperation,
    TimedChunk,
    wire_size,
)


class SessionStreamHandler(Protocol):
    def __call__(self, message: StreamMessage) -> None: ...


@dataclass(kw_only=True)
class Session:
    session_identity: SessionIdentity
    request: OmniRequest
    stages: tuple[str, ...]
    bindings: dict[str, int]
    limits: SessionLimits
    opened: list[str] = field(default_factory=list)
    pending: deque[tuple[TimedChunk, int]] = field(default_factory=deque)
    pending_bytes: int = 0
    pending_count: int = 0
    outputs: deque[tuple[OutputChunk, int]] = field(default_factory=deque)
    output_bytes: int = 0
    next_input: int = 0
    next_output: int = 0
    modality_end_ms: dict[str, float] = field(default_factory=dict)
    ended_modalities: set[str] = field(default_factory=set)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    output_wake: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pump: asyncio.Task[None] | None = None
    is_closing: bool = False
    is_closed: bool = False
    is_reading: bool = False
    error: BaseException | None = None
    cleanup_error: BaseException | None = None


class CoordinatorSessions:
    """Coordinator-owned sessions over fixed stage routes."""

    def __init__(self) -> None:
        self.is_sessions_stopping = False
        self.next_incarnation = 1
        self.session_unavailable_stages: set[str] = set()
        self.sessions: dict[str, Session] = {}
        self.session_stream_handlers: dict[str, SessionStreamHandler] = {}
        self.session_cleanup_tasks: set[asyncio.Task[None]] = set()

    def owned_session_task(
        self, coroutine: Coroutine[None, None, None]
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)
        self.session_cleanup_tasks.add(task)
        task.add_done_callback(self.session_task_done)
        return task

    def session_task_done(self, task: asyncio.Task[None]) -> None:
        self.session_cleanup_tasks.discard(task)
        if not task.cancelled():
            task.exception()
        else:
            pass

    def reject_session_metadata(self, request: object) -> None:
        if (
            isinstance(request, OmniRequest)
            and SESSION_METADATA_KEY in request.metadata
        ):
            raise ValueError(
                f"request metadata key {SESSION_METADATA_KEY!r} is reserved"
            )
        else:
            pass

    def get_session(self, session_identity: SessionIdentity) -> Session:
        session = self.sessions.get(session_identity.session_id)
        if session is None or session.session_identity != session_identity:
            raise ValueError("unknown or stale session reference")
        else:
            pass
        return session

    async def open_session(
        self,
        request: OmniRequest,
        *,
        stages: list[str],
        limits: SessionLimits | None = None,
        session_id: str | None = None,
    ) -> SessionIdentity:
        """Open a fixed linear route, upstream to downstream, before accepting input."""
        if (
            self.is_sessions_stopping
            or not self.running
            or self.fatal_error is not None
        ):
            raise RuntimeError(self.fatal_error or "Coordinator is not running")
        else:
            pass
        if (
            not stages
            or stages[0] != self.entry_stage
            or len(set(stages)) != len(stages)
        ):
            raise ValueError("stages must be a unique route beginning at entry_stage")
        else:
            pass
        session_id = session_id or str(uuid.uuid4())
        if session_id in self.sessions:
            raise ValueError("session ID already reserved")
        else:
            pass
        bindings = (
            assign_replica_bindings(
                self.logical_process_plan, self.binding_policy, session_id
            )
            or {}
        )
        owners = tuple(
            (
                self.replica_topology.resolve(stage, bindings[stage])
                if self.replica_topology.is_replicated(stage)
                else stage
            )
            for stage in stages
        )
        if any(owner not in self.stages for owner in owners):
            raise ValueError("session route contains an unregistered owner")
        else:
            pass
        if self.session_unavailable_stages.intersection(owners):
            raise ValueError(
                "session route contains an unregistered owner or unavailable owner"
            )
        else:
            pass
        session = Session(
            session_identity=SessionIdentity(session_id, self.next_incarnation),
            request=request,
            stages=owners,
            bindings=bindings,
            limits=limits or SessionLimits(),
        )
        self.next_incarnation += 1
        self.sessions[session_id] = session
        try:
            async with session.lock:
                for owner in owners:
                    # Note (Junnan Li): Record the attempt first; a stage may allocate before its reply is lost.
                    session.opened.append(owner)
                    await self.session_operation(session, "open", owner=owner)
                if (
                    self.is_sessions_stopping
                    or self.session_unavailable_stages.intersection(owners)
                ):
                    raise RuntimeError("session owners are shutting down")
                else:
                    pass
        except BaseException:
            await asyncio.shield(
                self.owned_session_task(self.close_session_state(session))
            )
            raise
        session.request = replace(request, inputs=None)
        session.pump = asyncio.create_task(self.pump_session(session))
        return session.session_identity

    async def append_session(
        self, session_identity: SessionIdentity, chunk: TimedChunk
    ) -> int:
        """Accept input in global seq order, independently of output consumption.

        Rejected input keeps its seq for retry; accepted input must not be resubmitted.
        """
        session = self.get_session(session_identity)
        if session.is_closing or session.is_closed:
            raise RuntimeError("session is closing")
        else:
            pass
        if chunk.seq != session.next_input:
            raise ValueError("input seq must be contiguous within an incarnation")
        else:
            pass
        if chunk.modality in session.ended_modalities:
            raise ValueError("input after EOS")
        else:
            pass
        if (
            not math.isfinite(chunk.t_start_ms)
            or not math.isfinite(chunk.duration_ms)
            or chunk.duration_ms < 0
        ):
            raise ValueError("input timing must be finite with a non-negative duration")
        else:
            pass
        if chunk.t_start_ms < session.modality_end_ms.get(chunk.modality, 0):
            raise ValueError("input timing overlaps or moves backwards")
        else:
            pass
        if isinstance(chunk.payload, bytes):
            encoded_bytes = wire_size(chunk.to_dict())
        else:
            # Note (Junnan Li): Snapshot a mutable payload so later caller edits cannot reach it.
            encoded = msgpack.packb(chunk.to_dict(), use_bin_type=True)
            encoded_bytes = len(encoded)
            chunk = TimedChunk.from_dict(msgpack.unpackb(encoded, raw=False))
        limits = session.limits
        if encoded_bytes > limits.max_chunk_bytes:
            raise ValueError(
                f"input chunk is {encoded_bytes} bytes; max_chunk_bytes is "
                f"{limits.max_chunk_bytes}"
            )
        else:
            pass
        if (
            session.pending_count >= limits.max_pending_chunks
            or session.pending_bytes + encoded_bytes > limits.max_pending_bytes
        ):
            raise QueueFullError()
        else:
            pass
        if (
            chunk.modality not in session.modality_end_ms
            and len(session.modality_end_ms) >= limits.max_modalities
        ):
            raise QueueFullError()
        else:
            pass
        session.pending.append((chunk, encoded_bytes))
        session.pending_count += 1
        session.pending_bytes += encoded_bytes
        session.next_input += 1
        session.modality_end_ms[chunk.modality] = chunk.t_start_ms + chunk.duration_ms
        if chunk.eos:
            session.ended_modalities.add(chunk.modality)
        else:
            pass
        session.wake.set()
        return chunk.seq

    async def session_outputs(
        self, session_identity: SessionIdentity
    ) -> AsyncIterator[OutputChunk]:
        """One output consumer; disconnect closes the owned session."""
        session = self.get_session(session_identity)
        if session.is_reading:
            raise RuntimeError("session already has an output consumer")
        else:
            pass
        session.is_reading = True
        try:
            while True:
                while session.outputs and not session.is_closing:
                    output, size = session.outputs.popleft()
                    session.output_bytes -= size
                    yield output
                if session.is_closed:
                    if session.error is not None:
                        raise session.error
                    else:
                        pass
                    return
                else:
                    pass
                session.output_wake.clear()
                await session.output_wake.wait()
        finally:
            session.is_reading = False
            await asyncio.shield(
                self.owned_session_task(self.close_session_state(session))
            )

    def emit_session_output(
        self,
        session: Session,
        input_seq: int,
        chunk: TimedChunk,
        *,
        kind: Literal["data", "input_done"] = "data",
    ) -> None:
        if session.is_closing:
            return
        else:
            pass
        output = OutputChunk(
            session_identity=session.session_identity,
            seq=session.next_output,
            input_seq=input_seq,
            modality=chunk.modality,
            t_start_ms=chunk.t_start_ms,
            duration_ms=chunk.duration_ms,
            payload=chunk.payload,
            format=chunk.format,
            eos=chunk.eos,
            kind=kind,
        )
        size = wire_size(output.to_dict())
        if (
            len(session.outputs) >= session.limits.max_output_chunks
            or session.output_bytes + size > session.limits.max_output_bytes
        ):
            raise QueueFullError()
        else:
            pass
        session.outputs.append((output, size))
        session.output_bytes += size
        session.next_output += 1
        session.output_wake.set()

    async def pump_session(self, session: Session) -> None:
        try:
            while not session.is_closing:
                if not session.pending:
                    session.wake.clear()
                    await asyncio.wait_for(
                        session.wake.wait(), session.limits.idle_timeout_s
                    )
                    continue
                else:
                    pass
                chunk, size = session.pending.popleft()
                try:
                    await self.session_operation(session, "append", chunk=chunk)
                    self.emit_session_output(
                        session,
                        chunk.seq,
                        replace(chunk, payload=None),
                        kind="input_done",
                    )
                finally:
                    session.pending_count -= 1
                    session.pending_bytes -= size
        except Exception as exc:
            session.error = exc
            self.owned_session_task(self.close_session_state(session))

    async def session_operation(
        self,
        session: Session,
        operation: Literal["open", "append", "close"],
        *,
        owner: str | None = None,
        chunk: TimedChunk | None = None,
    ) -> None:
        session_identity = session.session_identity
        session_operation = SessionOperation(
            operation=operation,
            session_identity=session_identity,
            stages=session.stages,
            chunk=chunk,
        )
        request = replace(
            session.request,
            metadata={
                **session.request.metadata,
                SESSION_METADATA_KEY: session_operation.to_dict(),
            },
        )
        request_id = f"session-{uuid.uuid4()}"

        if chunk is not None:
            input_seq = chunk.seq

            def output(msg: StreamMessage) -> None:
                try:
                    self.emit_session_output(
                        session, input_seq, TimedChunk.from_dict(msg.chunk)
                    )
                except Exception as exc:
                    self.reject_completion_future(request_id, exc)

            self.session_stream_handlers[request_id] = output
        else:
            pass

        async def run() -> None:
            await self.submit_request(
                request_id,
                request,
                target_stage=owner,
                terminal_stages=(
                    {self.replica_topology.logical_name(owner)}
                    if owner
                    else {self.replica_topology.logical_name(session.stages[-1])}
                ),
                replica_bindings=session.bindings,
                should_bypass_admission=operation == "close",
            )
            await self.completion_futures[request_id]

        try:
            if operation == "close":
                await run()
            else:
                await asyncio.wait_for(run(), session.limits.operation_timeout_s)
        except asyncio.TimeoutError as exc:
            self.begin_session_close(session)
            raise TimeoutError(f"session {operation} timed out") from exc
        except BaseException:
            # Note (Junnan Li): Request abort can yield before the pump sees this fatal failure.
            self.begin_session_close(session)
            raise
        finally:
            self.session_stream_handlers.pop(request_id, None)
            if request_id in self.requests:
                await self.abort(request_id)
            else:
                pass
            future = self.completion_futures.pop(request_id, None)
            if future is not None and not future.done():
                future.cancel()
            else:
                pass

    async def close_session(self, session_identity: SessionIdentity) -> None:
        session = self.sessions.get(session_identity.session_id)
        if session is None:
            return
        else:
            pass
        if session.session_identity != session_identity:
            raise ValueError("stale session reference")
        else:
            pass
        await asyncio.shield(self.owned_session_task(self.close_session_state(session)))
        if session.cleanup_error is not None:
            raise RuntimeError(
                "session cleanup incomplete; capacity remains reserved"
            ) from session.cleanup_error
        else:
            pass

    def begin_session_close(self, session: Session) -> None:
        session.is_closing = True
        # Note (Junnan Li): Close fences output like cancel; queued data is dropped, not drained.
        session.outputs.clear()
        session.output_bytes = 0
        session.wake.set()
        session.output_wake.set()

    def close_session_state(self, session: Session) -> Coroutine[None, None, None]:
        self.begin_session_close(session)
        return self.finish_session_close(session)

    async def finish_session_close(self, session: Session) -> None:
        async with session.lock:
            if session.is_closed:
                return
            else:
                pass
            await self.cleanup_session(session)

    async def cleanup_session(self, session: Session) -> None:
        self.begin_session_close(session)
        if session.pump is not None and session.pump is not asyncio.current_task():
            try:
                await asyncio.wait_for(
                    asyncio.shield(session.pump), session.limits.operation_timeout_s
                )
            except asyncio.TimeoutError as exc:
                session.cleanup_error = session.error = exc
                self.session_unavailable_stages.update(session.opened)
                session.pump.cancel()
                await asyncio.gather(session.pump, return_exceptions=True)
                session.is_closed = True
                session.output_wake.set()
                return
        else:
            pass
        session.pending.clear()
        session.pending_count = session.pending_bytes = 0
        unconfirmed = list(session.opened)
        for owner in reversed(session.opened):
            try:
                await self.session_operation(session, "close", owner=owner)
                unconfirmed.pop()
            except Exception as exc:
                session.cleanup_error = exc
                session.error = session.error or exc
                # Note (Junnan Li): An unacknowledged downstream owner may still use upstream data.
                self.session_unavailable_stages.update(unconfirmed)
                break
        session.is_closed = True
        session.output_wake.set()
        # Note (Junnan Li): An unacknowledged owner may still hold buffers; keep its capacity reserved.
        if session.cleanup_error is None:
            del self.sessions[session.session_identity.session_id]
        else:
            pass

    async def shutdown_stage_sessions(self, selected: set[str] | None) -> None:
        affected = set(self.stages) if selected is None else selected
        self.session_unavailable_stages.update(affected)
        if selected is None:
            await self.stop_sessions()
        else:
            await asyncio.gather(
                *(
                    self.close_session_state(session)
                    for session in list(self.sessions.values())
                    if affected.intersection(session.stages)
                )
            )

    async def stop_sessions(self) -> None:
        self.is_sessions_stopping = True
        await asyncio.gather(
            *(
                self.close_session_state(session)
                for session in list(self.sessions.values())
            )
        )
        await asyncio.gather(*self.session_cleanup_tasks, return_exceptions=True)

    async def fail_sessions(self, message: str) -> None:
        for session in list(self.sessions.values()):
            session.error = RuntimeError(message)
        await self.stop_sessions()
