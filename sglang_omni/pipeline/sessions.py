# SPDX-License-Identifier: Apache-2.0
"""Coordinator-owned session sequencing over ordinary pipeline requests."""
from __future__ import annotations

import asyncio
import math
import secrets
import uuid
from collections import deque
from collections.abc import Coroutine
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Callable, Literal, TypeVar

import msgpack

from sglang_omni.admission import QueueFullError
from sglang_omni.pipeline.replicas import assign_replica_bindings
from sglang_omni.proto import OmniRequest, StreamMessage
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    OutputChunk,
    SessionCommand,
    SessionLimits,
    SessionOp,
    SessionRef,
    TimedChunk,
    wire_size,
)

TaskResult = TypeVar("TaskResult")


@dataclass
class Session:
    ref: SessionRef
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
    ends: dict[str, float] = field(default_factory=dict)
    eos: set[str] = field(default_factory=set)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    output_wake: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    unit_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pump: asyncio.Task | None = None
    closing: bool = False
    closed: bool = False
    reading: bool = False
    error: BaseException | None = None
    cleanup_error: BaseException | None = None


class CoordinatorSessions:
    """Coordinator-owned sessions over fixed stage routes."""

    def __init__(self, max_sessions: int) -> None:
        self.max_sessions = max_sessions
        self.sessions_stopping = False
        self.session_unavailable_stages: set[str] = set()
        self.sessions: dict[str, Session] = {}
        self.session_stream_handlers: dict[str, Callable[[StreamMessage], None]] = {}
        self.session_cleanup_tasks: set[asyncio.Task] = set()

    def owned_session_task(
        self, coroutine: Coroutine[Any, Any, TaskResult]
    ) -> asyncio.Task[TaskResult]:
        task = asyncio.create_task(coroutine)
        self.session_cleanup_tasks.add(task)
        task.add_done_callback(self.session_task_done)
        return task

    def session_task_done(self, task: asyncio.Task) -> None:
        self.session_cleanup_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def reject_session_metadata(self, request: OmniRequest | Any) -> None:
        if (
            isinstance(request, OmniRequest)
            and SESSION_METADATA_KEY in request.metadata
        ):
            raise ValueError(
                f"request metadata key {SESSION_METADATA_KEY!r} is reserved"
            )

    def get_session(self, ref: SessionRef) -> Session:
        session = self.sessions.get(ref.session_id)
        if session is None or session.ref != ref:
            raise ValueError("unknown or stale session reference")
        return session

    async def open_session(
        self,
        request: OmniRequest,
        *,
        stages: list[str],
        limits: SessionLimits | None = None,
        session_id: str | None = None,
    ) -> SessionRef:
        """Open a fixed linear route, upstream to downstream, before accepting input."""
        if self.sessions_stopping or not self._running or self._fatal_error is not None:
            raise RuntimeError(self._fatal_error or "Coordinator is not running")
        if (
            not stages
            or stages[0] != self.entry_stage
            or len(set(stages)) != len(stages)
        ):
            raise ValueError("stages must be a unique route beginning at entry_stage")
        if len(self.sessions) >= self.max_sessions:
            raise QueueFullError()
        session_id = session_id or str(uuid.uuid4())
        if session_id in self.sessions:
            raise ValueError("session ID already reserved")
        bindings = (
            assign_replica_bindings(
                self._logical_process_plan, self._binding_policy, session_id
            )
            or {}
        )
        owners = tuple(
            (
                self._replica_topology.resolve(stage, bindings[stage])
                if self._replica_topology.is_replicated(stage)
                else stage
            )
            for stage in stages
        )
        if any(owner not in self._stages for owner in owners):
            raise ValueError("session route contains an unregistered owner")
        if self.session_unavailable_stages.intersection(owners):
            raise ValueError(
                "session route contains an unregistered owner or unavailable owner"
            )
        session = Session(
            SessionRef(session_id, secrets.randbelow((1 << 63) - 1) + 1),
            request,
            owners,
            bindings,
            limits or SessionLimits(),
        )
        self.sessions[session_id] = session
        try:
            async with session.lock:
                for owner in owners:
                    # Note (Junnan Li): Record the attempt first; a stage may allocate before its reply is lost.
                    session.opened.append(owner)
                    await self.session_command(session, "open", owner=owner)
                if (
                    self.sessions_stopping
                    or self.session_unavailable_stages.intersection(owners)
                ):
                    raise RuntimeError("session owners are shutting down")
        except BaseException:
            await asyncio.shield(
                self.owned_session_task(self.close_session_state(session))
            )
            raise
        session.request = replace(request, inputs=None)
        session.pump = asyncio.create_task(self.pump_session(session))
        return session.ref

    async def append_session(self, ref: SessionRef, chunk: TimedChunk) -> int:
        """Accept input in global seq order, independently of output consumption.

        Adapters map per-stream seq to this order. Rejected input keeps its seq
        for retry; accepted input advances it and must not be resubmitted.
        """
        session = self.get_session(ref)
        if session.closing or session.closed:
            raise RuntimeError("session is closing")
        if chunk.seq != session.next_input:
            raise ValueError("input seq must be contiguous within an incarnation")
        if chunk.modality in session.eos:
            raise ValueError("input after EOS")
        if (
            not math.isfinite(chunk.t_start_ms)
            or not math.isfinite(chunk.duration_ms)
            or chunk.duration_ms < 0
        ):
            raise ValueError("input timing must be finite with a non-negative duration")
        if chunk.t_start_ms < session.ends.get(chunk.modality, 0):
            raise ValueError("input timing overlaps or moves backwards")
        if isinstance(chunk.payload, bytes):
            size = wire_size(chunk.to_dict())
        else:
            # Note (Junnan Li): Snapshot a mutable payload so later caller edits cannot reach it.
            encoded = msgpack.packb(chunk.to_dict(), use_bin_type=True)
            size = len(encoded)
            chunk = TimedChunk.from_dict(msgpack.unpackb(encoded, raw=False))
        limits = session.limits
        if size > limits.max_chunk_bytes:
            raise ValueError(
                f"input chunk is {size} bytes; max_chunk_bytes is {limits.max_chunk_bytes}"
            )
        if (
            session.pending_count >= limits.max_pending_chunks
            or session.pending_bytes + size > limits.max_pending_bytes
        ):
            raise QueueFullError()
        if (
            chunk.modality not in session.ends
            and len(session.ends) >= limits.max_modalities
        ):
            raise QueueFullError()
        session.pending.append((chunk, size))
        session.pending_count += 1
        session.pending_bytes += size
        session.next_input += 1
        session.ends[chunk.modality] = chunk.t_start_ms + chunk.duration_ms
        if chunk.eos:
            session.eos.add(chunk.modality)
        session.wake.set()
        return chunk.seq

    async def session_outputs(self, ref: SessionRef) -> AsyncIterator[OutputChunk]:
        """One output consumer; disconnect closes the owned session."""
        session = self.get_session(ref)
        if session.reading:
            raise RuntimeError("session already has an output consumer")
        session.reading = True
        try:
            while True:
                while session.outputs and not session.closing:
                    output, size = session.outputs.popleft()
                    session.output_bytes -= size
                    if output.ref == session.ref or (
                        output.kind == "input_done"
                        and output.ref.incarnation == session.ref.incarnation
                    ):
                        yield output
                if session.closed:
                    if session.error is not None:
                        raise session.error
                    return
                session.output_wake.clear()
                await session.output_wake.wait()
        finally:
            session.reading = False
            await asyncio.shield(
                self.owned_session_task(self.close_session_state(session))
            )

    def emit_session_output(
        self,
        session: Session,
        ref: SessionRef,
        input_seq: int,
        chunk: TimedChunk,
        *,
        kind: Literal["data", "input_done"] = "data",
    ) -> None:
        if session.closing or (session.ref != ref and kind != "input_done"):
            return
        output = OutputChunk(
            ref=ref,
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
        session.outputs.append((output, size))
        session.output_bytes += size
        session.next_output += 1
        session.output_wake.set()

    async def pump_session(self, session: Session) -> None:
        try:
            while not session.closing:
                if not session.pending:
                    session.wake.clear()
                    await asyncio.wait_for(
                        session.wake.wait(), session.limits.idle_timeout_s
                    )
                    continue
                async with session.unit_lock:
                    if session.closing:
                        break
                    chunk, size = session.pending.popleft()
                    ref = session.ref
                    try:
                        await self.session_command(session, "append", chunk=chunk)
                        self.emit_session_output(
                            session,
                            ref,
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

    async def session_command(
        self,
        session: Session,
        op: SessionOp,
        *,
        owner: str | None = None,
        chunk: TimedChunk | None = None,
    ) -> None:
        ref = session.ref
        command = SessionCommand(
            op=op,
            ref=ref,
            stages=session.stages,
            chunk=chunk,
        )
        request = replace(
            session.request,
            metadata={
                **session.request.metadata,
                SESSION_METADATA_KEY: command.to_dict(),
            },
        )
        request_id = f"session-{uuid.uuid4()}"

        if chunk is not None:
            input_seq = chunk.seq

            def output(msg: StreamMessage) -> None:
                try:
                    self.emit_session_output(
                        session, ref, input_seq, TimedChunk.from_dict(msg.chunk)
                    )
                except Exception as exc:
                    self._reject_completion_future(request_id, exc)

            self.session_stream_handlers[request_id] = output

        async def run() -> None:
            await self._submit_request(
                request_id,
                request,
                target_stage=owner,
                terminal_stages=(
                    {self._replica_topology.logical_name(owner)}
                    if owner
                    else {self._replica_topology.logical_name(session.stages[-1])}
                ),
                replica_bindings=session.bindings,
                bypass_admission=op in {"abort", "close"},
            )
            await self._completion_futures[request_id]

        try:
            await asyncio.wait_for(run(), session.limits.command_timeout_s)
        except asyncio.TimeoutError as exc:
            self.begin_session_close(session)
            raise TimeoutError(f"session {op} timed out") from exc
        except BaseException:
            # Note (Junnan Li): Request abort can yield before the pump sees this fatal failure.
            self.begin_session_close(session)
            raise
        finally:
            self.session_stream_handlers.pop(request_id, None)
            if request_id in self._requests:
                await self.abort(request_id)
            future = self._completion_futures.pop(request_id, None)
            if future is not None and not future.done():
                future.cancel()

    async def abort_session(self, ref: SessionRef) -> SessionRef:
        """Fence output immediately; finish the active unit before changing stage state."""

        async def run() -> SessionRef:
            session = self.get_session(ref)
            async with session.lock:
                if session.closing:
                    raise RuntimeError("session is closing")
                session.ref = replace(ref, epoch=ref.epoch + 1)
                session.outputs = deque(
                    (output, size)
                    for output, size in session.outputs
                    if output.kind == "input_done"
                )
                session.output_bytes = sum(size for _, size in session.outputs)
                try:
                    # Note (Junnan Li): Wait for the active unit; cancelling it would close the session.
                    async with session.unit_lock:
                        for owner in reversed(session.opened):
                            await self.session_command(session, "abort", owner=owner)
                except BaseException as exc:
                    session.error = exc
                    await self.cleanup_session(session)
                    raise
                return session.ref

        return await asyncio.shield(self.owned_session_task(run()))

    async def close_session(self, ref: SessionRef) -> None:
        """Close the referenced incarnation regardless of its current output epoch."""
        session = self.sessions.get(ref.session_id)
        if session is None:
            return
        if session.ref.incarnation != ref.incarnation:
            raise ValueError("stale session reference")
        await asyncio.shield(self.owned_session_task(self.close_session_state(session)))
        if session.cleanup_error is not None:
            raise RuntimeError(
                "session cleanup incomplete; capacity remains reserved"
            ) from session.cleanup_error

    def begin_session_close(self, session: Session) -> None:
        session.closing = True
        # Note (Junnan Li): Close fences output like cancel; queued data is dropped, not drained.
        session.outputs.clear()
        session.output_bytes = 0
        session.wake.set()
        session.output_wake.set()

    def close_session_state(self, session: Session) -> Coroutine[Any, Any, None]:
        self.begin_session_close(session)
        return self.finish_session_close(session)

    async def finish_session_close(self, session: Session) -> None:
        async with session.lock:
            if session.closed:
                return
            await self.cleanup_session(session)

    async def cleanup_session(self, session: Session) -> None:
        self.begin_session_close(session)
        if session.pump is not None and session.pump is not asyncio.current_task():
            try:
                await asyncio.wait_for(
                    asyncio.shield(session.pump), session.limits.command_timeout_s
                )
            except asyncio.TimeoutError as exc:
                session.cleanup_error = session.error = exc
                self.session_unavailable_stages.update(session.opened)
                session.pump.cancel()
                await asyncio.gather(session.pump, return_exceptions=True)
                session.closed = True
                session.output_wake.set()
                return
        session.pending.clear()
        session.pending_count = session.pending_bytes = 0
        unconfirmed = list(session.opened)
        for owner in reversed(session.opened):
            try:
                await self.session_command(session, "close", owner=owner)
                unconfirmed.pop()
            except Exception as exc:
                session.cleanup_error = exc
                session.error = session.error or exc
                # Note (Junnan Li): An unacknowledged downstream owner may still use upstream data.
                self.session_unavailable_stages.update(unconfirmed)
                break
        session.closed = True
        session.output_wake.set()
        # Note (Junnan Li): An unacknowledged owner may still hold buffers; keep its capacity reserved.
        if session.cleanup_error is None:
            del self.sessions[session.ref.session_id]

    async def shutdown_stage_sessions(self, selected: set[str] | None) -> None:
        affected = set(self._stages) if selected is None else selected
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
        self.sessions_stopping = True
        await asyncio.gather(
            *(self.close_session_state(s) for s in list(self.sessions.values()))
        )
        await asyncio.gather(*self.session_cleanup_tasks, return_exceptions=True)

    async def fail_sessions(self, message: str) -> None:
        for session in list(self.sessions.values()):
            session.error = RuntimeError(message)
        await self.stop_sessions()
