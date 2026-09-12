# SPDX-License-Identifier: Apache-2.0
"""Coordinator-owned session sequencing over ordinary pipeline requests."""
from __future__ import annotations

import asyncio
import secrets
import uuid
from collections import deque
from collections.abc import Coroutine
from dataclasses import asdict, dataclass, field, replace
from typing import Any, AsyncIterator

import msgpack

from sglang_omni.admission import QueueFullError
from sglang_omni.pipeline.replicas import assign_replica_bindings
from sglang_omni.proto import OmniRequest, StreamMessage
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    OutputChunk,
    SessionLimits,
    SessionRef,
    TimedChunk,
    wire_size,
)


@dataclass
class _Session:
    ref: SessionRef
    request: OmniRequest
    stages: tuple[str, ...]
    bindings: dict[str, int]
    limits: SessionLimits
    opened: list[str] = field(default_factory=list)
    pending: deque = field(default_factory=deque)
    pending_bytes: int = 0
    pending_count: int = 0
    outputs: deque = field(default_factory=deque)
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

    def _init_sessions(self, max_sessions: int) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = max_sessions
        self._sessions_stopping = False
        self._session_unavailable_stages: set[str] = set()
        self._sessions: dict[str, _Session] = {}
        self._session_stream_handlers: dict[str, Any] = {}
        self._session_cleanup_tasks: set[asyncio.Task] = set()

    def _owned_session_task(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._session_cleanup_tasks.add(task)
        task.add_done_callback(self._session_task_done)
        return task

    def _session_task_done(self, task: asyncio.Task) -> None:
        self._session_cleanup_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def _session(self, ref: SessionRef) -> _Session:
        session = self._sessions.get(ref.session_id)
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
        if (
            self._sessions_stopping
            or not self._running
            or self._fatal_error is not None
        ):
            raise RuntimeError(self._fatal_error or "Coordinator is not running")
        if (
            not stages
            or stages[0] != self.entry_stage
            or len(set(stages)) != len(stages)
        ):
            raise ValueError("stages must be a unique route beginning at entry_stage")
        if len(self._sessions) >= self.max_sessions:
            raise QueueFullError()
        session_id = session_id or str(uuid.uuid4())
        if session_id in self._sessions:
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
        if self._session_unavailable_stages.intersection(owners):
            raise ValueError(
                "session route contains an unregistered owner or unavailable owner"
            )
        session = _Session(
            SessionRef(session_id, secrets.randbelow((1 << 63) - 1) + 1),
            request,
            owners,
            bindings,
            limits or SessionLimits(),
        )
        self._sessions[session_id] = session
        try:
            async with session.lock:
                for owner in owners:
                    # Include attempts: open may allocate before its reply is lost.
                    session.opened.append(owner)
                    await self._session_command(session, "open", owner=owner)
                if (
                    self._sessions_stopping
                    or self._session_unavailable_stages.intersection(owners)
                ):
                    raise RuntimeError("session owners are shutting down")
        except BaseException:
            await asyncio.shield(
                self._owned_session_task(self._close_session_state(session))
            )
            raise
        session.request = replace(request, inputs=None)
        session.pump = asyncio.create_task(self._pump_session(session))
        return session.ref

    async def append_session(self, ref: SessionRef, chunk: TimedChunk) -> int:
        """Accept input in global seq order, independently of output consumption.

        Adapters map per-stream seq to this order. Rejected input keeps its seq
        for retry; accepted input advances it and must not be resubmitted.
        """
        session = self._session(ref)
        if session.closing or session.closed:
            raise RuntimeError("session is closing")
        if chunk.seq != session.next_input:
            raise ValueError("input seq must be contiguous within an incarnation")
        if chunk.modality in session.eos:
            raise ValueError("input after EOS")
        if chunk.t_start_ms < session.ends.get(chunk.modality, 0):
            raise ValueError("input timing overlaps or moves backwards")
        if isinstance(chunk.payload, bytes):
            size = wire_size(asdict(chunk))
        else:
            encoded = msgpack.packb(asdict(chunk), use_bin_type=True)
            size = len(encoded)
        limits = session.limits
        if (
            size > limits.max_chunk_bytes
            or session.pending_count >= limits.max_pending_chunks
            or session.pending_bytes + size > limits.max_pending_bytes
        ):
            raise QueueFullError()
        if (
            chunk.modality not in session.ends
            and len(session.ends) >= limits.max_modalities
        ):
            raise QueueFullError()
        if not isinstance(chunk.payload, bytes):
            chunk = TimedChunk(**msgpack.unpackb(encoded, raw=False))
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
        session = self._session(ref)
        if session.reading:
            raise RuntimeError("session already has an output consumer")
        session.reading = True
        try:
            while True:
                while session.outputs:
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
                self._owned_session_task(self._close_session_state(session))
            )

    def _emit_session_output(
        self,
        session: _Session,
        ref: SessionRef,
        input_seq: int,
        chunk: TimedChunk,
        *,
        kind: str = "data",
    ) -> None:
        if session.closing or (session.ref != ref and kind != "input_done"):
            return
        output = OutputChunk(
            ref,
            session.next_output,
            input_seq,
            chunk.modality,
            chunk.t_start_ms,
            chunk.duration_ms,
            chunk.payload,
            chunk.format,
            chunk.eos,
            kind,
        )
        size = wire_size(asdict(output))
        if (
            len(session.outputs) >= session.limits.max_output_chunks
            or session.output_bytes + size > session.limits.max_output_bytes
        ):
            raise QueueFullError()
        session.outputs.append((output, size))
        session.output_bytes += size
        session.next_output += 1
        session.output_wake.set()

    async def _pump_session(self, session: _Session) -> None:
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
                        await self._session_command(session, "append", chunk=chunk)
                        self._emit_session_output(
                            session,
                            ref,
                            chunk.seq,
                            replace(chunk, payload=None),
                            kind="input_done",
                        )
                    finally:
                        session.pending_count -= 1
                        session.pending_bytes -= size
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            session.error = exc
            self._owned_session_task(self._close_session_state(session))

    async def _session_command(
        self,
        session: _Session,
        op: str,
        *,
        owner: str | None = None,
        chunk: TimedChunk | None = None,
    ) -> Any:
        ref = session.ref
        command = {
            "op": op,
            "ref": asdict(ref),
            "stages": list(session.stages),
            "output_limits": {
                "chunks": session.limits.max_output_chunks,
                "bytes": session.limits.max_output_bytes,
            },
        }
        if chunk is not None:
            command["chunk"] = asdict(chunk)
        request = replace(
            session.request,
            metadata={**session.request.metadata, SESSION_METADATA_KEY: command},
        )
        request_id = f"session-{uuid.uuid4()}"

        def output(msg: StreamMessage) -> None:
            try:
                self._emit_session_output(
                    session, ref, chunk.seq, TimedChunk(**msg.chunk)
                )
            except Exception as exc:
                self._reject_completion_future(request_id, exc)

        if chunk is not None:
            self._session_stream_handlers[request_id] = output
        try:
            async with asyncio.timeout(session.limits.command_timeout_s):
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
                return await self._completion_futures[request_id]
        except BaseException:
            # Note (Junnan Li): Request abort can yield before the pump sees this fatal failure.
            self._begin_session_close(session)
            raise
        finally:
            self._session_stream_handlers.pop(request_id, None)
            if request_id in self._requests:
                await self.abort(request_id)
            future = self._completion_futures.pop(request_id, None)
            if future is not None and not future.done():
                future.cancel()

    async def abort_session(self, ref: SessionRef) -> SessionRef:
        """Fence output immediately; finish the active unit before changing stage state."""
        task = self._owned_session_task(self._abort_session(ref))
        return await asyncio.shield(task)

    async def _abort_session(self, ref: SessionRef) -> SessionRef:
        session = self._session(ref)
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
                # Normal unit completion transfers KV back to its core session.
                # Canceling the pump here would abort that request and free KV.
                async with session.unit_lock:
                    for owner in reversed(session.opened):
                        await self._session_command(session, "abort", owner=owner)
            except BaseException as exc:
                session.error = exc
                await self._cleanup_session(session)
                raise
            return session.ref

    async def close_session(self, ref: SessionRef) -> None:
        """Close the referenced incarnation regardless of its current output epoch."""
        session = self._sessions.get(ref.session_id)
        if session is None:
            return
        if session.ref.incarnation != ref.incarnation:
            raise ValueError("stale session reference")
        await asyncio.shield(
            self._owned_session_task(self._close_session_state(session))
        )
        if session.cleanup_error is not None:
            raise RuntimeError(
                "session cleanup incomplete; capacity remains reserved"
            ) from session.cleanup_error

    def _begin_session_close(self, session: _Session) -> None:
        session.closing = True
        session.wake.set()

    def _close_session_state(self, session: _Session) -> Coroutine[Any, Any, None]:
        self._begin_session_close(session)
        return self._finish_session_close(session)

    async def _finish_session_close(self, session: _Session) -> None:
        async with session.lock:
            if session.closed:
                return
            await self._cleanup_session(session)

    async def _cleanup_session(self, session: _Session) -> None:
        session.closing = True
        session.wake.set()
        if session.pump is not None and session.pump is not asyncio.current_task():
            try:
                await asyncio.wait_for(
                    asyncio.shield(session.pump), session.limits.command_timeout_s
                )
            except asyncio.TimeoutError as exc:
                session.cleanup_error = session.error = exc
                self._session_unavailable_stages.update(session.opened)
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
                await self._session_command(session, "close", owner=owner)
                unconfirmed.pop()
            except Exception as exc:
                session.cleanup_error = exc
                session.error = session.error or exc
                # An unacknowledged downstream owner may still use upstream data.
                self._session_unavailable_stages.update(unconfirmed)
                break
        session.closed = True
        session.output_wake.set()
        # Never reclaim capacity on an unacknowledged close: a worker may still
        # own buffers or be finishing a command. Worker teardown owns that case.
        if session.cleanup_error is None:
            self._sessions.pop(session.ref.session_id, None)

    async def _shutdown_stage_sessions(self, selected: set[str] | None) -> None:
        affected = set(self._stages) if selected is None else selected
        self._session_unavailable_stages.update(affected)
        if selected is None:
            await self._stop_sessions()
        else:
            await asyncio.gather(
                *(
                    self._close_session_state(session)
                    for session in list(self._sessions.values())
                    if affected.intersection(session.stages)
                )
            )

    async def _stop_sessions(self) -> None:
        self._sessions_stopping = True
        await asyncio.gather(
            *(self._close_session_state(s) for s in list(self._sessions.values()))
        )
        await asyncio.gather(*self._session_cleanup_tasks, return_exceptions=True)

    async def _fail_sessions(self, message: str) -> None:
        for session in list(self._sessions.values()):
            session.error = RuntimeError(message)
        await self._stop_sessions()
