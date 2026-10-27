# SPDX-License-Identifier: Apache-2.0
"""Coordinator-owned session sequencing over ordinary pipeline requests."""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections import Counter, deque
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

import msgpack

from sglang_omni.admission import QueueFullError
from sglang_omni.pipeline.replicas import assign_replica_bindings
from sglang_omni.pipeline.session_delivery import DeliveryPacer
from sglang_omni.pipeline.types import SessionIdleTimeoutError, SessionOutputBudgetError
from sglang_omni.profiler.duplex_events import (
    emit_session_output_event,
    emit_session_unit_event,
    emit_session_unit_ready_event,
)
from sglang_omni.proto import OmniRequest, StreamMessage
from sglang_omni.proto.session import (
    DEFAULT_MAX_OUTPUT_LEAD_MS,
    SESSION_METADATA_KEY,
    OutputChunk,
    SessionGraph,
    SessionIdentity,
    SessionLimits,
    SessionOperation,
    SessionOperationKind,
    TimedChunk,
    wire_size,
)
from sglang_omni.proto.streams import INTERRUPT, TURN_DONE, control_type

logger = logging.getLogger(__name__)


class SessionStreamHandler(Protocol):
    def __call__(self, message: StreamMessage) -> None: ...


@dataclass(frozen=True, kw_only=True)
class PendingSessionChunk:
    chunk: TimedChunk
    encoded_bytes: int
    request_id: str
    # note (luojiaxuan): the chunk is a fragment the next input continues; it goes only to the route's first owner.
    continues: bool


def check_timing(chunk: TimedChunk, modality_end_ms: dict[str, float] | None) -> None:
    """Finite timing and, against a timeline per modality, one that does not move backwards."""
    if (
        not math.isfinite(chunk.t_start_ms)
        or not math.isfinite(chunk.duration_ms)
        or chunk.duration_ms < 0
    ):
        raise ValueError("input timing must be finite with a non-negative duration")
    elif modality_end_ms is None:
        pass
    elif chunk.t_start_ms < modality_end_ms.get(chunk.modality, 0):
        raise ValueError("input timing overlaps or moves backwards")
    else:
        pass


def admit_input(
    chunk: TimedChunk,
    limits: SessionLimits,
    modality_end_ms: dict[str, float],
    ended_modalities: set[str],
) -> tuple[TimedChunk, int]:
    """Input a session can take: none after its modality's EOS, within max_chunk_bytes and max_modalities; returns it with its wire size."""
    if chunk.modality in ended_modalities:
        raise ValueError("input after EOS")
    else:
        pass
    if isinstance(chunk.payload, bytes):
        encoded_bytes = wire_size(chunk.to_dict())
    else:
        # Note (Junnan Li): Snapshot a mutable payload so later caller edits cannot reach it.
        encoded = msgpack.packb(chunk.to_dict(), use_bin_type=True)
        encoded_bytes = len(encoded)
        chunk = TimedChunk.from_dict(msgpack.unpackb(encoded, raw=False))
    if encoded_bytes > limits.max_chunk_bytes:
        raise ValueError(
            f"input chunk is {encoded_bytes} bytes; max_chunk_bytes is "
            f"{limits.max_chunk_bytes}"
        )
    elif (
        chunk.modality not in modality_end_ms
        and len(modality_end_ms) >= limits.max_modalities
    ):
        raise QueueFullError()
    else:
        return chunk, encoded_bytes


def output_chunk(output: OutputChunk) -> TimedChunk:
    """A data output as the chunk a reader or a control operation receives."""
    return TimedChunk(
        output.modality,
        output.t_start_ms,
        output.duration_ms,
        0,
        output.payload,
        format=output.format,
        eos=output.eos,
        turn=output.turn,
    )


@dataclass(kw_only=True)
class Session:
    session_identity: SessionIdentity
    request: OmniRequest
    stages: tuple[str, ...]
    bindings: dict[str, int]
    limits: SessionLimits
    # A node of a graph merges its sources' clocks and ends; only a client's input keeps one timeline and one EOS per modality.
    is_node: bool = False
    opened: list[str] = field(default_factory=list)
    pending: deque[PendingSessionChunk] = field(default_factory=deque)
    pending_bytes: int = 0
    pending_count: int = 0
    # Note (Junnan Li): Waiting controls in arrival order, each with the input seq it arrived at; the pump stops before the first.
    control_barriers: deque[tuple[int, object]] = field(default_factory=deque)
    in_flight_append: Literal["accepted", "submitted"] | None = None
    # Note (Junnan Li): The appends the owners hold, by request id; an interrupt names those of its turn to every owner.
    submitted_appends: dict[str, TimedChunk] = field(default_factory=dict)
    outputs: deque[tuple[OutputChunk, int]] = field(default_factory=deque)
    output_bytes: int = 0
    pacer: DeliveryPacer | None = None
    next_input: int = 0
    next_output: int = 0
    next_delivered: int = 0
    modality_end_ms: dict[str, float] = field(default_factory=dict)
    ended_modalities: set[str] = field(default_factory=set)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    output_wake: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pump: asyncio.Task[None] | None = None
    # The read of the terminal owner: what the stage emits outside an append arrives on it.
    read_task: asyncio.Task[None] | None = None
    is_closing: bool = False
    is_closed: bool = False
    has_output_consumer: bool = False
    error: BaseException | None = None
    cleanup_error: BaseException | None = None


@dataclass(kw_only=True)
class NodeInbox:
    """What one node of a graph has not taken yet: (source, chunk, client seq) in arrival order, the chunks each source's edge holds, and `wake`, set whenever either changes."""

    queued: deque[tuple[str, TimedChunk, int | None]] = field(default_factory=deque)
    counts: Counter[str] = field(default_factory=Counter)
    wake: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(kw_only=True)
class SessionGraphRuntime:
    """One open of a session graph: one node session per node, routed by format.

    Client input goes to every node reading `input`; a node output goes to
    the nodes reading that format from it and, when published, to the
    client. Each node numbers its own input, so `client_sequences` maps a node's
    input seq back to the client seq it carries, and the client's input_done
    arrives once every input node has confirmed its copy. A node's inbox
    holds what its edges carry until its node session accepts it, each
    edge within its own bound. A node that publishes audio publishes
    through its pacer in `pacers`.
    """

    session_identity: SessionIdentity
    graph: SessionGraph
    nodes: dict[str, SessionIdentity]
    limits: SessionLimits
    inboxes: dict[str, NodeInbox]
    outputs: asyncio.Queue[tuple[OutputChunk, int, str] | None]
    output_bytes: int = 0
    pacers: dict[str, DeliveryPacer] = field(default_factory=dict)
    # note (Junnan Li): Interrupted reply IDs remain fenced until the graph closes; late tools may outlive later replies.
    interrupted_turns: dict[str, set[int]] = field(default_factory=dict)
    client_sequences: dict[str, dict[int, int]] = field(default_factory=dict)
    pending_input_done: dict[int, int] = field(default_factory=dict)
    output_end_ms: float = 0.0
    reply_start_ms: float = 0.0
    open_turns: set[int] = field(default_factory=set)
    eos_confirmed: set[str] = field(default_factory=set)
    eos_wake: asyncio.Event = field(default_factory=asyncio.Event)
    end_input: TimedChunk | None = None
    drain_task: asyncio.Task[None] | None = None
    modality_end_ms: dict[str, float] = field(default_factory=dict)
    ended_modalities: set[str] = field(default_factory=set)
    next_input: int = 0
    next_output: int = 0
    tasks: list[asyncio.Task[None]] = field(default_factory=list)
    forwarders: list[asyncio.Task[None]] = field(default_factory=list)
    error: BaseException | None = None
    cleanup_error: BaseException | None = None
    is_closing: bool = False
    has_output_consumer: bool = False
    close_task: asyncio.Task[None] | None = None


class CoordinatorSessions:
    """Coordinator-owned sessions over fixed stage routes and graphs of them."""

    def __init__(self, max_output_lead_ms: float = DEFAULT_MAX_OUTPUT_LEAD_MS) -> None:
        self.max_output_lead_ms = max_output_lead_ms
        self.is_sessions_stopping = False
        self.next_open_index = 1
        self.session_unavailable_stages: set[str] = set()
        self.sessions: dict[str, Session] = {}
        self.graphs: dict[str, SessionGraphRuntime] = {}
        self.session_route: Callable[[], list[str]] | None = None
        self.session_graph: SessionGraph | None = None
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
        session = self.sessions.get(session_identity.id)
        if session is None or session.session_identity != session_identity:
            raise ValueError("unknown or stale session reference")
        else:
            pass
        return session

    async def open_session(
        self,
        request: OmniRequest,
        *,
        stages: list[str] | None = None,
        limits: SessionLimits | None = None,
        session_id: str | None = None,
    ) -> SessionIdentity:
        """Open an explicit route, or else the pipeline's declared Session, before accepting input."""
        if stages is not None:
            return await self.open_route(
                request, stages=stages, limits=limits, session_id=session_id
            )
        elif self.session_graph is not None:
            return await self.open_graph(
                request, graph=self.session_graph, limits=limits, session_id=session_id
            )
        elif self.session_route is not None:
            return await self.open_route(
                request,
                stages=self.session_route(),
                limits=limits,
                session_id=session_id,
            )
        else:
            raise ValueError("this pipeline declares no Session; pass stages")

    async def open_route(
        self,
        request: OmniRequest,
        *,
        stages: list[str],
        limits: SessionLimits | None,
        session_id: str | None,
        is_node: bool = False,
    ) -> SessionIdentity:
        """Open fixed owners in upstream order before accepting input; a client route begins at entry_stage."""
        if (
            self.is_sessions_stopping
            or not self.running
            or self.fatal_error is not None
        ):
            raise RuntimeError(self.fatal_error or "Coordinator is not running")
        else:
            pass
        session_id = session_id or str(uuid.uuid4())
        bindings, owners = self.resolve_route_owners(stages, session_id, is_node)
        session = Session(
            session_identity=SessionIdentity(session_id, self.next_open_index),
            request=request,
            stages=owners,
            bindings=bindings,
            limits=limits or SessionLimits(),
            is_node=is_node,
        )
        if not is_node:
            session.pacer = DeliveryPacer(self.max_output_lead_ms, session.output_wake)
        else:
            pass
        self.next_open_index += 1
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
                session.request = replace(request, inputs=None)
                # Note (Junnan Li): The read is sent before any append, so the owner attaches it before an output can be emitted.
                request_id = await self.submit_session_operation(
                    session, "read", owner=session.stages[-1]
                )
                session.read_task = asyncio.create_task(
                    self.read_session(session, request_id)
                )
        except BaseException:
            await asyncio.shield(
                self.owned_session_task(self.close_session_state(session))
            )
            raise
        session.pump = asyncio.create_task(self.pump_session(session))
        return session.session_identity

    def resolve_route_owners(
        self, stages: list[str], session_id: str, is_node: bool
    ) -> tuple[dict[str, int], tuple[str, ...]]:
        """Check a route and resolve its stages to the registered, available owners of this session's replicas."""
        if (
            not stages
            or (stages[0] != self.entry_stage and not is_node)
            or len(set(stages)) != len(stages)
        ):
            raise ValueError("stages must be a unique route beginning at entry_stage")
        elif session_id in self.sessions or session_id in self.graphs:
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
        elif self.session_unavailable_stages.intersection(owners):
            raise ValueError(
                "session route contains an unregistered owner or unavailable owner"
            )
        else:
            return bindings, owners

    async def append_session(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        *,
        ready_timestamp_ns: int | None = None,
        ready_run_id: str | None = None,
        continues: bool = False,
    ) -> int:
        """Accept input in global seq order, independently of output consumption.

        Rejected input keeps its seq for retry; accepted input must not be
        resubmitted. A chunk that `continues` is a fragment of a graph node's
        chunk over max_chunk_bytes, and the next input continues it.
        """
        graph = self.graphs.get(session_identity.id)
        if graph is not None and graph.session_identity == session_identity:
            return self.append_graph(graph, chunk)
        else:
            pass
        session = self.get_session(session_identity)
        if session.is_closing or session.is_closed:
            raise RuntimeError("session is closing")
        else:
            pass
        if chunk.seq != session.next_input:
            raise ValueError("input seq must be contiguous within an open index")
        else:
            pass
        check_timing(chunk, None if session.is_node else session.modality_end_ms)
        chunk, encoded_bytes = admit_input(
            chunk, session.limits, session.modality_end_ms, session.ended_modalities
        )
        if (
            session.pending_count >= session.limits.max_pending_chunks
            or session.pending_bytes + encoded_bytes > session.limits.max_pending_bytes
        ):
            raise QueueFullError()
        else:
            pass
        request_id = f"session-{uuid.uuid4()}"
        session.pending.append(
            PendingSessionChunk(
                chunk=chunk,
                encoded_bytes=encoded_bytes,
                request_id=request_id,
                continues=continues,
            )
        )
        session.pending_count += 1
        session.pending_bytes += encoded_bytes
        session.next_input += 1
        session.modality_end_ms[chunk.modality] = chunk.t_start_ms + chunk.duration_ms
        if chunk.eos and not session.is_node:
            session.ended_modalities.add(chunk.modality)
        else:
            pass
        emit_session_unit_ready_event(
            request_id=request_id,
            session_identity=session.session_identity,
            input_chunk=chunk,
            ready_timestamp_ns=ready_timestamp_ns,
            ready_run_id=ready_run_id,
        )
        emit_session_unit_event(
            request_id=request_id,
            event_name="session_unit_admitted",
            session_identity=session.session_identity,
            input_chunk=chunk,
        )

        if chunk.eos and session.pacer is not None:
            session.pacer.end_input()
        else:
            pass
        session.wake.set()
        return chunk.seq

    async def session_outputs(
        self, session_identity: SessionIdentity
    ) -> AsyncIterator[OutputChunk]:
        """One output consumer; disconnect closes the owned session.

        A client route delivers through its pacer and numbers what it
        delivers; a released turn.done returns to the route as a control.
        """
        graph = self.graphs.get(session_identity.id)
        if graph is not None and graph.session_identity == session_identity:
            async for output in self.graph_outputs(graph):
                yield output
            return
        else:
            pass
        session = self.get_session(session_identity)
        if session.has_output_consumer:
            raise RuntimeError("session already has an output consumer")
        else:
            pass
        session.has_output_consumer = True
        loop = asyncio.get_running_loop()
        try:
            while True:
                session.output_wake.clear()
                if session.pacer is None:
                    while session.outputs and not session.is_closing:
                        output, size = session.outputs.popleft()
                        session.output_bytes -= size
                        yield output
                else:
                    while session.outputs and not session.is_closing:
                        output, size = session.outputs.popleft()
                        session.output_bytes -= session.pacer.offer(output, size)
                    session.pacer.unsent.extend(session.pacer.release(loop.time()))
                    while session.pacer.unsent and not session.is_closing:
                        output, size = session.pacer.unsent.popleft()
                        session.output_bytes -= size
                        self.return_turn_end(session.session_identity, output)
                        seq = session.next_delivered
                        session.next_delivered += 1
                        yield replace(output, seq=seq)
                if session.is_closed:
                    if session.error is not None:
                        raise session.error
                    else:
                        pass
                    return
                elif session.pacer is None or session.is_closing:
                    await session.output_wake.wait()
                else:
                    await session.pacer.wait()
        finally:
            session.has_output_consumer = False
            await asyncio.shield(
                self.owned_session_task(self.close_session_state(session))
            )

    def emit_session_output(
        self,
        session: Session,
        request_id: str,
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
            turn=chunk.turn,
            kind=kind,
        )
        size = wire_size(output.to_dict())
        if (
            len(session.outputs) >= session.limits.max_output_chunks
            or session.output_bytes + size > session.limits.max_output_bytes
        ):
            raise SessionOutputBudgetError()
        else:
            pass
        if kind == "data":
            emit_session_output_event(request_id=request_id, output_chunk=output)
        else:
            emit_session_unit_event(
                request_id=request_id,
                event_name="session_unit_finished",
                session_identity=session.session_identity,
                input_chunk=chunk,
            )
        session.outputs.append((output, size))
        session.output_bytes += size
        session.next_output += 1
        session.output_wake.set()

    async def pump_session(self, session: Session) -> None:
        try:
            while not session.is_closing:
                if not session.pending or (
                    session.control_barriers
                    and session.pending[0].chunk.seq >= session.control_barriers[0][0]
                ):
                    session.wake.clear()
                    try:
                        await asyncio.wait_for(
                            session.wake.wait(), session.limits.idle_timeout_s
                        )
                    except TimeoutError as exc:
                        raise SessionIdleTimeoutError(
                            session.limits.idle_timeout_s
                        ) from exc
                    else:
                        continue
                else:
                    pass
                pending_chunk = session.pending.popleft()
                chunk = pending_chunk.chunk
                session.in_flight_append = "accepted"
                try:
                    operation = "fragment" if pending_chunk.continues else "append"
                    request_id = await self.submit_session_operation(
                        session,
                        operation,
                        owner=session.stages[0] if pending_chunk.continues else None,
                        chunk=chunk,
                        request_id=pending_chunk.request_id,
                    )
                    if operation == "append":
                        session.submitted_appends[request_id] = chunk
                    else:
                        pass
                    session.in_flight_append = "submitted"
                    session.wake.set()
                    await self.finish_session_operation(session, operation, request_id)
                    self.emit_session_output(
                        session,
                        pending_chunk.request_id,
                        chunk.seq,
                        replace(chunk, payload=None),
                        kind="input_done",
                    )
                finally:
                    session.in_flight_append = None
                    session.pending_count -= 1
                    session.pending_bytes -= pending_chunk.encoded_bytes
                    session.wake.set()
        except Exception as exc:
            self.fail_session(session, exc)

    def fail_session(self, session: Session, error: BaseException) -> None:
        """End the session with its first error; its output stream raises it."""
        if session.error is None:
            logger.warning(
                "session %s failed", session.session_identity.id, exc_info=error
            )
        else:
            pass
        session.error = session.error or error
        self.owned_session_task(self.close_session_state(session))

    async def read_session(self, session: Session, request_id: str) -> None:
        """Hold the terminal owner's read until that owner closes the session; a read that ends any other way fails the session."""
        try:
            await self.finish_session_operation(session, "read", request_id)
            error = (
                None
                if session.is_closing
                else RuntimeError(
                    f"session owner {session.stages[-1]!r} closed the session"
                )
            )
        except Exception as exc:
            error = exc
        if error is not None:
            self.fail_session(session, error)
        else:
            pass

    async def control_session(
        self,
        session_identity: SessionIdentity,
        event: TimedChunk,
        *,
        stages: list[str] | None = None,
        should_preempt: bool = False,
        is_ordered: bool = True,
        preempted_request_ids: tuple[str, ...] = (),
    ) -> None:
        """Deliver a control event to the named logical stages, or to every stage of the route.

        An ordered control keeps its arrival order among the appends and
        controls of the session; with should_preempt it goes out once the
        earlier appends have reached their owners, which cancel the ones
        still in flight. A control that is not ordered goes out at once and
        only holds the appends after it. A graph orders the client's
        controls and sends its node control edges' events at once.
        """
        graph = self.graphs.get(session_identity.id)
        if graph is not None and graph.session_identity == session_identity:
            await self.control_graph(graph, event, stages, should_preempt)
            return
        else:
            pass
        session = self.get_session(session_identity)
        if session.is_closing or session.is_closed:
            raise RuntimeError("session is closing")
        elif control_type(event) == INTERRUPT and session.pacer is not None:
            session.output_bytes -= session.pacer.purge(event.turn)
        else:
            pass
        owners = [
            owner
            for owner in session.stages
            if stages is None or self.replica_topology.logical_name(owner) in stages
        ]
        if stages is not None and len(owners) != len(stages):
            raise ValueError(
                f"control stages {stages} are not all on the session route"
            )
        else:
            pass
        barrier = (session.next_input, object())
        session.control_barriers.append(barrier)
        try:
            while is_ordered:
                if session.is_closing:
                    raise RuntimeError("session is closing")
                elif (
                    session.control_barriers[0] is barrier
                    and (
                        not session.pending
                        or session.pending[0].chunk.seq >= barrier[0]
                    )
                    and (
                        session.in_flight_append is None
                        or (should_preempt and session.in_flight_append == "submitted")
                    )
                ):
                    break
                else:
                    session.wake.clear()
                    await session.wake.wait()
            for owner in owners:
                await self.session_operation(
                    session,
                    "control",
                    owner=owner,
                    chunk=event,
                    should_preempt=should_preempt,
                    preempted_request_ids=preempted_request_ids,
                )
            # Note (Junnan Li): Every owner aborted these appends as the control arrived, so none sends their completion.
            for request_id in preempted_request_ids:
                future = self.completion_futures.get(request_id)
                if future is not None and not future.done():
                    future.set_result(None)
                else:
                    pass
        finally:
            session.control_barriers.remove(barrier)
            session.wake.set()

    async def session_operation(
        self,
        session: Session,
        operation: SessionOperationKind,
        *,
        owner: str | None = None,
        chunk: TimedChunk | None = None,
        request_id: str | None = None,
        should_preempt: bool = False,
        preempted_request_ids: tuple[str, ...] = (),
    ) -> None:
        request_id = await self.submit_session_operation(
            session,
            operation,
            owner=owner,
            chunk=chunk,
            request_id=request_id,
            should_preempt=should_preempt,
            preempted_request_ids=preempted_request_ids,
        )
        await self.finish_session_operation(session, operation, request_id)

    async def submit_session_operation(
        self,
        session: Session,
        operation: SessionOperationKind,
        *,
        owner: str | None = None,
        chunk: TimedChunk | None = None,
        request_id: str | None = None,
        should_preempt: bool = False,
        preempted_request_ids: tuple[str, ...] = (),
    ) -> str:
        """Send one operation to its owner, or an append along the route; the chunks an append or a read streams back enter the session's outputs."""
        session_identity = session.session_identity
        session_operation = SessionOperation(
            operation=operation,
            session_identity=session_identity,
            stages=session.stages,
            chunk=chunk,
            should_preempt=should_preempt,
            preempted_request_ids=preempted_request_ids,
        )
        request = replace(
            session.request,
            metadata={
                **session.request.metadata,
                SESSION_METADATA_KEY: session_operation.to_dict(),
            },
        )
        if request_id is None:
            request_id = f"session-{uuid.uuid4()}"
        else:
            pass

        if operation in ("append", "read"):

            def emit_operation_output(message: StreamMessage) -> None:
                # Note (Junnan Li): A read's chunk belongs to the latest input the session accepted.
                input_seq = (
                    chunk.seq if chunk is not None else max(session.next_input - 1, 0)
                )
                try:
                    self.emit_session_output(
                        session,
                        request_id,
                        input_seq,
                        TimedChunk.from_dict(message.chunk),
                    )
                except Exception as exc:
                    self.reject_completion_future(request_id, exc)

            self.session_stream_handlers[request_id] = emit_operation_output
        else:
            pass
        try:
            await self.submit_request(
                request_id,
                request,
                target_stage=owner or session.stages[0],
                terminal_stages=(
                    {self.replica_topology.logical_name(owner)}
                    if owner
                    else {self.replica_topology.logical_name(session.stages[-1])}
                ),
                replica_bindings=session.bindings,
                should_bypass_admission=operation in ("close", "read"),
            )
        except BaseException:
            self.session_stream_handlers.pop(request_id, None)
            self.begin_session_close(session)
            raise
        return request_id

    async def finish_session_operation(
        self, session: Session, operation: SessionOperationKind, request_id: str
    ) -> None:
        """Await the operation's completion within the operation timeout; a close and a read have none."""

        async def run() -> None:
            await self.completion_futures[request_id]

        try:
            if operation in ("close", "read"):
                await run()
            else:
                await asyncio.wait_for(run(), session.limits.operation_timeout_s)
        except asyncio.TimeoutError as exc:
            error = TimeoutError(f"session {operation} timed out")
            self.fail_session(session, error)
            raise error from exc
        except Exception as exc:
            # Note (Junnan Li): Fail the session here, not in the pump: the abort below yields, and a pump that wakes to the fenced session exits without seeing this failure.
            self.fail_session(session, exc)
            raise
        except BaseException as exc:
            logger.warning(
                "session %s %s failed: %r", session.session_identity.id, operation, exc
            )
            self.begin_session_close(session)
            future = self.completion_futures.get(request_id)
            if (
                isinstance(exc, asyncio.CancelledError)
                and future is not None
                and future.done()
                and not future.cancelled()
                and isinstance(future.exception(), asyncio.CancelledError)
            ):
                # note (Richard Wang): an abort of this operation's request, such
                # as an admin drop, is a session failure, not a cancel of the
                # pump, so the pump still closes the session and its owners.
                error = RuntimeError(f"session {operation} was aborted")
                self.fail_session(session, error)
                raise error from exc
            else:
                raise
        finally:
            session.submitted_appends.pop(request_id, None)
            self.session_stream_handlers.pop(request_id, None)
            future = self.completion_futures.get(request_id)
            if future is not None:
                await self.release_submission(request_id, future)
            else:
                pass

    async def close_session(self, session_identity: SessionIdentity) -> None:
        graph = self.graphs.get(session_identity.id)
        if graph is not None and graph.session_identity == session_identity:
            await self.close_graph(graph)
            if graph.cleanup_error is not None:
                raise graph.cleanup_error
            else:
                return
        else:
            pass
        session = self.sessions.get(session_identity.id)
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
        if session.pacer is not None:
            session.pacer.items.clear()
            session.pacer.unsent.clear()
        else:
            pass
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

    def fail_session_cleanup(self, session: Session, error: BaseException) -> None:
        """A cleanup step failed: the session closes with the error and its owners stay reserved."""
        session.cleanup_error = session.error = error
        self.session_unavailable_stages.update(session.opened)
        session.is_closed = True
        session.output_wake.set()

    async def cleanup_session(self, session: Session) -> None:
        self.begin_session_close(session)
        cleanup_deadline_s = (
            asyncio.get_running_loop().time() + session.limits.operation_timeout_s
        )
        if session.read_task is not None:
            session.read_task.cancel()
            await asyncio.gather(session.read_task, return_exceptions=True)
        else:
            pass
        if session.pump is not None and session.pump is not asyncio.current_task():
            try:
                await asyncio.wait_for(
                    asyncio.shield(session.pump),
                    max(0, cleanup_deadline_s - asyncio.get_running_loop().time()),
                )
            except asyncio.TimeoutError as exc:
                session.pump.cancel()
                await asyncio.gather(session.pump, return_exceptions=True)
                self.fail_session_cleanup(session, exc)
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
            del self.sessions[session.session_identity.id]
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
        for graph in list(self.graphs.values()):
            graph.error = RuntimeError(message)
        await self.stop_sessions()

    def return_turn_end(
        self, session_identity: SessionIdentity, output: OutputChunk
    ) -> None:
        """A published turn.done returns to the node that published it as a control once delivered, through its pacer when it has one: the reply has reached the client."""
        if control_type(output) == TURN_DONE:
            self.owned_session_task(
                self.control_session(
                    session_identity, output_chunk(output), is_ordered=False
                )
            )
        else:
            pass
