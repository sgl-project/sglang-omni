# SPDX-License-Identifier: Apache-2.0
"""Session graph: node chains joined by data and control edges.

Each node is an ordinary session over its own linear stage chain. The graph
owns one session per node, feeds client input to the input nodes, and routes
every node output by its modality: a data edge appends it to the target node,
a control edge delivers it as a control event, and the output node's data
goes back to the client.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import AsyncIterator, Coroutine
from dataclasses import replace
from typing import Protocol

from sglang_omni.admission import QueueFullError
from sglang_omni.config.schema import GraphConfig, GraphEdgeConfig
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import (
    OutputChunk,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
)

# Arbitrary; short enough that an edge does not add audible latency.
FULL_QUEUE_RETRY_S = 0.005


class SessionBackend(Protocol):
    """The session API shared by Coordinator and Client."""

    async def open_session(
        self,
        request: OmniRequest,
        *,
        stages: list[str],
        limits: SessionLimits | None = None,
        session_id: str | None = None,
    ) -> SessionIdentity: ...

    async def append_session(
        self, session_identity: SessionIdentity, chunk: TimedChunk
    ) -> int: ...

    def session_outputs(
        self, session_identity: SessionIdentity
    ) -> AsyncIterator[OutputChunk]: ...

    async def control_session(
        self,
        session_identity: SessionIdentity,
        event: TimedChunk,
        *,
        stages: list[str] | None = None,
        should_preempt: bool = False,
    ) -> None: ...

    async def close_session(self, session_identity: SessionIdentity) -> None: ...


class GraphSession:
    """One open of a session graph; any node failure closes the whole graph."""

    def __init__(
        self,
        backend: SessionBackend,
        graph: GraphConfig,
        node_sessions: dict[str, SessionIdentity],
        limits: SessionLimits,
    ) -> None:
        self.backend = backend
        self.graph = graph
        self.node_sessions = node_sessions
        self.limits = limits
        self.edges_by_source: dict[tuple[str, str], list[GraphEdgeConfig]] = {}
        for edge in graph.edges:
            self.edges_by_source.setdefault((edge.source, edge.modality), []).append(
                edge
            )
        self.node_inboxes: dict[str, asyncio.Queue[TimedChunk]] = {
            node_name: asyncio.Queue() for node_name in node_sessions
        }
        self.client_outputs: asyncio.Queue[OutputChunk | None] = asyncio.Queue(
            maxsize=limits.max_output_chunks
        )
        self.next_input = 0
        self.error: BaseException | None = None
        self.is_closing = False
        self.close_task: asyncio.Task[None] | None = None
        self.forward_tasks: list[asyncio.Task[None]] = []
        self.route_tasks: list[asyncio.Task[None]] = []

    @classmethod
    async def open(
        cls,
        backend: SessionBackend,
        graph: GraphConfig,
        request: OmniRequest,
        *,
        limits: SessionLimits | None = None,
        session_id: str | None = None,
    ) -> GraphSession:
        """Open every node in declaration order; a failed open closes the nodes already open."""
        resolved_limits = limits or SessionLimits()
        graph_id = session_id or str(uuid.uuid4())
        input_node_names = {
            node_name
            for node_names in graph.inputs.values()
            for node_name in node_names
        }
        node_sessions: dict[str, SessionIdentity] = {}
        try:
            for node_name, node in graph.nodes.items():
                node_sessions[node_name] = await backend.open_session(
                    request,
                    stages=node.stages,
                    # Note (Dayuxiaoshui): A node fed only by edges may idle for a whole conversation; client input nodes bound the graph's idle time.
                    limits=(
                        resolved_limits
                        if node_name in input_node_names
                        else replace(resolved_limits, idle_timeout_s=math.inf)
                    ),
                    session_id=f"{graph_id}:{node_name}",
                )
        except BaseException:
            for session_identity in reversed(node_sessions.values()):
                await backend.close_session(session_identity)
            raise
        graph_session = cls(backend, graph, node_sessions, resolved_limits)
        for node_name in node_sessions:
            graph_session.forward_tasks.append(
                graph_session.start_task(graph_session.forward_inputs(node_name))
            )
            graph_session.route_tasks.append(
                graph_session.start_task(graph_session.route_outputs(node_name))
            )
        return graph_session

    def start_task(self, coroutine: Coroutine[None, None, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)
        task.add_done_callback(self.task_done)
        return task

    def task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or self.is_closing:
            return
        else:
            self.error = (
                self.error
                or task.exception()
                or RuntimeError("graph node session ended")
            )
            self.close_task = self.close_task or asyncio.create_task(self.shutdown())

    async def append(self, chunk: TimedChunk) -> None:
        """Accept client input in seq order; rejected input keeps its seq for retry."""
        if self.is_closing:
            raise RuntimeError("graph session is closing")
        elif chunk.seq != self.next_input:
            raise ValueError("input seq must be contiguous")
        else:
            pass
        node_names = self.graph.inputs.get(chunk.modality)
        if node_names is None:
            raise ValueError(f"graph has no input for modality {chunk.modality!r}")
        elif any(
            self.node_inboxes[node_name].qsize() >= self.limits.max_pending_chunks
            for node_name in node_names
        ):
            raise QueueFullError()
        else:
            pass
        for node_name in node_names:
            self.node_inboxes[node_name].put_nowait(chunk)
        self.next_input += 1

    async def outputs(self) -> AsyncIterator[OutputChunk]:
        """Data chunks of the output node; ends when the graph closes."""
        while True:
            output = await self.client_outputs.get()
            if output is not None:
                yield output
            elif self.error is not None:
                raise self.error
            else:
                return

    async def close(self) -> None:
        self.close_task = self.close_task or asyncio.create_task(self.shutdown())
        await asyncio.shield(self.close_task)

    async def forward_inputs(self, node_name: str) -> None:
        """Append client and edge chunks to one node, renumbering them in that node's seq."""
        session_identity = self.node_sessions[node_name]
        inbox = self.node_inboxes[node_name]
        node_sequence = 0
        while True:
            node_chunk = replace(await inbox.get(), seq=node_sequence)
            while True:
                try:
                    await self.backend.append_session(session_identity, node_chunk)
                    break
                except QueueFullError:
                    await asyncio.sleep(FULL_QUEUE_RETRY_S)
            node_sequence += 1

    async def route_outputs(self, node_name: str) -> None:
        async for output in self.backend.session_outputs(self.node_sessions[node_name]):
            if output.kind == "input_done":
                continue
            else:
                pass
            event = TimedChunk(
                modality=output.modality,
                t_start_ms=output.t_start_ms,
                duration_ms=output.duration_ms,
                seq=0,
                payload=output.payload,
                format=output.format,
                eos=output.eos,
            )
            for edge in self.edges_by_source.get((node_name, output.modality), []):
                if edge.kind == "control":
                    await self.backend.control_session(
                        self.node_sessions[edge.target],
                        event,
                        stages=edge.target_stages,
                        should_preempt=edge.preempt,
                    )
                else:
                    self.node_inboxes[edge.target].put_nowait(event)
            if node_name == self.graph.output:
                await self.client_outputs.put(output)
            else:
                pass

    async def shutdown(self) -> None:
        """Stop input, close nodes in reverse declaration order, then stop routing."""
        self.is_closing = True
        for task in self.forward_tasks:
            task.cancel()
        await asyncio.gather(*self.forward_tasks, return_exceptions=True)
        # Note (Dayuxiaoshui): Readers stay alive here; cancelling one would close its node out of order.
        for session_identity in reversed(self.node_sessions.values()):
            try:
                await self.backend.close_session(session_identity)
            except Exception as exc:
                # Note (Dayuxiaoshui): Keep closing the other nodes; the first error surfaces to the reader.
                self.error = self.error or exc
        for task in self.route_tasks:
            task.cancel()
        await asyncio.gather(*self.route_tasks, return_exceptions=True)
        # Note (Dayuxiaoshui): Close fences output like a session close; queued data is dropped.
        while not self.client_outputs.empty():
            self.client_outputs.get_nowait()
        self.client_outputs.put_nowait(None)
