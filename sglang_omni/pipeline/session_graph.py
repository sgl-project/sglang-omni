# SPDX-License-Identifier: Apache-2.0
"""The Session graph runtime: node sessions joined by bounded edges, control edges and paced delivery."""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections import Counter, deque
from collections.abc import AsyncIterator, Coroutine
from dataclasses import replace

from sglang_omni.admission import QueueFullError
from sglang_omni.pipeline.session_delivery import DeliveryPacer
from sglang_omni.pipeline.sessions import (
    CoordinatorSessions,
    NodeInbox,
    SessionGraphRuntime,
    admit_input,
    check_timing,
    output_chunk,
)
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import (
    GRAPH_INPUT,
    ControlEdge,
    OutputChunk,
    SessionGraph,
    SessionIdentity,
    SessionLimits,
    TimedChunk,
    wire_size,
)
from sglang_omni.proto.streams import (
    AUDIO,
    AUDIO_FORMATS,
    CONTROL,
    INTERRUPT,
    NODE_CONTROLS,
    REPLY_DELIVERED,
    REPORT_CONTROLS,
    SESSION_CONTROL,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    control_type,
    edge_event,
    holds_turn_end,
    is_interrupted_reply,
)

logger = logging.getLogger(__name__)

# Note (Junnan Li): A node's pending budget is a few units; a retry this short adds no audible latency.
FULL_QUEUE_RETRY_S = 0.005


class SessionGraphRunner(CoordinatorSessions):
    """Runs the coordinator's Session graphs over its node sessions."""

    async def control_graph(
        self,
        graph: SessionGraphRuntime,
        event: TimedChunk,
        stages: list[str] | None,
        should_preempt: bool,
    ) -> None:
        targets = graph.graph.control_targets(GRAPH_INPUT, control_type(event))
        if graph.is_closing:
            raise RuntimeError("session is closing")
        elif stages is not None:
            raise ValueError("a graph delivers a control along its control edges")
        elif not targets:
            raise ValueError(
                f"no node takes the control {control_type(event)!r} from {GRAPH_INPUT!r}"
            )
        else:
            pass
        for target, edge in targets:
            await self.control_edge(
                graph, target, edge, event, should_preempt=should_preempt
            )

    async def control_edge(
        self,
        graph: SessionGraphRuntime,
        target: str,
        edge: ControlEdge,
        event: TimedChunk,
        *,
        should_preempt: bool = False,
    ) -> None:
        """An interrupt naming a turn first drops what the Session still holds of that turn on its way to the target and from it.

        The target's inbox and its node session's unsent appends lose the
        turn's chunks from other nodes, and a later chunk of the turn no
        longer enters the inbox; the target's pacer drops the turn's output
        and tells the target's readers how much of the turn's audio it had
        handed to the client. A preempting interrupt also names the turn's
        appends already sent, and every owner of the target aborts them. The
        client's own input stays.
        """
        if control_type(event) == INTERRUPT and event.turn is not None:
            graph.interrupted_turns.setdefault(target, set()).add(event.turn)
            inbox = graph.inboxes[target]
            inbox.queued = deque(
                item
                for item in inbox.queued
                if item[0] == GRAPH_INPUT or item[1].turn != event.turn
            )
            inbox.counts = Counter(source for source, _, _ in inbox.queued)
            inbox.wake.set()
            node = self.sessions[graph.nodes[target].id]
            kept = deque(
                (chunk, size)
                for chunk, size in node.pending
                if chunk.turn != event.turn
                or chunk.seq in graph.client_sequences[target]
            )
            node.pending_count -= len(node.pending) - len(kept)
            node.pending_bytes -= sum(size for _, size in node.pending) - sum(
                size for _, size in kept
            )
            node.pending = kept
        else:
            pass
        if control_type(event) == INTERRUPT and target in graph.pacers:
            graph.output_bytes -= graph.pacers[target].purge(event.turn)
            await self.report_delivered(graph, target, event.turn)
        elif control_type(event) is None and self.is_interrupted(graph, target, event):
            return
        else:
            pass
        should_preempt = edge.should_preempt or should_preempt
        if (
            should_preempt
            and control_type(event) == INTERRUPT
            and event.turn is not None
        ):
            node = self.sessions[graph.nodes[target].id]
            # Note (Junnan Li): An append of the turn still being submitted is named once it is out, so no owner meets it after the control without its name.
            while node.in_flight_append == "accepted" and not node.is_closing:
                node.wake.clear()
                await node.wake.wait()
            preempted_request_ids = tuple(
                request_id
                for request_id, chunk in node.submitted_appends.items()
                if chunk.turn == event.turn
                and chunk.seq not in graph.client_sequences[target]
            )
        else:
            preempted_request_ids = ()
        await self.control_session(
            graph.nodes[target],
            event,
            should_preempt=should_preempt,
            is_ordered=edge.source == GRAPH_INPUT,
            preempted_request_ids=preempted_request_ids,
        )

    async def report_delivered(
        self, graph: SessionGraphRuntime, name: str, turn: int | None
    ) -> None:
        """Send the readers of a paced node's controls the audio of an interrupted turn the pacer handed to the client."""
        if turn is None:
            pass
        else:
            delivered = TimedChunk(
                CONTROL,
                0,
                0,
                0,
                {
                    "type": REPLY_DELIVERED,
                    "audio_ms": graph.pacers[name].delivered_ms(turn),
                },
                format=SESSION_CONTROL,
                turn=turn,
            )
            for reader in graph.graph.reading_nodes(name, SESSION_CONTROL):
                await self.put_graph_inbox(graph, reader, name, delivered)

    async def open_graph(
        self,
        request: OmniRequest,
        *,
        graph: SessionGraph,
        limits: SessionLimits | None,
        session_id: str | None,
    ) -> SessionIdentity:
        """Open one node session per node, then route between them."""
        limits = limits or SessionLimits()
        session_id = session_id or str(uuid.uuid4())
        if session_id in self.sessions or session_id in self.graphs:
            raise ValueError("session ID already reserved")
        else:
            pass
        nodes = await self.open_graph_nodes(request, graph, limits, session_id)
        runtime = SessionGraphRuntime(
            session_identity=SessionIdentity(session_id, self.next_open_index),
            graph=graph,
            nodes=nodes,
            limits=limits,
            inboxes={name: NodeInbox() for name in nodes},
            outputs=asyncio.Queue(maxsize=limits.max_output_chunks),
            pacers={
                name: DeliveryPacer(self.max_output_lead_ms, asyncio.Event())
                for name, formats in graph.publish.items()
                if AUDIO_FORMATS & set(formats)
            },
            client_sequences={name: {} for name in nodes},
        )
        self.next_open_index += 1
        self.graphs[session_id] = runtime
        self.start_graph_tasks(runtime)
        return runtime.session_identity

    async def open_graph_nodes(
        self,
        request: OmniRequest,
        graph: SessionGraph,
        limits: SessionLimits,
        session_id: str,
    ) -> dict[str, SessionIdentity]:
        """Open the node sessions upstream first; a failed open closes the ones already open."""
        input_nodes = graph.input_nodes()
        nodes: dict[str, SessionIdentity] = {}
        try:
            for name in graph.order():
                node = graph.nodes[name]
                nodes[name] = await self.open_route(
                    request,
                    stages=list(node.stages),
                    # Note (Junnan Li): A node fed only by other nodes may wait a whole conversation; the input nodes bound the idle time.
                    limits=(
                        limits
                        if name in input_nodes
                        else replace(limits, idle_timeout_s=math.inf)
                    ),
                    session_id=f"{session_id}:{name}",
                    is_node=True,
                )
        except BaseException:
            for identity in reversed(list(nodes.values())):
                await self.close_session(identity)
            raise
        return nodes

    def start_graph_tasks(self, graph: SessionGraphRuntime) -> None:
        """Start, per node, the task feeding its inbox, the task routing its outputs and, with a pacer, the task delivering them."""
        for name in graph.nodes:
            graph.forwarders.append(
                self.graph_task(graph, name, self.forward_graph_inputs(graph, name))
            )
            graph.tasks.append(
                self.graph_task(graph, name, self.route_graph_outputs(graph, name))
            )
        for name in graph.pacers:
            graph.tasks.append(
                self.graph_task(graph, name, self.deliver_paced_outputs(graph, name))
            )

    def graph_task(
        self,
        graph: SessionGraphRuntime,
        name: str,
        coroutine: Coroutine[None, None, None],
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)

        def done(task: asyncio.Task[None]) -> None:
            if task.cancelled() or graph.is_closing:
                return
            else:
                error = task.exception()
                if error is None:
                    error = RuntimeError(f"graph node {name!r} session ended")
                else:
                    logger.warning(
                        f"graph {graph.session_identity.id} node {name} failed",
                        exc_info=error,
                    )
                graph.error = graph.error or error
                self.begin_graph_close(graph)

        task.add_done_callback(done)
        return task

    def append_graph(self, graph: SessionGraphRuntime, chunk: TimedChunk) -> int:
        """Hand client input to every node reading it; rejected input keeps its seq."""
        if graph.is_closing:
            raise RuntimeError("session is closing")
        elif graph.end_input is not None:
            raise ValueError("input after EOS")
        elif chunk.seq != graph.next_input:
            raise ValueError("input seq must be contiguous within an open index")
        else:
            check_timing(chunk, graph.modality_end_ms)
        chunk, _ = admit_input(
            chunk, graph.limits, graph.modality_end_ms, graph.ended_modalities
        )
        reading_nodes = graph.graph.reading_nodes(GRAPH_INPUT, chunk.format)
        full = [
            name
            for name in reading_nodes
            if graph.inboxes[name].counts[GRAPH_INPUT]
            >= self.edge_capacity(graph, name, GRAPH_INPUT)
        ]
        # Note (Junnan Li): Every input the append confirms, its own or a dropped one, needs an output slot before anything changes.
        dropped_inputs = Counter(
            next(
                seq
                for source, _, seq in graph.inboxes[name].queued
                if source == GRAPH_INPUT
            )
            for name in full
        )
        confirmations = sum(
            count == graph.pending_input_done[seq]
            for seq, count in dropped_inputs.items()
        ) + (0 if reading_nodes else 1)
        if any(
            graph.graph.nodes[name].bound(GRAPH_INPUT).overflow == "block"
            for name in full
        ) or (
            graph.outputs.maxsize > 0
            and graph.outputs.qsize() + confirmations > graph.outputs.maxsize
        ):
            raise QueueFullError()
        else:
            pass
        if reading_nodes:
            graph.pending_input_done[chunk.seq] = len(reading_nodes)
        else:
            self.emit_graph_input_done(graph, GRAPH_INPUT, chunk)
        for name in full:
            inbox = graph.inboxes[name]
            index = next(
                i for i, item in enumerate(inbox.queued) if item[0] == GRAPH_INPUT
            )
            _, dropped, _ = inbox.queued[index]
            del inbox.queued[index]
            inbox.counts[GRAPH_INPUT] -= 1
            self.confirm_graph_input(graph, name, dropped)
        for name in reading_nodes:
            inbox = graph.inboxes[name]
            inbox.queued.append((GRAPH_INPUT, chunk, chunk.seq))
            inbox.counts[GRAPH_INPUT] += 1
            inbox.wake.set()
        if chunk.eos:
            graph.end_input = chunk
            for pacer in graph.pacers.values():
                pacer.end_input()
        else:
            pass
        graph.modality_end_ms[chunk.modality] = chunk.t_start_ms + chunk.duration_ms
        if chunk.eos:
            graph.ended_modalities.add(chunk.modality)
        else:
            pass
        graph.next_input += 1
        return chunk.seq

    def edge_capacity(
        self, graph: SessionGraphRuntime, target: str, source: str
    ) -> int:
        capacity = graph.graph.nodes[target].bound(source).capacity
        if capacity is None:
            return graph.limits.max_pending_chunks
        else:
            return capacity

    def is_interrupted(
        self, graph: SessionGraphRuntime, target: str, chunk: TimedChunk
    ) -> bool:
        return chunk.turn is not None and chunk.turn in graph.interrupted_turns.get(
            target, ()
        )

    async def put_graph_inbox(
        self, graph: SessionGraphRuntime, target: str, source: str, chunk: TimedChunk
    ) -> None:
        """Queue a node's chunk on its edge into `target`; a full edge holds the producer or drops its own oldest chunk, and a chunk of any turn interrupted at `target` is dropped."""
        inbox = graph.inboxes[target]
        capacity = self.edge_capacity(graph, target, source)
        overflow = graph.graph.nodes[target].bound(source).overflow
        while (
            overflow == "block"
            and inbox.counts[source] >= capacity
            and not self.is_interrupted(graph, target, chunk)
        ):
            inbox.wake.clear()
            await inbox.wake.wait()
        if self.is_interrupted(graph, target, chunk):
            return
        elif overflow == "drop_oldest" and inbox.counts[source] >= capacity:
            del inbox.queued[
                next(i for i, item in enumerate(inbox.queued) if item[0] == source)
            ]
            inbox.counts[source] -= 1
        else:
            pass
        inbox.queued.append((source, chunk, None))
        inbox.counts[source] += 1
        inbox.wake.set()
        graph.eos_wake.set()

    async def forward_graph_inputs(self, graph: SessionGraphRuntime, name: str) -> None:
        """Append one node's inbox in that node's own seq order; a chunk leaves the inbox once its node session accepts it."""
        identity = graph.nodes[name]
        inbox = graph.inboxes[name]
        loop = asyncio.get_running_loop()
        seq = 0
        deadline: float | None = None
        while True:
            if not inbox.queued:
                inbox.wake.clear()
                await inbox.wake.wait()
                continue
            else:
                source, chunk, client_seq = inbox.queued[0]
            if chunk.eos and client_seq is not None:
                earlier = graph.graph.order()[: graph.graph.order().index(name)]
                waiting = (
                    set(earlier)
                    & set(graph.graph.reading_nodes(GRAPH_INPUT, chunk.format))
                    - graph.eos_confirmed
                )
                node = graph.graph.nodes[name]
                # Note (Junnan Li): A node with a feedback edge takes the client EOS after the open turns end, so it still reads their replies' feedback.
                if waiting or (
                    graph.open_turns
                    and any(node.bound(source).feedback for source in node.edges())
                ):
                    if len(inbox.queued) > 1:
                        inbox.queued.rotate(-1)
                    else:
                        graph.eos_wake.clear()
                        await graph.eos_wake.wait()
                    continue
                else:
                    pass
            else:
                pass
            try:
                await self.append_session(identity, replace(chunk, seq=seq))
            except QueueFullError:
                if deadline is None:
                    deadline = loop.time() + graph.limits.operation_timeout_s
                elif loop.time() >= deadline:
                    raise TimeoutError(
                        f"graph node {name!r} rejected input seq {seq} for "
                        f"{graph.limits.operation_timeout_s} s"
                    ) from None
                else:
                    pass
                await asyncio.sleep(FULL_QUEUE_RETRY_S)
            else:
                # Note (Junnan Li): The append completes without yielding, so the head is still the chunk it took.
                inbox.queued.popleft()
                inbox.counts[source] -= 1
                inbox.wake.set()
                if client_seq is not None:
                    graph.client_sequences[name][seq] = client_seq
                else:
                    pass
                seq += 1
                deadline = None

    async def route_graph_outputs(self, graph: SessionGraphRuntime, name: str) -> None:
        """Deliver one node's outputs to its readers and, when published, to the client.

        A node with a pacer hands it what it publishes, and its published
        turn.done reaches the readers only when the pacer releases it; a
        node without one gets its published turn.done back once published.
        """
        async for output in self.session_outputs(graph.nodes[name]):
            if output.kind == "input_done":
                client_seq = graph.client_sequences[name].pop(output.input_seq, None)
                if client_seq is None:
                    pass
                else:
                    self.confirm_graph_input(
                        graph,
                        name,
                        TimedChunk(
                            output.modality,
                            output.t_start_ms,
                            output.duration_ms,
                            client_seq,
                            None,
                            format=output.format,
                            eos=output.eos,
                        ),
                    )
                continue
            else:
                pass
            chunk = output_chunk(output)
            if any(
                is_interrupted_reply(chunk, turn)
                for turn in graph.interrupted_turns.get(name, ())
            ):
                continue
            else:
                pass
            is_published = self.is_client_output(graph, name, chunk)
            control_targets = graph.graph.control_targets(name, edge_event(chunk))
            for target, edge in control_targets:
                await self.control_edge(graph, target, edge, chunk)
            if control_targets and not (
                is_published and control_type(chunk) in REPORT_CONTROLS
            ):
                continue
            else:
                pass
            if is_published and graph.graph.shared_timeline:
                output = self.time_graph_output(graph, output)
                chunk = output_chunk(output)
            else:
                pass
            pacer = graph.pacers.get(name)
            if not holds_turn_end(chunk, pacer is not None, is_published):
                for reader in graph.graph.reading_nodes(name, output.format):
                    await self.put_graph_inbox(graph, reader, name, chunk)
            else:
                pass
            if is_published and pacer is not None:
                size = self.reserve_graph_output(graph, name, output)
                graph.output_bytes -= pacer.offer(output, size)
            elif is_published:
                await self.publish_graph_output(
                    graph, output, self.reserve_graph_output(graph, name, output)
                )
                self.return_turn_end(graph.nodes[name], output)
            else:
                pass

    def is_client_output(
        self, graph: SessionGraphRuntime, name: str, chunk: TimedChunk
    ) -> bool:
        """Whether a node output reaches the client: its format is published, it is no node control, and a report comes from a node no other node reads it from."""
        event = control_type(chunk)
        if not graph.graph.is_published(name, chunk.format) or event in NODE_CONTROLS:
            return False
        elif event in REPORT_CONTROLS:
            return not graph.graph.reading_nodes(name, chunk.format)
        else:
            return True

    def time_graph_output(
        self, graph: SessionGraphRuntime, output: OutputChunk
    ) -> OutputChunk:
        """Timestamp published reply chunks on the graph's shared media clock."""
        input_end_ms = graph.modality_end_ms.get(AUDIO, 0.0)
        event = control_type(output)
        if output.format in AUDIO_FORMATS:
            start = max(graph.output_end_ms, graph.reply_start_ms)
            graph.output_end_ms = start + output.duration_ms
        elif event == TURN_STARTED:
            start = max(graph.output_end_ms, input_end_ms)
            graph.reply_start_ms = start
        elif event == TURN_DONE:
            start = max(graph.output_end_ms, graph.reply_start_ms)
        elif event == TURN_INTERRUPTED:
            graph.output_end_ms = input_end_ms
            start = input_end_ms
        elif event in (SPEECH_STARTED, SPEECH_STOPPED):
            return output
        else:
            start = input_end_ms
        return replace(output, t_start_ms=start)

    async def deliver_paced_outputs(
        self, graph: SessionGraphRuntime, name: str
    ) -> None:
        """Publish what one node's pacer releases; a released turn.done reaches the readers and returns to the node as a control."""
        pacer = graph.pacers[name]
        loop = asyncio.get_running_loop()
        while True:
            pacer.unsent.extend(pacer.release(loop.time()))
            while pacer.unsent:
                # Note (Junnan Li): A released chunk waits in the pacer, where an interrupt still drops it, until the client's queue has room.
                if graph.outputs.full():
                    await asyncio.sleep(FULL_QUEUE_RETRY_S)
                    continue
                else:
                    pass
                output, size = pacer.unsent.popleft()
                if any(
                    is_interrupted_reply(output, turn)
                    for turn in graph.interrupted_turns.get(name, ())
                ):
                    graph.output_bytes -= size
                    continue
                else:
                    pass
                pacer.sent(output)
                await self.publish_graph_output(graph, output, size)
                if control_type(output) == TURN_DONE:
                    for reader in graph.graph.reading_nodes(name, output.format):
                        await self.put_graph_inbox(
                            graph, reader, name, output_chunk(output)
                        )
                else:
                    pass
                self.return_turn_end(graph.nodes[name], output)
            await pacer.wait()

    async def publish_graph_output(
        self, graph: SessionGraphRuntime, output: OutputChunk, size: int
    ) -> None:
        """Hand a node output to the client, counted as `size` bytes of the output budget."""
        event = control_type(output)
        if event == TURN_QUEUED:
            graph.open_turns.add(output.turn)
        elif event in (TURN_DONE, TURN_EMPTY, TURN_INTERRUPTED):
            graph.open_turns.discard(output.turn)
            graph.eos_wake.set()
        else:
            pass
        published = replace(
            output,
            session_identity=graph.session_identity,
            seq=graph.next_output,
            input_seq=max(graph.next_input - 1, 0),
        )
        graph.next_output += 1
        await graph.outputs.put((published, size))

    def reserve_graph_output(
        self, graph: SessionGraphRuntime, name: str, output: OutputChunk
    ) -> int:
        """Count an output against max_output_bytes until the client takes it; over the budget fails the graph."""
        size = wire_size(output.to_dict())
        if graph.output_bytes + size > graph.limits.max_output_bytes:
            raise RuntimeError(
                f"output of graph node {name!r} exceeds max_output_bytes "
                f"{graph.limits.max_output_bytes}; the client is not reading"
            )
        else:
            graph.output_bytes += size
        return size

    def confirm_graph_input(
        self, graph: SessionGraphRuntime, name: str, chunk: TimedChunk
    ) -> None:
        """One reader is done with the client's input `chunk`, by taking or by dropping it; the client hears of it once every reader is."""
        if chunk.eos:
            graph.eos_confirmed.add(name)
            graph.eos_wake.set()
        else:
            pass
        graph.pending_input_done[chunk.seq] -= 1
        if graph.pending_input_done[chunk.seq] == 0:
            del graph.pending_input_done[chunk.seq]
            if chunk.eos:
                graph.drain_task = asyncio.create_task(
                    self.drain_graph(graph, name, chunk)
                )
                graph.tasks.append(graph.drain_task)
            else:
                self.emit_graph_input_done(graph, name, chunk)
        else:
            pass

    async def drain_graph(
        self, graph: SessionGraphRuntime, name: str, chunk: TimedChunk
    ) -> None:
        """EOS confirms only after routed work and published reply turns finish."""
        deadline = asyncio.get_running_loop().time() + graph.limits.operation_timeout_s
        try:
            while (
                graph.open_turns
                or graph.pending_input_done
                or any(inbox.queued for inbox in graph.inboxes.values())
                or any(
                    self.sessions[identity.id].pending_count
                    or self.sessions[identity.id].outputs
                    for identity in graph.nodes.values()
                )
                or any(pacer.items for pacer in graph.pacers.values())
            ):
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("Session graph did not drain after EOS")
                else:
                    await asyncio.sleep(FULL_QUEUE_RETRY_S)
            self.emit_graph_input_done(graph, name, chunk)
        except Exception as error:
            graph.error = graph.error or error
            self.begin_graph_close(graph)

    def emit_graph_input_done(
        self, graph: SessionGraphRuntime, name: str, chunk: TimedChunk
    ) -> None:
        output = OutputChunk(
            session_identity=graph.session_identity,
            seq=graph.next_output,
            input_seq=chunk.seq,
            modality=chunk.modality,
            t_start_ms=chunk.t_start_ms,
            duration_ms=chunk.duration_ms,
            payload=None,
            format=chunk.format,
            eos=chunk.eos,
            kind="input_done",
        )
        if graph.outputs.full():
            raise RuntimeError(
                f"graph node {name!r} confirmed input seq {chunk.seq} while "
                "max_output_chunks outputs wait for the client"
            )
        else:
            graph.outputs.put_nowait(
                (output, self.reserve_graph_output(graph, name, output))
            )
        graph.next_output += 1

    async def graph_outputs(
        self, graph: SessionGraphRuntime
    ) -> AsyncIterator[OutputChunk]:
        if graph.has_output_consumer:
            raise RuntimeError("session already has an output consumer")
        else:
            graph.has_output_consumer = True
        try:
            while (item := await graph.outputs.get()) is not None:
                output, size = item
                graph.output_bytes -= size
                yield output
            if graph.error is not None:
                raise graph.error
            else:
                return
        finally:
            graph.has_output_consumer = False
            await asyncio.shield(self.close_graph(graph))

    def begin_graph_close(self, graph: SessionGraphRuntime) -> None:
        if graph.close_task is None:
            graph.close_task = self.owned_session_task(self.finish_graph_close(graph))
        else:
            pass

    async def close_graph(self, graph: SessionGraphRuntime) -> None:
        self.begin_graph_close(graph)
        assert graph.close_task is not None
        await asyncio.shield(graph.close_task)

    async def stop_graph_tasks(
        self, graph: SessionGraphRuntime, tasks: list[asyncio.Task[None]]
    ) -> None:
        """Cancel the tasks and wait for them; the first that failed instead of stopping fails the graph."""
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks)
        else:
            pass
        for task in tasks:
            if not task.cancelled() and task.exception() is not None:
                graph.error = graph.error or task.exception()
            else:
                pass

    async def finish_graph_close(self, graph: SessionGraphRuntime) -> None:
        """Stop input, close the nodes downstream first (`nodes` holds them upstream first), then stop routing."""
        graph.is_closing = True
        await self.stop_graph_tasks(graph, graph.forwarders)
        unclosed: dict[str, Exception] = {}
        for name, identity in reversed(list(graph.nodes.items())):
            node = self.sessions.get(identity.id)
            try:
                await self.close_session(identity)
            except Exception as exc:
                unclosed[name] = exc
            if node is not None and node.error is not None:
                graph.error = graph.error or node.error
            else:
                pass
        if unclosed:
            graph.cleanup_error = RuntimeError(
                f"graph nodes {sorted(unclosed)} did not close: {unclosed}"
            )
            graph.cleanup_error.__cause__ = next(iter(unclosed.values()))
            graph.error = graph.error or next(iter(unclosed.values()))
        else:
            pass
        await self.stop_graph_tasks(graph, graph.tasks)
        # Note (Junnan Li): Close fences output like a route close; queued data is dropped.
        while not graph.outputs.empty():
            graph.outputs.get_nowait()
        graph.output_bytes = 0
        graph.outputs.put_nowait(None)
        self.graphs.pop(graph.session_identity.id, None)
