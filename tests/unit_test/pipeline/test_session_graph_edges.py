# SPDX-License-Identifier: Apache-2.0
"""Session graph edges over spawned stage owners: bounds, order, interruption fencing and control delivery."""

from __future__ import annotations

import asyncio
import multiprocessing
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.config.session_audio import REALTIME_MAX_PENDING_CHUNKS
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    ControlEdge,
    DataEdge,
    OutputChunk,
    SessionGraph,
    SessionIdentity,
    SessionLimits,
    SessionNode,
    TimedChunk,
)
from sglang_omni.proto.streams import (
    INTERRUPT,
    SESSION_CONTROL,
    TOOL,
    TOOL_CALL,
    TOOL_RESPONSE,
    TURN_INTERRUPTED,
    control_type,
)
from sglang_omni.scheduling.session import (
    DEFAULT_MAX_OPEN_SESSIONS,
    ChunkEmitter,
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from tests.unit_test.fixtures.session_outputs import control_chunk
from tests.unit_test.fixtures.session_pipeline import (
    StageEvent,
    block_session_cleanup,
    event_log,
    pipeline,
    wait_until,
)

OUTPUT_TIMEOUT_S = 5
APPEND_RETRY_S = 0.01
GATE_POLL_S = 0.01
# Every edge holds one chunk and every node session one pending append, so a held node backs its edges up at once.
LIMITS = SessionLimits(max_pending_chunks=1)
SPEECH_STARTED = {"text": "speech_started"}
SPEECH_STOPPED = {"text": "speech_stopped"}
# A chunk crosses an inbox, a node session, a stage process and its pump; this outlasts those hops.
SETTLE_S = 0.3
FACTORY = f"{__name__}.create_edge_stage"


@dataclass
class EdgeState:
    fail: bool
    interrupted: set[int | None] = field(default_factory=set)


class EdgeHooks(SessionHooks):
    """Answers each text input with name(text) once the node's push gate is open; an interrupt silences the turn it names."""

    def __init__(self, name: str, events: Queue[StageEvent], gates: dict[str, Event]):
        self.name = name
        self.events = events
        self.push_gate = gates[f"{name}.push"]
        self.control_gate = gates[f"{name}.control"]
        self.states: dict[SessionIdentity, EdgeState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.events.put(("open", self.name, session_identity.id))
        self.states[session_identity] = EdgeState(
            fail=request.params.get("fail") == self.name
        )

    def close(self, session_identity: SessionIdentity) -> None:
        self.events.put(("close", self.name, session_identity.id))
        self.states.pop(session_identity)

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        state = self.states[context.session_identity]
        text = chunk.payload["text"]
        self.events.put(
            ("append", self.name, context.session_identity.id, f"{chunk.format}:{text}")
        )
        while not self.push_gate.wait(GATE_POLL_S):
            if context.cancelled.is_set():
                return payload
            else:
                pass
        if state.fail:
            raise RuntimeError(f"{self.name} failed")
        elif chunk.turn in state.interrupted or context.cancelled.is_set():
            pass
        else:
            context.emit(
                TimedChunk(
                    chunk.modality,
                    chunk.t_start_ms,
                    chunk.duration_ms,
                    chunk.seq,
                    {"text": f"{self.name}({text})"},
                    format=chunk.format,
                    turn=chunk.turn,
                )
            )
        return payload

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        self.events.put(
            (
                "control",
                self.name,
                session_identity.id,
                control_type(event) or event.format,
            )
        )
        self.control_gate.wait()
        if control_type(event) == INTERRUPT:
            self.states[session_identity].interrupted.add(event.turn)
        else:
            pass


class FrontHooks(SessionHooks):
    """Emits an interrupt and then turn.interrupted of the chunk's turn when it hears SPEECH_STARTED, and hands the turn's audio on as one text at SPEECH_STOPPED, as a dialog node does."""

    def __init__(self) -> None:
        self.emitters: dict[SessionIdentity, ChunkEmitter] = {}
        self.heard: list[str] = []

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        pass

    def attach_output_stream(
        self, session_identity: SessionIdentity, emit: ChunkEmitter
    ) -> None:
        self.emitters[session_identity] = emit

    def close(self, session_identity: SessionIdentity) -> None:
        self.emitters.pop(session_identity)

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if chunk.payload == SPEECH_STARTED:
            for event in (INTERRUPT, TURN_INTERRUPTED):
                context.emit(
                    control_chunk(
                        event, chunk.t_start_ms, seq=chunk.seq, turn=chunk.turn
                    )
                )
        elif chunk.payload == SPEECH_STOPPED:
            text = " ".join(self.heard)
            self.heard.clear()
            context.emit(
                replace(chunk, modality="text", payload={"text": text}, format="text")
            )
        else:
            self.heard.append(chunk.payload["text"])
        return payload


class ToolHooks(SessionHooks):
    """Answers each tool call with an empty tool_response of the call's turn."""

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        pass

    def close(self, session_identity: SessionIdentity) -> None:
        pass

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        context.emit(
            TimedChunk(
                TOOL,
                chunk.t_start_ms,
                0,
                chunk.seq,
                {"responses": []},
                TOOL_RESPONSE,
                turn=chunk.turn,
            )
        )
        return payload


def create_edge_stage(
    name: str,
    events: Queue[StageEvent],
    gates: dict[str, Event],
    max_open_sessions: int,
) -> SessionScheduler:
    if name == "front":
        hooks = FrontHooks()
    elif name == "tool":
        hooks = ToolHooks()
    else:
        hooks = EdgeHooks(name, events, gates)
    return SessionScheduler(hooks, max_open_sessions=max_open_sessions)


def text_chunk(text: str, seq: int, turn: int | None = None) -> TimedChunk:
    return TimedChunk("text", seq * 20, 20, seq, {"text": text}, "text", turn=turn)


def speech_chunk(seq: int, turn: int, payload: dict = SPEECH_STARTED) -> TimedChunk:
    return TimedChunk("audio", seq * 20, 20, seq, payload, "pcm16", turn=turn)


async def settle() -> None:
    """Let every hop run until it blocks."""
    await asyncio.sleep(SETTLE_S)


def cascade_graph(order: tuple[str, ...] = ("front", "llm", "tts")) -> SessionGraph:
    interrupt = ControlEdge("front", INTERRUPT, should_preempt=True)
    nodes = {
        "front": SessionNode(("front",), ("input:pcm16",)),
        "llm": SessionNode(("llm",), ("input:text",), controls=(interrupt,)),
        "tts": SessionNode(("tts",), ("llm:text",), controls=(interrupt,)),
    }
    return SessionGraph({name: nodes[name] for name in order}, {"tts": ("text",)})


def dialog_graph() -> SessionGraph:
    """A dialog-like front that publishes its turn controls and feeds the llm, which takes its interrupt as a preempting control."""
    interrupt = ControlEdge("front", INTERRUPT, should_preempt=True)
    return SessionGraph(
        {
            "front": SessionNode(("front",), ("input:pcm16",)),
            "llm": SessionNode(("llm",), ("front:text",), controls=(interrupt,)),
        },
        {"front": (SESSION_CONTROL,), "llm": ("text",)},
    )


def edge_graph(edge: DataEdge) -> SessionGraph:
    return SessionGraph(
        {
            "src": SessionNode(("src",), ("input:text",)),
            "sink": SessionNode(("sink",), (edge,)),
        },
        {"sink": ("text",)},
    )


class Gates:
    """One push and one control gate per node, open unless a test holds them."""

    def __init__(self, names: tuple[str, ...]) -> None:
        context = multiprocessing.get_context("spawn")
        self.events: dict[str, Event] = {}
        for name in names:
            for kind in ("push", "control"):
                gate = context.Event()
                gate.set()
                self.events[f"{name}.{kind}"] = gate

    def hold(self, name: str, kind: str = "push") -> None:
        self.events[f"{name}.{kind}"].clear()

    def release(self, name: str, kind: str = "push") -> None:
        self.events[f"{name}.{kind}"].set()


class Log:
    """The stage events seen so far, read from the workers' queue on demand."""

    def __init__(self, events: Queue[StageEvent]) -> None:
        self.events = events
        self.seen: list[StageEvent] = []

    def read(self) -> list[StageEvent]:
        self.seen.extend(event_log(self.events))
        return self.seen

    def of(self, kind: str, name: str) -> list[str]:
        return [event[-1] for event in self.read() if event[:2] == (kind, name)]

    def names(self, kind: str) -> list[str]:
        return [event[1] for event in self.read() if event[0] == kind]


@asynccontextmanager
async def open_graph(
    tmp_path: Path,
    graph: SessionGraph,
    names: tuple[str, ...],
    params: dict[str, object] | None = None,
    limits: SessionLimits = LIMITS,
    max_open_sessions: int = DEFAULT_MAX_OPEN_SESSIONS,
) -> AsyncIterator[tuple[Coordinator, SessionIdentity, Gates, Log]]:
    gates = Gates(names)
    async with pipeline(
        tmp_path,
        names=list(names),
        factories={name: FACTORY for name in names},
        factory_kwargs={"gates": gates.events, "max_open_sessions": max_open_sessions},
    ) as (coordinator, events, _):
        coordinator.session_graph = graph
        identity = await coordinator.open_session(
            OmniRequest(None, params=params or {}), limits=limits
        )
        log = Log(events)
        try:
            yield coordinator, identity, gates, log
        finally:
            for name in names:
                gates.release(name)
                gates.release(name, "control")
            if identity.id in coordinator.graphs:
                await coordinator.close_session(identity)
            else:
                pass
            await wait_until(
                lambda: not coordinator.sessions and not coordinator.graphs
            )


async def append(
    coordinator: Coordinator, identity: SessionIdentity, chunk: TimedChunk
) -> None:
    """Append as a client does: a full edge rejects the append, and the client retries."""
    while True:
        try:
            await coordinator.append_session(identity, chunk)
            return
        except QueueFullError:
            await asyncio.sleep(APPEND_RETRY_S)


async def next_data(outputs: AsyncIterator[OutputChunk]) -> OutputChunk:
    while (
        output := await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)
    ).kind != "data":
        pass
    return output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_pending_chunks, unsent_appends",
    [(1, 0), (REALTIME_MAX_PENDING_CHUNKS, 2)],
    ids=["no_unsent_appends", "unsent_appends"],
)
async def test_interrupt_drops_queued_stale_output_but_not_the_next_turn(
    tmp_path, max_pending_chunks, unsent_appends
):
    """Declared downstream first, the nodes still open upstream first, take the interrupt upstream first and close downstream first.

    With the realtime pending limit, the interrupt meets appends of the turn that the target's node session has not sent yet, and drops them. The client's EOS still drains the graph afterwards.
    """
    graph = cascade_graph(order=("tts", "llm", "front"))
    async with open_graph(
        tmp_path,
        graph,
        ("front", "llm", "tts"),
        limits=SessionLimits(max_pending_chunks=max_pending_chunks),
    ) as (coordinator, identity, gates, log):
        assert log.names("open") == ["llm", "tts"]
        outputs = coordinator.session_outputs(identity)
        gates.hold("tts")
        for seq in range(3):
            await append(coordinator, identity, text_chunk(f"old{seq}", seq, turn=1))
        await wait_until(lambda: len(log.of("append", "llm")) == 3)
        await settle()
        assert log.of("append", "tts") == ["text:llm(old0)"]
        tts_node = coordinator.sessions[coordinator.graphs[identity.id].nodes["tts"].id]
        await wait_until(lambda: len(tts_node.pending) == unsent_appends)
        await append(coordinator, identity, speech_chunk(3, turn=1))
        await wait_until(lambda: log.names("control") == ["llm", "tts"])
        gates.release("tts")
        await append(coordinator, identity, text_chunk("new", 4, turn=2))
        output = await next_data(outputs)
        assert output.payload == {"text": "tts(llm(new))"}
        assert log.of("append", "tts") == ["text:llm(old0)", "text:llm(new)"]
        await append(
            coordinator, identity, replace(text_chunk("end", 5, turn=2), eos=True)
        )
        while (
            output := await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)
        ).kind != "input_done" or output.input_seq != 5:
            pass
        await outputs.aclose()
        await coordinator.close_session(identity)
        await wait_until(lambda: log.names("close") == ["tts", "llm"])


@pytest.mark.asyncio
async def test_input_arriving_during_interrupt_waits_and_is_not_fenced(tmp_path):
    async with open_graph(tmp_path, cascade_graph(), ("front", "llm", "tts")) as (
        coordinator,
        identity,
        gates,
        log,
    ):
        outputs = coordinator.session_outputs(identity)
        gates.hold("llm", "control")
        await append(coordinator, identity, speech_chunk(0, turn=1))
        await wait_until(lambda: "llm" in log.names("control"))
        await append(coordinator, identity, text_chunk("new", 1, turn=2))
        await settle()
        assert log.of("append", "llm") == []
        gates.release("llm", "control")
        output = await next_data(outputs)
        assert output.payload == {"text": "tts(llm(new))"}
        await outputs.aclose()


@pytest.mark.asyncio
async def test_an_owner_slow_to_take_an_interrupt_holds_neither_the_input_acks_nor_the_stop(
    tmp_path,
):
    """While an owner of the interrupted node has not yet taken the preempting interrupt, the client still hears turn.interrupted and has its input confirmed, and the node's next input waits for the interrupt."""
    async with open_graph(tmp_path, dialog_graph(), ("front", "llm")) as (
        coordinator,
        identity,
        gates,
        log,
    ):
        outputs = coordinator.session_outputs(identity)
        stopped_turns: list[int | None] = []
        confirmed_inputs: list[int] = []

        async def hear_the_stop_and_the_confirmations() -> None:
            while not stopped_turns or confirmed_inputs != [0, 1]:
                output = await anext(outputs)
                if output.kind == "input_done":
                    confirmed_inputs.append(output.input_seq)
                elif control_type(output) == TURN_INTERRUPTED:
                    stopped_turns.append(output.turn)
                else:
                    pass

        gates.hold("llm", "control")
        try:
            await append(coordinator, identity, speech_chunk(0, turn=1))
            await wait_until(lambda: "llm" in log.names("control"))
            await append(
                coordinator, identity, speech_chunk(1, turn=2, payload={"text": "new"})
            )
            hearing = asyncio.create_task(hear_the_stop_and_the_confirmations())
            # A timed-out read of the outputs would close the session behind the held owner, so this wait leaves the read running.
            heard_while_held, _ = await asyncio.wait(
                {hearing}, timeout=OUTPUT_TIMEOUT_S
            )
            await append(
                coordinator, identity, speech_chunk(2, turn=2, payload=SPEECH_STOPPED)
            )
            await settle()
            appended_before_the_interrupt = log.of("append", "llm")
        finally:
            gates.release("llm", "control")
        await asyncio.wait_for(hearing, OUTPUT_TIMEOUT_S)
        assert heard_while_held == {hearing}
        assert stopped_turns == [1]
        assert appended_before_the_interrupt == []
        while (
            output := await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)
        ).format != "text":
            pass
        assert output.payload == {"text": "llm(new)"}
        assert log.of("append", "llm") == ["text:new"]
        await outputs.aclose()


@pytest.mark.asyncio
async def test_drop_oldest_edge_keeps_newest_without_blocking_producer(tmp_path):
    graph = edge_graph(DataEdge("src:text", capacity=2, overflow="drop_oldest"))
    async with open_graph(tmp_path, graph, ("src", "sink")) as (
        coordinator,
        identity,
        gates,
        log,
    ):
        outputs = coordinator.session_outputs(identity)
        gates.hold("sink")
        await append(coordinator, identity, text_chunk("0", 0))
        await wait_until(lambda: log.of("append", "sink") == ["text:src(0)"])
        for seq in range(1, 5):
            await append(coordinator, identity, text_chunk(str(seq), seq))
        receipts = [
            (await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)).input_seq
            for _ in range(5)
        ]
        assert receipts == [0, 1, 2, 3, 4]
        gates.release("sink")
        await wait_until(lambda: len(log.of("append", "sink")) == 3)
        await settle()
        assert log.of("append", "sink") == ["text:src(0)", "text:src(3)", "text:src(4)"]
        await outputs.aclose()


@pytest.mark.asyncio
async def test_a_drop_oldest_client_edge_keeps_the_newest_input_and_never_rejects(
    tmp_path,
):
    graph = SessionGraph(
        {
            "rt": SessionNode(
                ("rt",), (DataEdge("input", capacity=2, overflow="drop_oldest"),)
            )
        },
        {"rt": ("text",)},
    )
    async with open_graph(tmp_path, graph, ("rt",)) as (
        coordinator,
        identity,
        gates,
        log,
    ):
        outputs = coordinator.session_outputs(identity)
        gates.hold("rt")
        await coordinator.append_session(identity, text_chunk("0", 0))
        await wait_until(lambda: log.of("append", "rt") == ["text:0"])
        for seq in range(1, 5):
            await coordinator.append_session(identity, text_chunk(str(seq), seq))
        dropped = [
            (await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)).input_seq
            for _ in range(2)
        ]
        assert dropped == [1, 2]
        gates.release("rt")
        received = [
            await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S) for _ in range(6)
        ]
        assert log.of("append", "rt") == ["text:0", "text:3", "text:4"]
        assert sorted(o.input_seq for o in received if o.kind == "input_done") == [
            0,
            3,
            4,
        ]
        await outputs.aclose()


@pytest.mark.asyncio
async def test_a_tool_response_on_a_control_edge_passes_the_queued_audio(tmp_path):
    graph = SessionGraph(
        {
            "chat": SessionNode(
                ("chat",),
                (DataEdge("input:pcm16", capacity=2),),
                controls=(ControlEdge("tool", TOOL_RESPONSE),),
            ),
            "tool": SessionNode(("tool",), (f"input:{TOOL_CALL}",)),
        },
        {"chat": ("pcm16",)},
    )
    async with open_graph(tmp_path, graph, ("chat", "tool")) as (
        coordinator,
        identity,
        gates,
        log,
    ):
        gates.hold("chat")

        def frame(seq: int) -> TimedChunk:
            return TimedChunk("audio", seq * 20, 20, seq, {"text": f"f{seq}"}, "pcm16")

        await append(coordinator, identity, frame(0))
        await wait_until(lambda: log.of("append", "chat") == ["pcm16:f0"])
        for seq in (1, 2):
            await coordinator.append_session(identity, frame(seq))
        with pytest.raises(QueueFullError):
            await coordinator.append_session(identity, frame(3))
        call = TimedChunk("tool", 60, 0, 3, {"calls": []}, TOOL_CALL, turn=1)
        await coordinator.append_session(identity, call)
        await settle()
        assert log.of("append", "chat") == ["pcm16:f0"]
        gates.release("chat")
        await wait_until(lambda: len(log.of("append", "chat")) == 3)
        assert [
            event[2:]
            for event in log.read()
            if event[1] == "chat" and event[0] != "open"
        ] == [
            (identity.id + ":chat", "pcm16:f0"),
            (identity.id + ":chat", TOOL_RESPONSE),
            (identity.id + ":chat", "pcm16:f1"),
            (identity.id + ":chat", "pcm16:f2"),
        ]


@pytest.mark.asyncio
async def test_blocking_edge_holds_producer_until_consumer_drains(tmp_path):
    graph = edge_graph(DataEdge("src:text", capacity=1))
    async with open_graph(tmp_path, graph, ("src", "sink")) as (
        coordinator,
        identity,
        gates,
        log,
    ):
        outputs = coordinator.session_outputs(identity)
        gates.hold("sink")
        await append(coordinator, identity, text_chunk("0", 0))
        await wait_until(lambda: log.of("append", "sink") == ["text:src(0)"])
        for seq in (1, 2):
            await append(coordinator, identity, text_chunk(str(seq), seq))
        receipts = [
            (await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)).input_seq
            for _ in range(2)
        ]
        await settle()
        assert receipts == [0, 1]
        assert coordinator.graphs[identity.id].inboxes["sink"].counts["src"] == 1
        gates.release("sink")
        await wait_until(lambda: len(log.of("append", "sink")) == 3)
        assert log.of("append", "sink") == ["text:src(0)", "text:src(1)", "text:src(2)"]
        await outputs.aclose()


@pytest.mark.asyncio
async def test_node_failure_fails_outputs_and_closes_the_rest(tmp_path):
    graph = cascade_graph(order=("tts", "llm", "front"))
    async with open_graph(
        tmp_path, graph, ("front", "llm", "tts"), params={"fail": "llm"}
    ) as (coordinator, identity, _, log):
        outputs = coordinator.session_outputs(identity)
        assert log.names("open") == ["llm", "tts"]
        await append(coordinator, identity, text_chunk("boom", 0, turn=1))
        # The failing append and the read the stage closes race to report it; the client sees one of them.
        with pytest.raises(
            RuntimeError, match="llm failed|session owner 'llm' closed the session"
        ):
            await next_data(outputs)
        await wait_until(lambda: not coordinator.graphs)
        assert sorted(log.names("close")) == ["llm", "tts"]
        with pytest.raises(ValueError, match="unknown or stale session"):
            await coordinator.append_session(identity, text_chunk("late", 1, turn=1))


@pytest.mark.asyncio
async def test_an_open_waits_for_a_closing_session_to_free_its_stage(
    tmp_path, monkeypatch
):
    """A client that closed its session and opened the next one at once was refused with QueueFullError: the close had not reached the full stage yet.

    The open now waits for that close. With no session closing, it still fails with QueueFullError at once.
    """
    graph = SessionGraph(
        {"rt": SessionNode(("rt",), ("input:text",))}, {"rt": ("text",)}
    )
    async with open_graph(tmp_path, graph, ("rt",), max_open_sessions=1) as (
        coordinator,
        first,
        _,
        _,
    ):
        with pytest.raises(QueueFullError):
            await asyncio.wait_for(
                coordinator.open_session(OmniRequest(None), limits=LIMITS),
                OUTPUT_TIMEOUT_S,
            )
        entered, release = block_session_cleanup(monkeypatch, coordinator)
        closing = asyncio.create_task(coordinator.close_session(first))
        await entered.wait()
        opening = asyncio.create_task(
            coordinator.open_session(
                OmniRequest(None), limits=LIMITS, session_id="next"
            )
        )
        await wait_until(
            lambda: "next:rt" in coordinator.sessions
            and coordinator.sessions["next:rt"].is_closing
        )
        release.set()
        await closing
        second = await asyncio.wait_for(opening, OUTPUT_TIMEOUT_S)
        await coordinator.close_session(second)
