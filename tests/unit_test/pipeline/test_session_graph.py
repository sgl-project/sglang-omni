# SPDX-License-Identifier: Apache-2.0
"""Session graphs: node routes joined by format-typed edges inside the coordinator."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from multiprocessing.queues import Queue

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.config.schema import SessionControlEdgeConfig
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.session_delivery import DeliveryPacer
from sglang_omni.pipeline.session_graph import SessionGraphRunner
from sglang_omni.pipeline.sessions import NodeInbox, Session, SessionGraphRuntime
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
    wire_size,
)
from sglang_omni.proto.streams import (
    AUDIO,
    CONTROL,
    IMAGE,
    IMAGE_FRAME,
    INTERRUPT,
    PCM16,
    REPLY_DELIVERED,
)
from sglang_omni.scheduling.session import (
    ChunkEmitter,
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from tests.unit_test.fixtures.session_outputs import control_chunk
from tests.unit_test.fixtures.session_pipeline import (
    VAD_FACTORY,
    StageEvent,
    chunk,
    event_log,
    pipeline,
    wait_until,
)

NODES = ["left", "right"]
OUTPUT_TIMEOUT_S = 5
# The client's unit in the hop test, and how long after its append the upstream node emits.
HOP_UNIT_S = 1.0
HOP_EMIT_AFTER_S = 0.1


def chain_graph() -> SessionGraph:
    return SessionGraph(
        nodes={
            "left": SessionNode(stages=("left",), inputs=("input",)),
            "right": SessionNode(stages=("right",), inputs=("left:text",)),
        },
        publish={"right": ("text",)},
    )


async def collect(outputs, count: int) -> list[OutputChunk]:
    return [
        await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S) for _ in range(count)
    ]


def test_graph_declaration_names_its_sources_and_readers():
    graph = chain_graph()
    assert graph.input_nodes() == ("left",)
    assert graph.reading_nodes("left", "text") == ("right",)
    assert graph.reading_nodes("left", "audio") == ()
    assert graph.reading_nodes("input", None) == ("left",)
    assert graph.is_published("right", "text") and not graph.is_published(
        "left", "text"
    )
    graph = control_graph()
    assert graph.control_targets("left", "interrupt") == (
        ("right", ControlEdge("left", "interrupt", should_preempt=True)),
    )
    assert graph.control_targets("input", "interrupt") == (
        ("right", ControlEdge("input", "interrupt")),
    )
    assert graph.control_targets("left", "turn.done") == ()
    assert graph.reading_nodes("left", "session") == ()


def voicechat_tool(
    *responses: str | DataEdge, controls: tuple[ControlEdge, ...] = ()
) -> dict[str, SessionNode]:
    """A chat node and the tool node it calls, answered on `responses` or `controls`."""
    return {
        "voicechat": SessionNode(
            ("voicechat",), ("input", *responses), controls=controls
        ),
        "tool": SessionNode(("tool",), ("voicechat:tool_call",)),
    }


def after_a(
    *inputs: str | DataEdge, controls: tuple[ControlEdge, ...] = ()
) -> dict[str, SessionNode]:
    return {
        "a": SessionNode(("a",), ("input",)),
        "b": SessionNode(("b",), inputs, controls=controls),
    }


def test_a_marked_feedback_edge_or_a_control_edge_closes_a_cycle_without_ordering_it():
    feedback = DataEdge("tool:tool_response", overflow="drop_oldest", feedback=True)
    assert SessionGraph(voicechat_tool(feedback), {}).order() == ("voicechat", "tool")
    controlled = voicechat_tool(controls=(ControlEdge("tool", "tool_response"),))
    assert SessionGraph(controlled, {}).order() == ("voicechat", "tool")


@pytest.mark.parametrize(
    ("nodes", "publish", "message"),
    [
        ({"a": SessionNode(("x",), ("b",))}, {}, "unknown inputs"),
        (
            {"a": SessionNode(("x",), ("b",)), "b": SessionNode(("y",), ("a",))},
            {},
            "no node reads",
        ),
        ({"a": SessionNode(("x",), ("input",))}, {"b": ("text",)}, "publish"),
        (after_a(controls=(ControlEdge("c", "interrupt"),)), {}, "unknown inputs"),
        (
            after_a(
                "input", DataEdge("a:text", capacity=2), DataEdge("a:audio", capacity=3)
            ),
            {},
            "bounds its edge from 'a' twice",
        ),
        (after_a("input", DataEdge("a", capacity=0)), {}, "capacity of at least 1"),
        (voicechat_tool("tool:tool_response"), {}, "cycle"),
        (
            voicechat_tool(DataEdge("tool:tool_response", feedback=True)),
            {},
            "must not block",
        ),
    ],
)
def test_graph_rejects_a_miswired_declaration(nodes, publish, message):
    with pytest.raises(ValueError, match=message):
        SessionGraph(nodes, publish)


def control_graph() -> SessionGraph:
    return SessionGraph(
        nodes={
            "left": SessionNode(stages=("left",), inputs=("input",)),
            "right": SessionNode(
                stages=("right",),
                inputs=(),
                controls=(
                    ControlEdge("left", "interrupt", should_preempt=True),
                    ControlEdge("input", "interrupt"),
                ),
            ),
        },
        publish={"left": ("session",)},
    )


@pytest.mark.asyncio
async def test_a_control_edge_delivers_the_event_as_a_control_operation(tmp_path):
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = control_graph()
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_control": "interrupt"})
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        received = await collect(outputs, 1)
        assert [item.kind for item in received] == ["input_done"]
        await wait_until(
            lambda: ("control", "right", f"{identity.id}:right", "interrupt")
            in event_log(events)
        )
        event = TimedChunk("control", 0, 0, 0, {"type": "interrupt"}, "session")
        await coordinator.control_session(identity, event)
        await wait_until(
            lambda: ("control", "right", f"{identity.id}:right", "interrupt")
            in event_log(events)
        )
        with pytest.raises(ValueError, match="control edges"):
            await coordinator.control_session(identity, event, stages=["right"])
        with pytest.raises(ValueError, match="no node takes"):
            await coordinator.control_session(
                identity, TimedChunk("control", 0, 0, 0, {"type": "nudge"}, "session")
            )
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


def test_a_pipeline_declares_its_session_as_one_route_or_one_graph():
    with pytest.raises(ValueError, match="session_route or session_graph"):
        Coordinator(
            "ipc://done",
            "ipc://abort",
            "left",
            session_route=lambda: NODES,
            session_graph=chain_graph(),
        )


@pytest.mark.asyncio
async def test_edge_chunks_reach_their_reader_and_only_published_data_reaches_the_client(
    tmp_path,
):
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = chain_graph()
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_format": "text"})
        )
        outputs = coordinator.session_outputs(identity)
        for seq in range(3):
            await coordinator.append_session(identity, chunk(seq))
        received = await collect(outputs, 6)
        receipts = [item for item in received if item.kind == "input_done"]
        data = [item for item in received if item.kind == "data"]
        assert [item.input_seq for item in receipts] == [0, 1, 2]
        assert [item.seq for item in received] == list(range(6))
        assert all(item.session_identity == identity for item in received)
        assert [item.payload["stage"] for item in data] == ["right"] * 3
        assert [item.payload["input"]["count"] for item in data] == [1, 2, 3]
        assert [item.payload["input"]["stage"] for item in data] == ["left"] * 3
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)
        closes = [event[1] for event in event_log(events) if event[0] == "close"]
        assert closes == ["right", "left"]


def test_graph_outputs_count_against_max_output_bytes_until_the_client_takes_them():
    output = OutputChunk(
        SessionIdentity("graph", 1), 0, 0, "text", 0, 0, bytes(400), "text"
    )
    size = wire_size(output.to_dict())
    graph = SessionGraphRuntime(
        session_identity=SessionIdentity("graph", 1),
        graph=chain_graph(),
        nodes={},
        limits=SessionLimits(max_output_bytes=size * 2 - 1),
        inboxes={},
        outputs=asyncio.Queue(),
    )
    sessions = SessionGraphRunner()
    assert sessions.reserve_graph_output(graph, "right", output) == size
    assert graph.output_bytes == size
    with pytest.raises(RuntimeError, match="node 'right' exceeds max_output_bytes"):
        sessions.reserve_graph_output(graph, "right", output)


@pytest.mark.asyncio
async def test_input_fans_out_to_every_reader_and_input_done_waits_for_all(tmp_path):
    graph = SessionGraph(
        nodes={name: SessionNode(stages=(name,), inputs=("input",)) for name in NODES},
        publish={name: ("text",) for name in NODES},
    )
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = graph
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_format": "text"})
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        received = await collect(outputs, 3)
        assert sorted(item.kind for item in received) == ["data", "data", "input_done"]
        assert {item.payload["stage"] for item in received if item.kind == "data"} == {
            "left",
            "right",
        }
        appends = [event for event in event_log(events) if event[0] == "append"]
        assert sorted(event[1] for event in appends) == NODES
        with pytest.raises(ValueError, match="contiguous"):
            await coordinator.append_session(identity, chunk(5))
        await coordinator.close_session(identity)
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)
        with pytest.raises(StopAsyncIteration):
            await anext(outputs)


@pytest.mark.asyncio
async def test_camera_frames_reach_only_the_nodes_that_read_them(tmp_path):
    names = ["vad", "seeing", "cascade"]
    graph = SessionGraph(
        nodes={
            "vad": SessionNode(
                stages=("vad",), inputs=("input:pcm16", "input:session")
            ),
            "seeing": SessionNode(
                stages=("seeing",), inputs=("input:pcm16", "input:image_frame")
            ),
            "cascade": SessionNode(stages=("cascade",), inputs=("input:pcm16",)),
        },
        publish={},
    )
    assert graph.reading_nodes("input", IMAGE_FRAME) == ("seeing",)
    async with pipeline(tmp_path, names=names) as (coordinator, events, _):
        coordinator.session_graph = graph
        identity = await coordinator.open_session(OmniRequest(None, {}))
        outputs = coordinator.session_outputs(identity)
        frame = TimedChunk(IMAGE, 0, 0, 0, b"\xff\xd8frame", format=IMAGE_FRAME)
        audio = TimedChunk(AUDIO, 0, 20, 1, b"pcm", format=PCM16)
        await coordinator.append_session(identity, frame)
        await coordinator.append_session(identity, audio)
        received = await collect(outputs, 2)
        assert [(item.kind, item.input_seq) for item in received] == [
            ("input_done", 0),
            ("input_done", 1),
        ]
        appends = [event for event in event_log(events) if event[0] == "append"]
        assert sorted(event[1] for event in appends) == [
            "cascade",
            "seeing",
            "seeing",
            "vad",
        ]
        await coordinator.close_session(identity)


@pytest.mark.asyncio
async def test_failed_node_open_closes_the_nodes_already_open(tmp_path):
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = chain_graph()
        with pytest.raises(RuntimeError, match="open failed"):
            await coordinator.open_session(OmniRequest(None, {"fail_open": "right"}))
        await wait_until(lambda: not coordinator.sessions)
        assert not coordinator.graphs
        log = event_log(events)
        opened = [event[1] for event in log if event[0] == "open"]
        closed = [event[1] for event in log if event[0] == "close"]
        assert "left" not in opened or "left" in closed


@pytest.mark.asyncio
async def test_a_node_that_ends_fails_the_graph_to_its_reader(tmp_path):
    async with pipeline(tmp_path, names=NODES) as (coordinator, _, _):
        coordinator.session_graph = chain_graph()
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_format": "text"})
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        await collect(outputs, 2)
        await coordinator.close_session(coordinator.graphs[identity.id].nodes["right"])
        with pytest.raises(RuntimeError, match="session ended"):
            await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S)
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


@pytest.mark.asyncio
async def test_open_without_a_route_follows_the_pipeline_route(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, _):
        identity = await coordinator.open_session(OmniRequest(None))
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        received = await collect(outputs, 2)
        assert {item.kind for item in received} == {"data", "input_done"}
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions)
        assert [event[1] for event in event_log(events) if event[0] == "open"] == [
            "source",
            "sink",
        ]
    async with pipeline(tmp_path, names=NODES) as (coordinator, _, _):
        with pytest.raises(ValueError, match="declares no Session"):
            await coordinator.open_session(OmniRequest(None))


def vad_graph() -> SessionGraph:
    return SessionGraph(
        nodes={
            "vad": SessionNode(stages=("vad",), inputs=("input",)),
            "right": SessionNode(stages=("right",), inputs=("input", "vad:session")),
        },
        publish={"vad": ("session",)},
    )


def audio(seq: int, payload: bytes, eos: bool = False) -> TimedChunk:
    return TimedChunk("audio", seq * 100, 100, seq, payload, "pcm16", eos=eos)


@pytest.mark.asyncio
async def test_the_vad_node_publishes_speech_boundaries_that_its_readers_also_take(
    tmp_path,
):
    async with pipeline(
        tmp_path, names=["vad", "right"], factories={"vad": VAD_FACTORY}
    ) as (coordinator, events, _):
        coordinator.session_graph = vad_graph()
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_every": 1000})
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, audio(0, bytes(3200)))
        await coordinator.append_session(identity, audio(1, b"\x00\x10" * 1600))
        await coordinator.append_session(identity, audio(2, bytes(3200)))
        received = await collect(outputs, 5)
        boundaries = [
            (item.payload["type"], item.payload["t_ms"], item.t_start_ms)
            for item in received
            if item.kind == "data"
        ]
        assert boundaries == [
            ("speech.started", 100, 100),
            ("speech.stopped", 200, 200),
        ]
        assert [item.input_seq for item in received if item.kind == "input_done"] == [
            0,
            1,
            2,
        ]
        appended: list[int] = []

        def right_took_every_chunk() -> bool:
            appended.extend(
                e[3] for e in event_log(events) if e[:2] == ("append", "right")
            )
            return len(appended) >= 5

        await wait_until(right_took_every_chunk)
        assert sorted(appended) == [0, 1, 2, 3, 4]
        await coordinator.close_session(identity)


def hop_graph() -> SessionGraph:
    return SessionGraph(
        nodes={
            "left": SessionNode(stages=("left",), inputs=("input",)),
            "right": SessionNode(stages=("right",), inputs=("input", "left:session")),
        },
        publish={"left": ("session",)},
    )


@pytest.mark.asyncio
async def test_a_chunk_emitted_between_appends_reaches_the_reader_before_the_next_unit(
    tmp_path,
):
    """A node-to-node hop is not quantized to the client's appends: what a node emits after its append completes is forwarded at once."""
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = hop_graph()
        identity = await coordinator.open_session(
            OmniRequest(
                None,
                {"emit_control": "speech.stopped", "emit_after": HOP_EMIT_AFTER_S},
            )
        )
        outputs = coordinator.session_outputs(identity)
        log: list = []

        def right_appends() -> list[int]:
            log.extend(event_log(events))
            return [
                e[3]
                for e in log
                if e[:3] == ("append", "right", f"{identity.id}:right")
            ]

        await coordinator.append_session(identity, chunk(0))
        await wait_until(lambda: 1 in right_appends(), timeout=HOP_UNIT_S)
        received = await collect(outputs, 2)
        assert [(item.kind, item.input_seq) for item in received] == [
            ("input_done", 0),
            ("data", 0),
        ]
        assert received[1].payload == {"type": "speech.stopped"}
        await coordinator.append_session(identity, chunk(1))
        await wait_until(lambda: 2 in right_appends())
        assert right_appends() == [0, 1, 2]
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


@pytest.mark.asyncio
async def test_a_node_merges_the_clocks_of_its_sources(tmp_path):
    """A control from another node may lie before the client's last control; only the client's own input keeps one timeline."""
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = SessionGraph(
            hop_graph().nodes, {"right": ("text",)}
        )
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_control": "speech.started"})
        )
        outputs = coordinator.session_outputs(identity)
        late = TimedChunk("control", 5000, 0, 0, {"type": "configure"}, "session")
        await coordinator.append_session(identity, late)
        await coordinator.append_session(identity, chunk(1))
        received = await collect(outputs, 2)
        assert [item.kind for item in received] == ["input_done", "input_done"]
        await wait_until(
            lambda: ("append", "right", f"{identity.id}:right", 3) in event_log(events)
        )
        with pytest.raises(ValueError, match="moves backwards"):
            await coordinator.append_session(
                identity,
                TimedChunk("control", 10, 0, 2, {"type": "configure"}, "session"),
            )
        await coordinator.close_session(identity)


@pytest.mark.asyncio
async def test_a_node_control_reaches_its_readers_but_never_the_client(tmp_path):
    async with pipeline(tmp_path, names=NODES) as (coordinator, events, _):
        coordinator.session_graph = SessionGraph(
            nodes={
                "left": SessionNode(stages=("left",), inputs=("input",)),
                "right": SessionNode(stages=("right",), inputs=("left:session",)),
            },
            publish={"left": ("session",)},
        )
        identity = await coordinator.open_session(
            OmniRequest(None, {"emit_control": "reply.progress"})
        )
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(identity, chunk(0))
        received = await collect(outputs, 1)
        assert [item.kind for item in received] == ["input_done"]
        await wait_until(
            lambda: ("append", "right", f"{identity.id}:right", 0) in event_log(events)
        )
        await coordinator.close_session(identity)
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)


class NodeOutputsRunner(SessionGraphRunner):
    """Graph routing over one node whose outputs are given; node controls are recorded."""

    def __init__(self, node_outputs: list[OutputChunk]) -> None:
        super().__init__()
        self.node_outputs = node_outputs
        self.controls: list[TimedChunk] = []

    async def session_outputs(self, session_identity):
        for output in self.node_outputs:
            yield output

    async def control_session(
        self,
        session_identity,
        event,
        *,
        stages=None,
        should_preempt=False,
        is_ordered=True,
        preempted_request_ids=(),
    ):
        self.controls.append(event)


def reply_runtime(sessions: SessionGraphRunner) -> SessionGraphRuntime:
    """A graph of one `reply` node interrupted by the client, with its node session registered."""
    node = SessionIdentity("graph:reply")
    sessions.sessions[node.id] = Session(
        session_identity=node,
        request=OmniRequest(None),
        stages=("reply",),
        bindings={},
        limits=SessionLimits(),
        is_node=True,
    )
    return SessionGraphRuntime(
        session_identity=SessionIdentity("graph"),
        graph=SessionGraph(
            nodes={
                "reply": SessionNode(
                    ("reply",),
                    ("input",),
                    controls=(ControlEdge("input", "interrupt"),),
                )
            },
            publish={"reply": ("text_delta", "session")},
        ),
        nodes={"reply": node},
        limits=SessionLimits(),
        inboxes={"reply": NodeInbox()},
        outputs=asyncio.Queue(),
        client_sequences={"reply": {}},
    )


def reply_output(output_format: str, payload: dict, turn: int) -> OutputChunk:
    modality = "control" if output_format == "session" else "text"
    return OutputChunk(
        SessionIdentity("graph:reply"),
        0,
        0,
        modality,
        0,
        0,
        payload,
        output_format,
        turn=turn,
    )


@pytest.mark.asyncio
async def test_an_interrupted_node_publishes_only_the_end_of_the_interrupted_turn():
    sessions = NodeOutputsRunner(
        [
            reply_output("text_delta", {"text": "stale"}, 1),
            reply_output("session", {"type": "turn.done"}, 1),
            reply_output("session", {"type": "turn.interrupted"}, 1),
            reply_output("text_delta", {"text": "next"}, 2),
        ]
    )
    graph = reply_runtime(sessions)
    interrupt = TimedChunk("control", 0, 0, 0, {"type": "interrupt"}, "session", turn=1)
    await sessions.control_graph(graph, interrupt, None, False)
    assert sessions.controls == [interrupt]
    await sessions.route_graph_outputs(graph, "reply")
    published = []
    while not graph.outputs.empty():
        published.append(graph.outputs.get_nowait()[0])
    assert [(item.payload, item.turn) for item in published] == [
        ({"type": "turn.interrupted"}, 1),
        ({"text": "next"}, 2),
    ]


def report_runtime(sessions: SessionGraphRunner) -> SessionGraphRuntime:
    """A `tts` node whose session output the `aligner` reads, and a `dialog` that takes the aligner's progress on a control edge; both publish `session`."""
    nodes = {
        "tts": SessionNode(("tts",), ("input",)),
        "aligner": SessionNode(("aligner",), ("tts:session",)),
        "dialog": SessionNode(
            ("dialog",),
            ("input",),
            controls=(ControlEdge("aligner", "reply.progress"),),
        ),
    }
    identities = {name: SessionIdentity(f"graph:{name}") for name in nodes}
    for name, identity in identities.items():
        sessions.sessions[identity.id] = Session(
            session_identity=identity,
            request=OmniRequest(None),
            stages=(name,),
            bindings={},
            limits=SessionLimits(),
            is_node=True,
        )
    return SessionGraphRuntime(
        session_identity=SessionIdentity("graph"),
        graph=SessionGraph(
            nodes=nodes, publish={"tts": ("session",), "aligner": ("session",)}
        ),
        nodes=identities,
        limits=SessionLimits(),
        inboxes={name: NodeInbox() for name in nodes},
        outputs=asyncio.Queue(),
        client_sequences={name: {} for name in nodes},
    )


@pytest.mark.asyncio
async def test_a_report_reaches_the_client_only_from_a_node_no_node_reads():
    progress = {"type": "reply.progress", "text_end": 4, "audio_ms": 500.0}
    sessions = NodeOutputsRunner(
        [
            reply_output("session", progress, 1),
            reply_output("session", {"type": "turn.started"}, 1),
        ]
    )
    graph = report_runtime(sessions)
    await sessions.route_graph_outputs(graph, "tts")
    assert graph.outputs.get_nowait()[0].payload == {"type": "turn.started"}
    assert graph.outputs.empty()
    assert [chunk.payload for _, chunk, _ in graph.inboxes["aligner"].queued] == [
        progress,
        {"type": "turn.started"},
    ]
    sessions.node_outputs = [reply_output("session", progress, 1)]
    await sessions.route_graph_outputs(graph, "aligner")
    assert [event.payload for event in sessions.controls] == [progress]
    published = graph.outputs.get_nowait()[0]
    assert (published.payload, published.turn) == (progress, 1)
    assert graph.outputs.empty()


def input_runtime(
    nodes: dict[str, SessionNode], limits: SessionLimits
) -> SessionGraphRuntime:
    """An open graph of `nodes` whose tasks are not started, so client input stays in their inboxes."""
    return SessionGraphRuntime(
        session_identity=SessionIdentity("graph"),
        graph=SessionGraph(nodes=nodes, publish={}),
        nodes={name: SessionIdentity(f"graph:{name}") for name in nodes},
        limits=limits,
        inboxes={name: NodeInbox() for name in nodes},
        outputs=asyncio.Queue(maxsize=limits.max_output_chunks),
        client_sequences={name: {} for name in nodes},
    )


def test_a_graph_append_keeps_the_input_contract():
    sessions = SessionGraphRunner()
    graph = input_runtime(
        {"reply": SessionNode(("reply",), ("input",))},
        SessionLimits(max_chunk_bytes=256, max_modalities=2),
    )
    with pytest.raises(ValueError, match="max_chunk_bytes"):
        sessions.append_graph(graph, TimedChunk("audio", 0, 20, 0, bytes(512)))
    payload = {"text": "first"}
    sessions.append_graph(graph, TimedChunk("text", 0, 0, 0, payload, "text_delta"))
    payload["text"] = "edited"
    assert graph.inboxes["reply"].queued[0][1].payload == {"text": "first"}
    sessions.append_graph(graph, TimedChunk("audio", 0, 20, 1, b"pcm", eos=True))
    with pytest.raises(ValueError, match="after EOS"):
        sessions.append_graph(graph, TimedChunk("audio", 20, 20, 2, b"pcm"))
    with pytest.raises(ValueError, match="after EOS"):
        sessions.append_graph(graph, TimedChunk("image", 0, 0, 2, b"png"))
    assert graph.next_input == 2


def test_a_drop_oldest_append_that_cannot_confirm_the_dropped_input_changes_nothing():
    sessions = SessionGraphRunner()
    edge = DataEdge("input", capacity=1, overflow="drop_oldest")
    graph = input_runtime(
        {name: SessionNode((name,), (edge,)) for name in NODES},
        SessionLimits(max_output_chunks=1),
    )
    sessions.append_graph(graph, chunk(0))
    graph.outputs.put_nowait(None)
    with pytest.raises(QueueFullError):
        sessions.append_graph(graph, chunk(1))
    assert graph.next_input == 1
    assert graph.pending_input_done == {0: 2}
    assert [[item[2] for item in graph.inboxes[name].queued] for name in NODES] == [
        [0],
        [0],
    ]
    graph.outputs.get_nowait()
    assert sessions.append_graph(graph, chunk(1)) == 1
    confirmed = graph.outputs.get_nowait()[0]
    assert (confirmed.kind, confirmed.input_seq) == ("input_done", 0)
    assert graph.pending_input_done == {1: 2}


def test_control_edge_does_not_address_internal_stages():
    with pytest.raises(ValueError, match="target_stages"):
        SessionControlEdgeConfig(
            kind="control",
            source="tool",
            event="tool_response",
            target_stages=["perception"],
        )


@pytest.mark.asyncio
async def test_node_control_broadcast_and_late_tool_reply_fence(tmp_path):
    async with pipeline(tmp_path) as (coordinator, events, _):
        coordinator.session_graph = SessionGraph(
            {"model": SessionNode(("source", "sink"), ("input",))}, {}
        )
        identity = await coordinator.open_session(OmniRequest(None))
        graph = coordinator.graphs[identity.id]
        for event, turns in (("interrupt", (41, 42)), ("tool_response", (41, 42, 43))):
            for turn in turns:
                chunk = replace(
                    control_chunk(event, turn=turn),
                    format="session" if event == "interrupt" else event,
                )
                await coordinator.control_edge(
                    graph,
                    "model",
                    ControlEdge("input", event, should_preempt=True),
                    chunk,
                )
        controls = []

        def received_controls():
            controls.extend(
                event for event in event_log(events) if event[0] == "control"
            )
            return len(controls) == 6

        await wait_until(received_controls)
        assert [(event[1], event[3]) for event in controls] == [
            (stage, event)
            for event in ("interrupt", "interrupt", "tool_response")
            for stage in ("source", "sink")
        ]
        await coordinator.close_session(identity)


@pytest.mark.asyncio
async def test_an_interrupt_drops_released_audio_still_waiting_for_the_client():
    identity = SessionIdentity("graph", 1)
    graph = SessionGraphRuntime(
        session_identity=identity,
        graph=SessionGraph(
            nodes={"speaker": SessionNode(stages=("speaker",), inputs=("input",))},
            publish={"speaker": ("pcm16",)},
        ),
        nodes={"speaker": SessionIdentity("graph:speaker", 1)},
        limits=SessionLimits(max_output_chunks=1),
        inboxes={},
        outputs=asyncio.Queue(maxsize=1),
    )
    pacer = DeliveryPacer(250, asyncio.Event())
    graph.pacers["speaker"] = pacer

    def reply(kind: str | None, turn: int) -> OutputChunk:
        if kind is None:
            return OutputChunk(
                identity, 0, 0, "audio", 0, 100, bytes(3200), "pcm16", turn=turn
            )
        else:
            return OutputChunk(
                identity, 0, 0, "control", 0, 0, {"type": kind}, "session", turn=turn
            )

    filler = reply(None, 0)
    graph.outputs.put_nowait((filler, 0))
    pacer.offer(reply("turn.started", 1), 10)
    pacer.offer(reply(None, 1), 3300)
    graph.output_bytes = 3310
    runner = SessionGraphRunner()
    delivery = asyncio.create_task(runner.deliver_paced_outputs(graph, "speaker"))
    await asyncio.sleep(0.05)
    graph.interrupted_turns["speaker"] = {1}
    graph.output_bytes -= pacer.purge(1)
    assert (await graph.outputs.get())[0] is filler
    graph.output_bytes += 3300
    pacer.offer(reply(None, 1), 3300)
    await asyncio.sleep(0.05)
    assert graph.outputs.empty() and graph.output_bytes == 0
    graph.output_bytes += 10
    graph.output_bytes -= pacer.offer(reply("turn.interrupted", 1), 10)
    delivered, _ = await asyncio.wait_for(graph.outputs.get(), OUTPUT_TIMEOUT_S)
    await asyncio.sleep(0.05)
    delivery.cancel()
    await asyncio.gather(delivery, return_exceptions=True)
    assert delivered.payload == {"type": "turn.interrupted"}
    assert graph.outputs.empty()
    assert graph.output_bytes == 10


@pytest.mark.asyncio
async def test_an_interrupt_tells_the_paced_node_readers_the_audio_it_delivered():
    identity = SessionIdentity("graph", 1)
    graph = SessionGraphRuntime(
        session_identity=identity,
        graph=SessionGraph(
            nodes={
                "speaker": SessionNode(stages=("speaker",), inputs=("input",)),
                "listener": SessionNode(
                    stages=("listener",), inputs=("speaker:session",)
                ),
            },
            publish={"speaker": ("pcm16",)},
        ),
        nodes={
            "speaker": SessionIdentity("graph:speaker", 1),
            "listener": SessionIdentity("graph:listener", 1),
        },
        limits=SessionLimits(),
        inboxes={"speaker": NodeInbox(), "listener": NodeInbox()},
        outputs=asyncio.Queue(),
    )
    pacer = DeliveryPacer(250, asyncio.Event())
    graph.pacers["speaker"] = pacer
    for turn, t_start_ms in ((1, 0), (2, 0), (2, 100)):
        pacer.offer(
            OutputChunk(
                identity,
                0,
                0,
                "audio",
                t_start_ms,
                100,
                bytes(3200),
                "pcm16",
                turn=turn,
            ),
            0,
        )
    runner = SessionGraphRunner()
    delivery = asyncio.create_task(runner.deliver_paced_outputs(graph, "speaker"))
    await wait_until(lambda: graph.outputs.qsize() == 2)
    delivery.cancel()
    await asyncio.gather(delivery, return_exceptions=True)
    # Note (Junnan Li): The pacer joins turn 2's two chunks into one; only the latest turn keeps its count.
    assert (pacer.delivered_ms(1), pacer.delivered_ms(2)) == (0, 200)
    await runner.report_delivered(graph, "speaker", 2)
    await runner.report_delivered(graph, "speaker", None)
    [(source, delivered, _)] = graph.inboxes["listener"].queued
    assert source == "speaker" and delivered.turn == 2
    assert delivered.payload == {"type": REPLY_DELIVERED, "audio_ms": 200}


class TurnFront(SessionHooks):
    """Makes each client chunk the unit of a new turn and answers the client's interrupt with one naming the current turn."""

    def __init__(self) -> None:
        self.turns: dict[SessionIdentity, int] = {}
        self.emitters: dict[SessionIdentity, ChunkEmitter] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.turns[session_identity] = 0

    def attach_output_stream(
        self, session_identity: SessionIdentity, emit: ChunkEmitter
    ) -> None:
        self.emitters[session_identity] = emit

    def close(self, session_identity: SessionIdentity) -> None:
        self.turns.pop(session_identity)
        self.emitters.pop(session_identity)

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        turn = self.turns[session_identity]
        self.emitters[session_identity](control_chunk(INTERRUPT, turn=turn))

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        self.turns[context.session_identity] += 1
        turn = self.turns[context.session_identity]
        unit = TimedChunk(CONTROL, 0, 0, chunk.seq, {}, format="unit", turn=turn)
        self.emitters[context.session_identity](unit)
        return payload


class HeldRelay(SessionHooks):
    """The source holds turn 1 until it is cancelled and then hands on what it has, a payload the sink cannot take."""

    def __init__(self, name: str, events: Queue[StageEvent]) -> None:
        self.name = name
        self.events = events

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        pass

    def close(self, session_identity: SessionIdentity) -> None:
        pass

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        self.events.put(("append", self.name, context.session_identity.id, chunk.turn))
        if self.name == "source" and chunk.turn == 1:
            context.cancelled.wait(OUTPUT_TIMEOUT_S)
            payload.data = {}
        elif self.name == "source":
            payload.data = {"turn": chunk.turn}
        else:
            reply = {"turn": payload.data["turn"]}
            context.emit(TimedChunk("text", 0, 0, chunk.seq, reply, format="text"))
        return payload


def create_preempt_stage(name: str, events: Queue[StageEvent]) -> SessionScheduler:
    if name == "front":
        return SessionScheduler(TurnFront())
    else:
        return SessionScheduler(HeldRelay(name, events))


def preempt_graph() -> SessionGraph:
    return SessionGraph(
        nodes={
            "front": SessionNode(
                ("front",), ("input",), controls=(ControlEdge("input", INTERRUPT),)
            ),
            "relay": SessionNode(
                ("source", "sink"),
                ("front:unit",),
                controls=(ControlEdge("front", INTERRUPT, should_preempt=True),),
            ),
        },
        publish={"relay": ("text",)},
    )


@pytest.mark.asyncio
async def test_a_preempting_interrupt_drops_its_turn_at_every_owner_of_the_target(
    tmp_path,
):
    """The source sends no completion for the turn it dropped, so turn 2 follows only because the interrupt completes it."""
    factory = f"{__name__}.create_preempt_stage"
    async with pipeline(
        tmp_path,
        other_terminals=("front",),
        factories={"source": factory, "sink": factory, "front": factory},
    ) as (coordinator, events, _):
        coordinator.session_graph = preempt_graph()
        identity = await coordinator.open_session(OmniRequest(None))
        outputs = coordinator.session_outputs(identity)
        log: list[StageEvent] = []

        def has_appended(stage: str, turn: int) -> bool:
            log.extend(event_log(events))
            return any(
                event[0:2] == ("append", stage) and event[3] == turn for event in log
            )

        await coordinator.append_session(identity, chunk(0))
        await wait_until(lambda: has_appended("source", 1))
        await coordinator.control_session(identity, control_chunk(INTERRUPT))
        await coordinator.append_session(identity, chunk(1))
        received = []
        while not any(output.format == "text" for output in received):
            received.append(await asyncio.wait_for(anext(outputs), OUTPUT_TIMEOUT_S))
        assert [output.payload for output in received if output.format == "text"] == [
            {"turn": 2}
        ]
        assert has_appended("sink", 2) and not has_appended("sink", 1)
        await outputs.aclose()
        await wait_until(lambda: not coordinator.sessions and not coordinator.graphs)
