# SPDX-License-Identifier: Apache-2.0
"""Mock graph nodes: a VAD that raises interrupts, an LLM that calls a tool, and a tool node."""

from __future__ import annotations

from collections.abc import Mapping
from multiprocessing.queues import Queue

from sglang_omni.config.schema import (
    GraphConfig,
    GraphEdgeConfig,
    GraphNodeConfig,
    StageConfig,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import (
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from sglang_omni.scheduling.tool_session import create_tool_scheduler
from tests.unit_test.fixtures.session_pipeline import StageEvent

LONG_UNIT_STEPS = 100
LONG_UNIT_STEP_S = 0.02


class GraphNodeHooks(SessionHooks):
    """Role comes from the stage name: vad or llm."""

    def __init__(self, name: str, events: Queue[StageEvent]) -> None:
        self.name = name
        self.events = events

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.events.put(("open", self.name, session_identity.id))

    def append(
        self,
        chunk: TimedChunk,
        payload: StagePayload,
        context: SessionContext,
    ) -> StagePayload:
        session_id = context.session_identity.id
        self.events.put(("append", self.name, session_id, chunk.seq))
        if self.name == "vad":
            if chunk.payload == b"speech":
                context.emit(
                    TimedChunk(
                        "interrupt", chunk.t_start_ms, 0, 0, {"reason": "speech"}
                    )
                )
            else:
                pass
        elif chunk.modality == "tool_response":
            context.emit(TimedChunk("text", chunk.t_start_ms, 0, 0, chunk.payload))
        elif chunk.payload == b"fail":
            raise RuntimeError("llm unit failed")
        elif chunk.payload == b"long":
            for _ in range(LONG_UNIT_STEPS):
                if context.cancelled.wait(LONG_UNIT_STEP_S):
                    self.events.put(("cancelled", self.name, session_id))
                    break
                else:
                    pass
        elif chunk.payload == b"call":
            context.emit(
                TimedChunk(
                    "tool_call",
                    chunk.t_start_ms,
                    chunk.duration_ms,
                    0,
                    {"calls": [{"name": "add", "arguments": {"a": 2, "b": 3}}]},
                )
            )
        else:
            assert isinstance(chunk.payload, bytes)
            context.emit(
                TimedChunk(
                    "text", chunk.t_start_ms, 0, 0, {"heard": chunk.payload.decode()}
                )
            )
        payload.data = {}
        return payload

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        self.events.put(("control", self.name, session_identity.id))

    def close(self, session_identity: SessionIdentity) -> None:
        self.events.put(("close", self.name, session_identity.id))


def add_numbers(arguments: Mapping[str, object]) -> dict[str, object]:
    left, right = arguments["a"], arguments["b"]
    if not isinstance(left, int) or not isinstance(right, int):
        raise TypeError("a and b must be integers")
    else:
        return {"sum": left + right}


def make_graph_scheduler(name: str, events: Queue[StageEvent]) -> SessionScheduler:
    return SessionScheduler(GraphNodeHooks(name, events))


def make_tool_scheduler(name: str, events: Queue[StageEvent]) -> SessionScheduler:
    """The pipeline fixture passes name and events to every factory; the tool node logs nothing."""
    return create_tool_scheduler({"add": f"{__name__}.add_numbers"})


GRAPH_STAGES = [
    StageConfig(
        name="vad",
        process="vad",
        terminal=True,
        factory_path=f"{__name__}.make_graph_scheduler",
    ),
    StageConfig(
        name="llm",
        process="llm",
        terminal=True,
        factory_path=f"{__name__}.make_graph_scheduler",
    ),
    StageConfig(
        name="tool",
        process="tool",
        terminal=True,
        factory_path=f"{__name__}.make_tool_scheduler",
    ),
]

GRAPH = GraphConfig(
    nodes={
        "vad": GraphNodeConfig(stages=["vad"]),
        "llm": GraphNodeConfig(stages=["llm"]),
        "tool": GraphNodeConfig(stages=["tool"]),
    },
    inputs={"audio": ["vad", "llm"]},
    output="llm",
    edges=[
        GraphEdgeConfig(
            source="vad",
            target="llm",
            modality="interrupt",
            kind="control",
            preempt=True,
        ),
        GraphEdgeConfig(source="llm", target="tool", modality="tool_call"),
        GraphEdgeConfig(source="tool", target="llm", modality="tool_response"),
    ],
)


def audio(sequence: int, pcm: bytes) -> TimedChunk:
    return TimedChunk("audio", sequence * 20, 20, sequence, pcm)
