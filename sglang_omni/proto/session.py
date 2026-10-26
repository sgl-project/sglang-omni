# SPDX-License-Identifier: Apache-2.0
"""Model-independent, bounded session operation and output contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import msgpack
import msgspec
from typing_extensions import NotRequired

SESSION_METADATA_KEY: Literal["omni_session"] = "omni_session"
GRAPH_INPUT = "input"
# Note (Junnan Li): msgspec encodes bytes as base64 text by default; keep them native on both sides.
BUILTIN_TYPES = (bytes,)
ChunkPayload = bytes | dict[str, object] | None
DEFAULT_MAX_MODALITIES = 8
DEFAULT_MAX_PENDING_CHUNKS = 16
DEFAULT_MAX_PENDING_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_OUTPUT_CHUNKS = 64
DEFAULT_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_CHUNK_BYTES = 1024 * 1024
DEFAULT_OPERATION_TIMEOUT_S = 30.0
DEFAULT_IDLE_TIMEOUT_S = 300.0
DEFAULT_SESSION_UPDATE_TIMEOUT_S = 60.0
DEFAULT_MAX_OUTPUT_LEAD_MS = 1000.0
# Note (Junnan Li): Msgpack bin headers grow by 1 byte at 256 bytes and 3 bytes at 65536.
MSGPACK_BIN8_LIMIT = 256
MSGPACK_BIN16_LIMIT = 65536
MSGPACK_BIN16_HEADER_GROWTH = 1
MSGPACK_BIN32_HEADER_GROWTH = 3


class SessionIdentityDict(TypedDict):
    id: str
    open_index: int


class TimedChunkDict(TypedDict):
    modality: str
    t_start_ms: float
    duration_ms: float
    seq: int
    payload: ChunkPayload
    format: str | None
    eos: bool
    turn: int | None


class OutputChunkDict(TypedDict):
    session_identity: SessionIdentityDict
    seq: int
    input_seq: int
    modality: str
    t_start_ms: float
    duration_ms: float
    payload: ChunkPayload
    format: str | None
    eos: bool
    turn: int | None
    kind: Literal["data", "input_done"]


SessionOperationKind = Literal["open", "append", "close", "control", "read"]


class SessionOperationDict(TypedDict):
    operation: SessionOperationKind
    session_identity: SessionIdentityDict
    stages: list[str]
    chunk: TimedChunkDict | None
    should_preempt: bool
    preempted_request_ids: NotRequired[list[str]]


@dataclass(frozen=True)
class SessionIdentity:
    """One open of an id. open_index is the number issued for that open."""

    id: str
    open_index: int = 1

    def to_dict(self) -> SessionIdentityDict:
        return {"id": self.id, "open_index": self.open_index}


@dataclass(frozen=True)
class TimedChunk:
    """Input seq is global within one open_index; turn identifies one cancellable reply.

    All producers contributing to a reply preserve its turn, including tools.
    Tool call IDs identify individual calls; they are not reply turns.
    A producer without a reply-turn contract leaves turn unset."""

    modality: str
    t_start_ms: float
    duration_ms: float
    seq: int
    payload: ChunkPayload
    format: str | None = None
    eos: bool = False
    turn: int | None = None

    def to_dict(self) -> TimedChunkDict:
        return {
            "modality": self.modality,
            "t_start_ms": self.t_start_ms,
            "duration_ms": self.duration_ms,
            "seq": self.seq,
            "payload": self.payload,
            "format": self.format,
            "eos": self.eos,
            "turn": self.turn,
        }

    @classmethod
    def from_dict(cls, data: object) -> TimedChunk:
        if not isinstance(data, dict):
            raise ValueError("timed chunk must be an object")
        else:
            return msgspec.convert(
                data, type=cls, strict=True, builtin_types=BUILTIN_TYPES
            )


@dataclass(frozen=True)
class OutputChunk:
    """input_seq identifies the originating pipeline input, not a stream seq."""

    session_identity: SessionIdentity
    seq: int
    input_seq: int
    modality: str
    t_start_ms: float
    duration_ms: float
    payload: ChunkPayload
    format: str | None = None
    eos: bool = False
    turn: int | None = None
    kind: Literal["data", "input_done"] = "data"

    def to_dict(self) -> OutputChunkDict:
        return {
            "session_identity": self.session_identity.to_dict(),
            "seq": self.seq,
            "input_seq": self.input_seq,
            "modality": self.modality,
            "t_start_ms": self.t_start_ms,
            "duration_ms": self.duration_ms,
            "payload": self.payload,
            "format": self.format,
            "eos": self.eos,
            "turn": self.turn,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: object) -> OutputChunk:
        if not isinstance(data, dict):
            raise ValueError("output chunk must be an object")
        else:
            return msgspec.convert(
                data, type=cls, strict=True, builtin_types=BUILTIN_TYPES
            )


def parse_input(entry: str) -> tuple[str, str | None]:
    """The source and format of an input written `source` or `source:format`; a None format takes every format."""
    source, _, format = entry.partition(":")
    return source, format or None


def control_targets(
    controls: Iterable[tuple[str, Iterable[ControlEdge]]],
    source: str,
    event: str | None,
) -> tuple[tuple[str, ControlEdge], ...]:
    """The nodes, with their edge, among (node, control edges) that take this control event of this source, in the given order."""
    return tuple(
        (name, edge)
        for name, edges in controls
        for edge in edges
        if edge.source == source and edge.event == event
    )


@dataclass(frozen=True)
class ControlEdge:
    """A control event of `source` delivered to the node as a control operation, not as input.

    `source` is `input` or a node, `event` the control type it carries;
    Every stage on the node route receives the event in route order;
    `should_preempt` cancels the node's in-flight appends first.
    """

    source: str
    event: str
    should_preempt: bool = False


@dataclass(frozen=True)
class DataEdge:
    """A data input of a node written with the bound of its edge; `source` is written like a plain input.

    The edge from one source holds at most `capacity` chunks the node has
    not taken yet, `SessionLimits.max_pending_chunks` when None. When it is
    full, `block` holds the producer and `drop_oldest` drops the edge's
    oldest chunk. On the client's input edge `block` rejects the append, and
    `drop_oldest` confirms the dropped input to the client as taken, so a
    real-time node keeps the newest input. `feedback` marks the edge that closes a cycle: the graph's
    order leaves it out, and it must not block, since two nodes each held
    on the other's full edge never wake.
    """

    source: str
    capacity: int | None = None
    overflow: Literal["block", "drop_oldest"] = "block"
    feedback: bool = False


@dataclass(frozen=True)
class SessionNode:
    """One node of a session graph: a linear stage route and the sources it reads.

    An input is `input` (the session input), `input:<format>`, `<node>` or
    `<node>:<format>`, or a `DataEdge` naming one of these with the bound
    of its edge; without a format the node takes every chunk its source
    emits. The inputs naming one source are one edge. `controls` are the
    control edges the node receives.
    """

    stages: tuple[str, ...]
    inputs: tuple[str | DataEdge, ...]
    controls: tuple[ControlEdge, ...] = ()

    def edges(self) -> dict[str, frozenset[str] | None]:
        """Formats taken from each source; None takes every format."""
        edges: dict[str, frozenset[str] | None] = {}
        for entry in self.inputs:
            source, format = parse_input(
                entry if isinstance(entry, str) else entry.source
            )
            formats = edges.get(source, frozenset())
            if format is None or formats is None:
                edges[source] = None
            else:
                edges[source] = formats | {format}
        return edges

    def bound(self, source: str) -> DataEdge:
        """The edge from `source` with its declared bound, or with the default one."""
        return next(
            (
                entry
                for entry in self.inputs
                if isinstance(entry, DataEdge)
                and parse_input(entry.source)[0] == source
            ),
            DataEdge(source),
        )


@dataclass(frozen=True)
class SessionGraph:
    """Nodes routed by chunk format; `publish` names the formats of each node that reach the client.

    `order` is the nodes upstream first; a cycle of data edges is declared
    by a feedback edge (see `DataEdge`).
    """

    nodes: dict[str, SessionNode]
    publish: dict[str, tuple[str, ...]]
    # note (Junnan Li): Reply components may emit untimed chunks; the coordinator gives their published outputs one shared media clock.
    shared_timeline: bool = False

    def __post_init__(self) -> None:
        if not self.nodes or GRAPH_INPUT in self.nodes:
            raise ValueError(
                f"a graph names at least one node, none called {GRAPH_INPUT!r}"
            )
        else:
            pass
        for name, node in self.nodes.items():
            if not node.stages or not (node.inputs or node.controls):
                raise ValueError(f"node {name!r} needs a stage route and inputs")
            else:
                pass
            sources = set(node.edges()) | {edge.source for edge in node.controls}
            unknown = sorted(
                source
                for source in sources
                if source != GRAPH_INPUT and source not in self.nodes
            )
            if unknown or name in sources:
                raise ValueError(
                    f"node {name!r} reads unknown inputs {unknown or [name]}"
                )
            else:
                pass
            for source in node.edges():
                bounds = {
                    (entry.capacity, entry.overflow, entry.feedback)
                    for entry in node.inputs
                    if isinstance(entry, DataEdge)
                    and parse_input(entry.source)[0] == source
                }
                bound = node.bound(source)
                if len(bounds) > 1:
                    raise ValueError(
                        f"node {name!r} bounds its edge from {source!r} twice: {sorted(bounds, key=str)}"
                    )
                elif bound.capacity is not None and bound.capacity < 1:
                    raise ValueError(
                        f"the edge from {source!r} into {name!r} needs a capacity of at least 1"
                    )
                elif bound.feedback and bound.overflow == "block":
                    raise ValueError(
                        f"the feedback edge from {source!r} into {name!r} must not block"
                    )
                else:
                    pass
        if not self.input_nodes():
            raise ValueError(f"no node reads {GRAPH_INPUT!r}")
        elif not set(self.publish) <= set(self.nodes):
            raise ValueError(
                f"publish names unknown nodes {sorted(set(self.publish) - set(self.nodes))}"
            )
        else:
            self.order()

    def order(self) -> tuple[str, ...]:
        """Node names, upstream first, ties in declaration order.

        Data edges other than feedback edges and control edges order the
        nodes. A control edge holds no queue, so where it closes a cycle the
        data edges alone order its nodes; a cycle of data edges needs a
        feedback edge.
        """
        data_sources = {
            name: {
                source
                for source in node.edges()
                if source != GRAPH_INPUT and not node.bound(source).feedback
            }
            for name, node in self.nodes.items()
        }
        sources = {
            name: data_sources[name]
            | {edge.source for edge in node.controls if edge.source != GRAPH_INPUT}
            for name, node in self.nodes.items()
        }
        order: list[str] = []
        while len(order) < len(self.nodes):
            remaining = [name for name in self.nodes if name not in order]
            ready = [name for name in remaining if sources[name] <= set(order)] or [
                name for name in remaining if data_sources[name] <= set(order)
            ]
            if not ready:
                raise ValueError(
                    f"nodes {sorted(remaining)} form a cycle of data edges; "
                    "declare the input that closes it with feedback: true"
                )
            else:
                order.append(ready[0])
        return tuple(order)

    def input_nodes(self) -> tuple[str, ...]:
        return tuple(
            name for name, node in self.nodes.items() if GRAPH_INPUT in node.edges()
        )

    def reading_nodes(self, source: str, format: str | None) -> tuple[str, ...]:
        """Nodes that take chunks of this format from this source."""
        return tuple(
            name
            for name, node in self.nodes.items()
            if source in node.edges()
            and (node.edges()[source] is None or format in node.edges()[source])
        )

    def control_targets(
        self, source: str, event: str | None
    ) -> tuple[tuple[str, ControlEdge], ...]:
        """Nodes, with their edge, that take this control event of this source, upstream first: a target's stale output is fenced before its readers hear the event."""
        return control_targets(
            ((name, self.nodes[name].controls) for name in self.order()), source, event
        )

    def is_published(self, source: str, format: str | None) -> bool:
        return format in self.publish.get(source, ())


@dataclass(frozen=True)
class ResourceUsage:
    kv_tokens: int = 0
    slots: dict[str, int] = field(default_factory=dict)
    bytes: int = 0


@dataclass(frozen=True)
class SessionLimits:
    max_modalities: int = DEFAULT_MAX_MODALITIES
    max_pending_chunks: int = DEFAULT_MAX_PENDING_CHUNKS
    max_pending_bytes: int = DEFAULT_MAX_PENDING_BYTES
    max_output_chunks: int = DEFAULT_MAX_OUTPUT_CHUNKS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES
    operation_timeout_s: float = DEFAULT_OPERATION_TIMEOUT_S
    idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S


@dataclass(frozen=True)
class SessionOperation:
    """Coordinator-to-stage session operation, carried in request metadata.

    An append carries the input chunk; a control carries its event as the
    chunk and targets one stage, and with should_preempt that stage cancels
    the session's in-flight appends as soon as the control arrives. A graph's
    interrupt also names its turn's appends in preempted_request_ids; each
    stage it reaches aborts them, dropping their results and anything they
    send it later. A read
    targets the terminal owner and stays open until the close answers it; the
    chunks the stage emits outside an append stream on it.
    """

    operation: SessionOperationKind
    session_identity: SessionIdentity
    stages: tuple[str, ...]
    chunk: TimedChunk | None = None
    should_preempt: bool = False
    preempted_request_ids: tuple[str, ...] = ()

    def to_dict(self) -> SessionOperationDict:
        fields: SessionOperationDict = {
            "operation": self.operation,
            "session_identity": self.session_identity.to_dict(),
            "stages": list(self.stages),
            "chunk": None if self.chunk is None else self.chunk.to_dict(),
            "should_preempt": self.should_preempt,
        }
        if self.preempted_request_ids:
            fields["preempted_request_ids"] = list(self.preempted_request_ids)
        else:
            pass
        return fields

    @classmethod
    def from_dict(cls, data: object) -> SessionOperation:
        if not isinstance(data, dict):
            raise ValueError("session operation must be an object")
        else:
            return msgspec.convert(
                data, type=cls, strict=True, builtin_types=BUILTIN_TYPES
            )


def find_session_operation(
    metadata: Mapping[str, object],
) -> SessionOperation | None:
    """Return the operation in request metadata, or None for an ordinary request."""
    operation_fields = metadata.get(SESSION_METADATA_KEY)
    if operation_fields is None:
        return None
    else:
        return SessionOperation.from_dict(operation_fields)


def wire_size(chunk_fields: TimedChunkDict | OutputChunkDict) -> int:
    """Return the msgpack wire size of a chunk dict without copying a binary payload."""
    payload = chunk_fields["payload"]
    if isinstance(payload, bytes):
        payload_size = len(payload)
        if payload_size < MSGPACK_BIN8_LIMIT:
            header_growth = 0
        elif payload_size < MSGPACK_BIN16_LIMIT:
            header_growth = MSGPACK_BIN16_HEADER_GROWTH
        else:
            header_growth = MSGPACK_BIN32_HEADER_GROWTH
        packed_without_payload = msgpack.packb(
            {**chunk_fields, "payload": b""}, use_bin_type=True
        )
        return len(packed_without_payload) + payload_size + header_growth
    else:
        return len(msgpack.packb(chunk_fields, use_bin_type=True))
