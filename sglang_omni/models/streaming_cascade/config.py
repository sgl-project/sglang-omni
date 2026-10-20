# SPDX-License-Identifier: Apache-2.0
"""Cascade components expanded into CPU stages of the coordinator Session graph."""

from collections.abc import Mapping
from typing import ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    model_serializer,
    model_validator,
)

from sglang_omni.config.schema import (
    SessionControlEdgeConfig,
    SessionDataEdgeConfig,
    SessionNodeConfig,
    StageConfig,
)
from sglang_omni.config.session_audio import AudioSessionPipelineConfig, StageGroup
from sglang_omni.models.nemotron3_5_asr.config import (
    Nemotron3_5ASRPipelineConfig,
    Nemotron3_5ASRStageConfig,
)
from sglang_omni.models.streaming_cascade.base import ComponentConfig
from sglang_omni.models.streaming_cascade.components import COMPONENTS
from sglang_omni.proto.session import (
    GRAPH_INPUT,
    ControlEdge,
    DataEdge,
    SessionGraph,
    SessionNode,
    parse_input,
)
from sglang_omni.proto.streams import (
    AUDIO_FORMATS,
    COMMIT,
    CONFIGURE,
    INTERRUPT,
    PCM16,
    REPLY_PROGRESS,
    SESSION_CONTROL,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    TRUNCATE,
    AudioPort,
    SessionPorts,
    published_audio_port,
)

NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
NODE_FIELDS = frozenset({"name", "inputs"})
INPUT_FORMATS = frozenset({PCM16, SESSION_CONTROL})
INSTRUCTIONS_FIELD = "system_prompt"
SAMPLING_FIELDS = {
    "temperature": "temperature",
    "max_new_tokens_per_unit": "max_tokens",
}
# `turn_detection` belongs to the VAD node of the Session; the cascade takes the key and changes nothing.
UPDATE_KEYS = frozenset({"nodes", "instructions", "sampling", "turn_detection"})


class CascadeSessionNodeConfig(SessionNodeConfig):
    """One node of the Session graph and the sources it reads.

    A node is a stage route (`stages`) or a node of a `kind` the pipeline's
    CPU stage runs, with that kind's options beside `kind`.
    """

    model_config = ConfigDict(extra="allow")

    stages: list[str] | None = Field(default=None, min_length=1)
    kind: str | None = None
    inputs: list[str | SessionDataEdgeConfig | SessionControlEdgeConfig] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def check_form(self) -> "CascadeSessionNodeConfig":
        if (self.stages is None) == (self.kind is None):
            raise ValueError("a session node names either stages or a kind")
        elif self.kind is None and self.model_extra:
            raise ValueError(
                f"options {sorted(self.model_extra)} belong to a node of a kind"
            )
        else:
            return self

    @property
    def options(self) -> dict[str, object]:
        """The options of the node's kind."""
        return dict(self.model_extra or {})

    @property
    def sources(self) -> frozenset[str]:
        """The nodes, or `input`, this node reads data or controls from."""
        return frozenset(
            parse_input(entry if isinstance(entry, str) else entry.source)[0]
            for entry in self.inputs
        )


class NodeSettings(BaseModel):
    """One node: `{"name", "inputs", "kind", ...options}` in JSON."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME_PATTERN)
    inputs: tuple[str | SessionDataEdgeConfig | SessionControlEdgeConfig, ...] = Field(
        min_length=1
    )
    component: SerializeAsAny[ComponentConfig]

    @model_validator(mode="before")
    @classmethod
    def split_options(cls, data: object) -> object:
        if isinstance(data, dict) and "component" not in data:
            options = {k: v for k, v in data.items() if k not in NODE_FIELDS}
            return {
                **{k: v for k, v in data.items() if k in NODE_FIELDS},
                "component": COMPONENTS.parse(options),
            }
        else:
            return data

    @model_serializer
    def flatten(self) -> dict[str, object]:
        return {
            "name": self.name,
            "inputs": [
                entry if isinstance(entry, str) else entry.model_dump()
                for entry in self.inputs
            ],
            **self.component.model_dump(),
        }

    @property
    def data_inputs(self) -> list[str]:
        return [
            entry if isinstance(entry, str) else entry.source
            for entry in self.inputs
            if not isinstance(entry, SessionControlEdgeConfig)
        ]

    @property
    def sources(self) -> frozenset[str]:
        return frozenset(parse_input(entry)[0] for entry in self.data_inputs) | {
            edge.source for edge in self.controls
        }

    def edges(self) -> dict[str, frozenset[str]]:
        """The formats this node takes from each of its sources."""
        edges: dict[str, frozenset[str]] = {}
        for entry in self.data_inputs:
            source, format = parse_input(entry)
            formats = self.consumes if format is None else frozenset({format})
            edges[source] = edges.get(source, frozenset()) | (formats & self.consumes)
        return edges

    @property
    def controls(self) -> tuple[ControlEdge, ...]:
        """The control edges this node takes controls on."""
        return tuple(
            entry.edge()
            for entry in self.inputs
            if isinstance(entry, SessionControlEdgeConfig)
        )

    @property
    def consumes(self) -> frozenset[str]:
        return COMPONENTS.kind(self.component.kind).consumes | {SESSION_CONTROL}

    @property
    def required(self) -> frozenset[str]:
        """Consumed formats one of the inputs must produce."""
        return COMPONENTS.kind(self.component.kind).required_of(self.component)

    @property
    def produces(self) -> frozenset[str]:
        return COMPONENTS.kind(self.component.kind).produces

    @property
    def opens_turns(self) -> bool:
        return COMPONENTS.kind(self.component.kind).opens_turns

    @property
    def ends_turns(self) -> bool:
        return COMPONENTS.kind(self.component.kind).ends_turns

    def has_option(self, name: str) -> bool:
        return name in type(self.component).model_fields


def unusable_inputs(
    node: NodeSettings, produced: dict[str, frozenset[str]]
) -> list[str]:
    """`node:format` inputs whose source does not produce or node not consume the format."""
    unusable = []
    for entry in node.data_inputs:
        source, format = parse_input(entry)
        if format is not None and (
            format not in node.consumes
            or format not in produced.get(source, frozenset({format}))
        ):
            unusable.append(entry)
        else:
            pass
    return unusable


class ComponentGraphConfig(BaseModel):
    """Component options and format validation; execution belongs to SessionGraphRunner."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[NodeSettings] = Field(min_length=1)
    publish: dict[str, list[str]]

    @model_validator(mode="after")
    def check_graph(self) -> "ComponentGraphConfig":
        self.check_names()
        produced = self.produced_formats()
        self.check_inputs(produced)
        self.check_publish(produced)
        self.check_published_audio()
        self.check_turn_roles()
        self.check_progress()
        return self

    def check_names(self) -> None:
        if GRAPH_INPUT in {node.name for node in self.nodes}:
            raise ValueError(f"{GRAPH_INPUT!r} names the session input, not a node")
        elif not any(GRAPH_INPUT in node.sources for node in self.nodes):
            raise ValueError(f"no node reads {GRAPH_INPUT!r}")
        else:
            pass

    @property
    def sources(self) -> dict[str, frozenset[str]]:
        """Formats read from native stage nodes."""
        names = {node.name for node in self.nodes} | {GRAPH_INPUT}
        sources: dict[str, frozenset[str]] = {}
        for node in self.nodes:
            for entry in node.data_inputs:
                source, format = parse_input(entry)
                if source not in names:
                    sources[source] = sources.get(source, frozenset()) | (
                        node.consumes if format is None else {format}
                    )
                else:
                    pass
        return sources

    def produced_formats(self) -> dict[str, frozenset[str]]:
        """The formats each source emits, the session input and the Session's other nodes among them."""
        return {
            GRAPH_INPUT: INPUT_FORMATS,
            **self.sources,
            **{node.name: node.produces for node in self.nodes},
        }

    def check_inputs(self, produced: dict[str, frozenset[str]]) -> None:
        for node in self.nodes:
            edges = node.edges()
            unusable = unusable_inputs(node, produced)
            offered = frozenset().union(
                *(formats & produced[name] for name, formats in edges.items())
            )
            missing = node.required - offered - {SESSION_CONTROL}
            if node.name in node.sources:
                raise ValueError(f"node {node.name!r} reads itself")
            elif unusable:
                raise ValueError(f"node {node.name!r} cannot take {unusable}")
            elif missing:
                raise ValueError(
                    f"no input of node {node.name!r} produces {sorted(missing)}"
                )
            else:
                pass

    def check_publish(self, produced: dict[str, frozenset[str]]) -> None:
        names = {node.name for node in self.nodes}
        for name, formats in self.publish.items():
            if name not in names:
                raise ValueError(f"cannot publish unknown node {name!r}")
            elif not set(formats) <= produced[name]:
                raise ValueError(
                    f"node {name!r} does not produce {sorted(set(formats) - produced[name])}"
                )
            else:
                pass

    def check_published_audio(self) -> None:
        published_audio_port(self.published_audio_ports())

    def published_audio_ports(self) -> list[AudioPort]:
        """The format and rate of each audio format a node publishes."""
        ports = []
        for name, formats in self.publish.items():
            rate = COMPONENTS.kind(self.node(name).component.kind).sample_rate_hz
            audio = sorted(AUDIO_FORMATS & set(formats))
            if audio and rate is None:
                raise ValueError(
                    f"published audio node {name!r} must declare its sample rate"
                )
            else:
                ports.extend(AudioPort(format, rate) for format in audio)
        return ports

    def check_turn_roles(self) -> None:
        enders = sorted(
            name
            for name, formats in self.publish.items()
            if SESSION_CONTROL in formats and self.node(name).ends_turns
        )
        silent_openers = sorted(
            node.name
            for node in self.nodes
            if node.opens_turns
            and SESSION_CONTROL not in self.publish.get(node.name, ())
        )
        if not enders and any(node.opens_turns for node in self.nodes):
            raise ValueError("no published node ends turns")
        elif len(enders) > 1:
            raise ValueError(f"turns end twice: {enders} all publish turn.done")
        elif silent_openers:
            raise ValueError(
                f"nodes {silent_openers} open turns and must publish {SESSION_CONTROL!r}"
            )
        else:
            pass

    def check_progress(self) -> None:
        """One source reports a node's reply progress, and a control edge does not divert an event a node reads as data, since an event a control edge takes skips its source's data readers."""
        diverted = {
            (edge.source, edge.event) for node in self.nodes for edge in node.controls
        }
        for node in self.nodes:
            reporters = sorted(
                {edge.source for edge in node.controls if edge.event == REPLY_PROGRESS}
            )
            lost = sorted(
                f"{source}:{event}"
                for source, formats in node.edges().items()
                if SESSION_CONTROL in formats
                for event in COMPONENTS.kind(node.component.kind).reads_events
                if (source, event) in diverted
            )
            if len(reporters) > 1:
                raise ValueError(
                    f"node {node.name!r} takes {REPLY_PROGRESS} from {reporters}; one source reports it"
                )
            elif lost:
                raise ValueError(
                    f"node {node.name!r} reads {lost} as data, but a control edge takes them"
                )
            else:
                pass

    def node(self, name: str) -> NodeSettings:
        return next(node for node in self.nodes if node.name == name)

    def apply(self, update: Mapping[str, object]) -> "ComponentGraphConfig":
        """Merge a realtime-standard update into these settings.

        `instructions`, `sampling` and `turn_detection` reach every node with
        the matching option. `nodes` maps a node name to options: an entry
        without `kind`, or with the current kind, changes only the listed
        options; a different `kind` replaces the component. Names and inputs
        cannot change.
        """
        unknown = set(update) - UPDATE_KEYS
        explicit = update.get("nodes") or {}
        if unknown:
            raise ValueError(f"unknown cascade settings {sorted(unknown)}")
        elif not isinstance(explicit, dict):
            raise ValueError("nodes must map node names to options")
        else:
            pass
        components = {node.name: node.component.model_dump() for node in self.nodes}
        for name, options in [*self.standard_options(update), *explicit.items()]:
            current = components.get(name)
            if current is None:
                raise ValueError(f"unknown node {name!r}")
            elif not isinstance(options, dict):
                raise ValueError(f"options of node {name!r} must be an object")
            elif "name" in options or "inputs" in options:
                raise ValueError("an update cannot rename or rewire a node")
            elif options.get("kind", current["kind"]) == current["kind"]:
                components[name] = {**current, **options}
            else:
                components[name] = dict(options)
        return ComponentGraphConfig.model_validate(
            {
                **self.model_dump(),
                "nodes": [
                    {
                        "name": s.name,
                        "inputs": s.model_dump()["inputs"],
                        **components[s.name],
                    }
                    for s in self.nodes
                ],
            }
        )

    def standard_options(
        self, update: Mapping[str, object]
    ) -> list[tuple[str, dict[str, object]]]:
        instructions = update.get("instructions")
        sampling = update.get("sampling") or {}
        turn_detection = update.get("turn_detection") or {}
        if instructions is not None and not isinstance(instructions, str):
            raise ValueError("instructions must be a string")
        elif not isinstance(sampling, dict) or not isinstance(turn_detection, dict):
            raise ValueError("sampling and turn_detection must be objects")
        elif set(sampling) - set(SAMPLING_FIELDS):
            raise ValueError(
                f"unsupported sampling parameters {sorted(set(sampling) - set(SAMPLING_FIELDS))}"
            )
        else:
            pass
        values: dict[str, object] = {
            SAMPLING_FIELDS[key]: value
            for key, value in sampling.items()
            if value is not None
        }
        if instructions is not None:
            values[INSTRUCTIONS_FIELD] = instructions
        else:
            pass
        options = [
            (node.name, {k: v for k, v in values.items() if node.has_option(k)})
            for node in self.nodes
        ]
        return [(name, changed) for name, changed in options if changed]

    def ports(self, unit_ms: int) -> SessionPorts:
        hot_settings = frozenset(
            setting
            for setting, fields in (
                ("instructions", (INSTRUCTIONS_FIELD,)),
                ("sampling", tuple(SAMPLING_FIELDS.values())),
            )
            if any(node.has_option(f) for node in self.nodes for f in fields)
        )
        return SessionPorts(
            input=(AudioPort(PCM16, SESSION_INPUT_SAMPLE_RATE_HZ),),
            output=published_audio_port(self.published_audio_ports()),
            unit_ms=unit_ms,
            controls=frozenset({COMMIT, CONFIGURE, INTERRUPT, TRUNCATE}),
            hot_settings=hot_settings,
            sampling_parameters=(
                tuple(SAMPLING_FIELDS) if "sampling" in hot_settings else ()
            ),
        )


NEMOTRON_ASR_MODEL_PATH = "nvidia/nemotron-3.5-asr-streaming-0.6b"


def nemotron_asr_stages() -> list[StageConfig]:
    return [
        Nemotron3_5ASRStageConfig(
            name="asr",
            process="asr",
            gpu=0,
            terminal=True,
            factory_path="sglang_omni.models.nemotron3_5_asr.stages.create_nemotron3_5_asr_executor",
        )
    ]


class StreamingCascadePipelineConfig(AudioSessionPipelineConfig):
    """Declare cascade components and optional native ASR stages as a Session graph."""

    architecture: ClassVar[str] = "StreamingCascadePipeline"
    session_nodes: dict[str, CascadeSessionNodeConfig]
    stage_groups: ClassVar[tuple[StageGroup, ...]] = (
        *AudioSessionPipelineConfig.stage_groups,
        StageGroup(
            nemotron_asr_stages,
            Nemotron3_5ASRPipelineConfig,
            None,
            NEMOTRON_ASR_MODEL_PATH,
        ),
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        **AudioSessionPipelineConfig.stage_config_types,
        **Nemotron3_5ASRPipelineConfig.stage_config_types,
    }
    timeout_s: float = Field(default=120, gt=0, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def add_stages(cls, data: object) -> object:
        """Each component gets one CPU stage; native stage routes keep their stages."""
        if (
            not isinstance(data, dict)
            or "stages" in data
            or "session_nodes" not in data
        ):
            return data
        else:
            pass
        nodes = {
            name: CascadeSessionNodeConfig.model_validate(node)
            for name, node in data["session_nodes"].items()
        }
        assembled = super().add_stages(
            {
                **data,
                "session_nodes": {
                    name: {"stages": node.stages, "inputs": node.inputs}
                    for name, node in nodes.items()
                    if node.kind is None
                },
            }
        )
        assembled["session_nodes"] = data["session_nodes"]
        for name, node in nodes.items():
            if node.kind is not None:
                assembled["stages"].append(
                    StageConfig(
                        name=name,
                        process="cascade",
                        terminal=True,
                        factory_path="sglang_omni.models.streaming_cascade.session.create_scheduler",
                    )
                )
            else:
                pass
        assembled["entry_stage"] = (
            data.get("entry_stage") or assembled["stages"][0].name
        )
        return assembled

    def check_local_conflicts(self) -> None:
        """Check the final declaration before building its Session graph."""
        super().check_local_conflicts()
        if any(node.kind is not None for node in self.session_nodes.values()):
            self.component_graph()
        else:
            pass

    def session_graph(self) -> SessionGraph:
        """Expand component shorthands without adding another router or host node."""
        nodes = {}
        for name, node in self.session_nodes.items():
            inputs = []
            consumes = (
                COMPONENTS.kind(node.kind).consumes | {SESSION_CONTROL}
                if node.kind
                else None
            )
            for entry in node.inputs:
                if isinstance(entry, SessionControlEdgeConfig):
                    continue
                else:
                    source, format = parse_input(
                        entry if isinstance(entry, str) else entry.source
                    )
                formats = (
                    (format,) if format else (sorted(consumes) if consumes else (None,))
                )
                for accepted in formats:
                    selected = source if accepted is None else f"{source}:{accepted}"
                    inputs.append(
                        selected
                        if isinstance(entry, str)
                        else DataEdge(
                            selected,
                            entry.capacity,
                            entry.overflow,
                            entry.feedback,
                        )
                    )
            nodes[name] = SessionNode(
                stages=(name,) if node.kind else tuple(node.stages),
                inputs=tuple(inputs),
                controls=tuple(
                    entry.edge()
                    for entry in node.inputs
                    if isinstance(entry, SessionControlEdgeConfig)
                ),
            )
        return SessionGraph(
            shared_timeline=True,
            nodes=nodes,
            publish={
                name: tuple(formats) for name, formats in self.session_publish.items()
            },
        )

    def component_graph(self) -> ComponentGraphConfig:
        """Validate component options, formats and publication at configuration time."""
        components = {
            name: node
            for name, node in self.session_nodes.items()
            if node.kind is not None
        }
        return ComponentGraphConfig.model_validate(
            {
                "nodes": [
                    {
                        "name": name,
                        "inputs": node.inputs,
                        "kind": node.kind,
                        **node.options,
                    }
                    for name, node in components.items()
                ],
                "publish": {
                    name: formats
                    for name, formats in self.session_publish.items()
                    if name in components
                },
            }
        )

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        node = self.session_nodes.get(stage_name)
        if node is not None and node.kind is not None:
            return {
                "name": stage_name,
                "settings": self.component_graph().model_dump(mode="json"),
                "timeout_s": self.timeout_s,
            }
        else:
            return super().stage_factory_kwargs(stage_name)

    def node_ports(
        self, name: str, node: CascadeSessionNodeConfig
    ) -> SessionPorts | None:
        if node.kind is not None:
            return self.component_graph().ports(self.unit_ms)
        else:
            return super().node_ports(name, node)


EntryClass = StreamingCascadePipelineConfig
