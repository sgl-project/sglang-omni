# SPDX-License-Identifier: Apache-2.0
"""The registry of component kinds, by name, so a deployment or a running Session can replace one node while the others keep their state."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sglang_omni.models.streaming_cascade.alignment import (
    Alignment,
    AlignmentConfig,
    probe_alignment_worker,
)
from sglang_omni.models.streaming_cascade.base import (
    SPEECH_SAMPLE_RATE_HZ,
    ComponentConfig,
    StreamComponent,
    probe_openai_service,
)
from sglang_omni.models.streaming_cascade.dialog import Dialog, DialogConfig
from sglang_omni.models.streaming_cascade.generators import (
    OpenAIChatGenerator,
    OpenAIChatGeneratorConfig,
    probe_chat_service,
)
from sglang_omni.models.streaming_cascade.recognizers import (
    OpenAITranscriptionRecognizer,
    OpenAITranscriptionRecognizerConfig,
    RealtimeTranscriptionRecognizer,
    RealtimeTranscriptionRecognizerConfig,
    probe_realtime_service,
)
from sglang_omni.models.streaming_cascade.running_transcript import (
    RunningTranscriptConfig,
)
from sglang_omni.models.streaming_cascade.synthesizers import (
    OpenAISpeechSynthesizer,
    OpenAISpeechSynthesizerConfig,
)
from sglang_omni.proto.streams import (
    MESSAGES,
    PCM16,
    PCM_F32,
    REPLY_DELIVERED,
    REPLY_PROGRESS,
    RUNNING_TEXT,
    SESSION_CONTROL,
    TEXT_DELTA,
    TRANSCRIPT_FORMAT,
    TRUNCATE,
)
from sglang_omni.utils.imports import import_string

MAX_PENDING_OUTPUT_CHUNKS = 1024
MAX_PENDING_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_NODE_QUEUE_CHUNKS = 1024

OptionalFormats = frozenset[str] | Callable[["ComponentConfig"], frozenset[str]]


@dataclass(frozen=True)
class ComponentKind:
    """A registered kind. `build` is a constructor or its import path.

    `opens_turns` marks a kind that numbers turns with turn.queued and
    `ends_turns` one whose turn.done ends a reply. `optional` names consumed
    formats a node runs without, as a set or as a function of the node's
    config. `probe` checks at startup that the service a configured node uses
    answers. `reads_events` names the control events the kind reads among
    its data inputs, which a control edge taking them from the same source
    would divert.
    """

    config_cls: type[ComponentConfig]
    build: Callable[..., StreamComponent] | str
    consumes: frozenset[str]
    produces: frozenset[str]
    sample_rate_hz: int | None = None
    opens_turns: bool = False
    ends_turns: bool = False
    optional: OptionalFormats = frozenset()
    probe: Callable[[ComponentConfig], Awaitable[None]] | None = None
    reads_events: frozenset[str] = frozenset()

    def required_of(self, config: ComponentConfig) -> frozenset[str]:
        if callable(self.optional):
            return self.consumes - self.optional(config)
        else:
            return self.consumes - self.optional


class ComponentRegistry:
    def __init__(self) -> None:
        self.kinds: dict[str, ComponentKind] = {}

    def register(
        self,
        config_cls: type[ComponentConfig],
        build: Callable[..., StreamComponent] | str,
        *,
        consumes: frozenset[str],
        produces: frozenset[str],
        sample_rate_hz: int | None = None,
        opens_turns: bool = False,
        ends_turns: bool = False,
        optional: OptionalFormats = frozenset(),
        probe: Callable[[ComponentConfig], Awaitable[None]] | None = None,
        reads_events: frozenset[str] = frozenset(),
    ) -> None:
        kind = config_cls.model_fields["kind"].default
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{config_cls.__name__} must default its kind")
        elif not consumes or not produces:
            raise ValueError(f"component kind {kind!r} must consume and produce")
        else:
            self.kinds[kind] = ComponentKind(
                config_cls,
                build,
                consumes,
                produces,
                sample_rate_hz,
                opens_turns,
                ends_turns,
                optional,
                probe,
                reads_events,
            )

    def kind(self, name: str) -> ComponentKind:
        if name not in self.kinds:
            raise ValueError(f"unknown component kind {name!r}")
        else:
            return self.kinds[name]

    def parse(self, spec: object) -> ComponentConfig:
        if isinstance(spec, ComponentConfig):
            return spec
        elif not isinstance(spec, dict) or not isinstance(spec.get("kind"), str):
            raise ValueError("a component requires a string kind")
        else:
            return self.kind(spec["kind"]).config_cls.model_validate(spec)

    def build(self, config: ComponentConfig) -> StreamComponent:
        build = self.kind(config.kind).build
        factory = import_string(build) if isinstance(build, str) else build
        return factory(config)


COMPONENTS = ComponentRegistry()


RECOGNIZER_INPUTS = frozenset({PCM16, SESSION_CONTROL})
COMPONENTS.register(
    RunningTranscriptConfig,
    "sglang_omni.models.streaming_cascade.running_transcript.build_running_transcript",
    consumes=RECOGNIZER_INPUTS | {RUNNING_TEXT},
    produces=frozenset({TRANSCRIPT_FORMAT}),
)
COMPONENTS.register(
    RealtimeTranscriptionRecognizerConfig,
    RealtimeTranscriptionRecognizer,
    consumes=RECOGNIZER_INPUTS,
    produces=frozenset({TRANSCRIPT_FORMAT}),
    probe=probe_realtime_service,
)
COMPONENTS.register(
    OpenAITranscriptionRecognizerConfig,
    OpenAITranscriptionRecognizer,
    consumes=RECOGNIZER_INPUTS,
    produces=frozenset({TRANSCRIPT_FORMAT}),
    probe=probe_openai_service,
)


COMPONENTS.register(
    DialogConfig,
    Dialog,
    consumes=frozenset({TRANSCRIPT_FORMAT, TEXT_DELTA, SESSION_CONTROL}),
    produces=frozenset({MESSAGES, SESSION_CONTROL}),
    opens_turns=True,
)
COMPONENTS.register(
    OpenAIChatGeneratorConfig,
    OpenAIChatGenerator,
    consumes=frozenset({MESSAGES}),
    produces=frozenset({TEXT_DELTA, SESSION_CONTROL}),
    ends_turns=True,
    probe=probe_chat_service,
)
COMPONENTS.register(
    OpenAISpeechSynthesizerConfig,
    OpenAISpeechSynthesizer,
    consumes=frozenset({TEXT_DELTA, SESSION_CONTROL}),
    produces=frozenset({PCM_F32, SESSION_CONTROL}),
    sample_rate_hz=SPEECH_SAMPLE_RATE_HZ,
    ends_turns=True,
    probe=probe_openai_service,
)
COMPONENTS.register(
    AlignmentConfig,
    Alignment,
    consumes=frozenset({TEXT_DELTA, PCM_F32, SESSION_CONTROL}),
    produces=frozenset({SESSION_CONTROL}),
    probe=probe_alignment_worker,
    reads_events=frozenset({REPLY_PROGRESS, REPLY_DELIVERED, TRUNCATE}),
)
