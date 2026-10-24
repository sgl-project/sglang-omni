# SPDX-License-Identifier: Apache-2.0
"""Standard stream vocabulary and the ports a Session publishes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from sglang_omni.proto.session import OutputChunk, TimedChunk

AUDIO = "audio"
IMAGE = "image"
TEXT = "text"
TRANSCRIPT = "transcript"
CONTROL = "control"
TOOL = "tool"

PCM16 = "pcm16"  # bytes, signed 16-bit little-endian mono at the port's rate
PCM_F32 = "pcm_f32le"  # bytes, float32 little-endian mono at the port's rate
AUDIO_FORMATS = frozenset({PCM16, PCM_F32})
IMAGE_FRAME = "image_frame"  # bytes, one JPEG or PNG camera frame at its unit's time
PCM16_BYTES_PER_SAMPLE = 2
SESSION_INPUT_SAMPLE_RATE_HZ = 16000
SESSION_UNIT_MS = 200
MAX_HISTORY_CHARS = 64 * 1024
# {"text": str, "language"?: str}, one increment of a reply
TEXT_DELTA = "text_delta"
# {"messages": [{"role", "content"}, ...], "language"?: str}
MESSAGES = "messages"
# {"text": str, "full_text": str, "is_first_output": bool}; an ASR stage's one running text
RUNNING_TEXT = "text"
# {"text": str, "is_final": bool, "language"?: str}
TRANSCRIPT_FORMAT = "transcript"
# {"calls": [{"id": str, "name": str, "arguments": object}, ...]}
TOOL_CALL = "tool_call"
# {"responses": [{"id": str, "name": str, "response": object}, ...]}
TOOL_RESPONSE = "tool_response"
SESSION_CONTROL = "session"  # {"type": <control type>, ...}
# `turn_detection.type` of a realtime session: speech boundaries end turns, or an end-of-turn model holds a turn across a pause
SERVER_VAD = "server_vad"
SEMANTIC_VAD = "semantic_vad"

COMMIT = "commit"  # the utterance ends here
CONFIGURE = "configure"  # settings change; answered with CONFIGURED
INTERRUPT = "interrupt"  # stop the reply of the chunk's turn, or the open reply
# {"t_ms": float}; the listener heard the interrupted reply up to here
TRUNCATE = "truncate"
SPEECH_STARTED = "speech.started"  # {"t_ms": float}
SPEECH_STOPPED = "speech.stopped"  # {"t_ms": float}
TURN_QUEUED = "turn.queued"  # an utterance waits for its reply
TURN_STARTED = "turn.started"  # a reply begins
TURN_DONE = "turn.done"  # the reply and its audio are complete
TURN_EMPTY = "turn.empty"  # the utterance carried no words; no reply
TURN_INTERRUPTED = "turn.interrupted"  # the reply stopped before its end
CONFIGURED = "configured"  # {"nodes": {name: kind}}
# {"text_end": int, "audio_ms": float}; the reply text's first text_end characters have audio, ending audio_ms after the reply's first sample
REPLY_PROGRESS = "reply.progress"
# {"audio_ms": float}; an interrupt stopped the reply after audio_ms of its audio reached the client
REPLY_DELIVERED = "reply.delivered"
# Controls between nodes; a graph delivers them to readers and control edges, never to the client.
NODE_CONTROLS = frozenset({REPLY_DELIVERED})
# Reports a node makes on its own output; the client hears a report only from a node whose output no node reads as data, the end of the route that builds it.
REPORT_CONTROLS = frozenset({REPLY_PROGRESS})


def chunk_language(chunk: TimedChunk) -> str | None:
    """The language of the user's speech that a transcript, messages or text delta chunk carries; None while unknown."""
    return chunk.payload.get("language")


def control_type(chunk: TimedChunk | OutputChunk) -> str | None:
    """The control type of a SESSION_CONTROL chunk; None for any other chunk and for an input confirmation, which carries no payload."""
    if chunk.format != SESSION_CONTROL or chunk.payload is None:
        return None
    elif isinstance(chunk.payload, dict) and isinstance(chunk.payload.get("type"), str):
        return chunk.payload["type"]
    else:
        raise ValueError(
            f"a {SESSION_CONTROL!r} chunk carries a string type: {chunk!r:.300}"
        )


def edge_event(chunk: TimedChunk) -> str | None:
    """The event a control edge takes a node's chunk by: the control type of a SESSION_CONTROL chunk, the format of any other, such as `tool_response`."""
    if chunk.format == SESSION_CONTROL:
        return control_type(chunk)
    else:
        return chunk.format


def is_interrupted_reply(chunk: TimedChunk | OutputChunk, turn: int | None) -> bool:
    """A chunk of the interrupted reply other than the turn.interrupted that ends it."""
    return chunk.turn == turn and control_type(chunk) != TURN_INTERRUPTED


def holds_turn_end(
    chunk: TimedChunk | OutputChunk, publishes_audio: bool, is_published: bool
) -> bool:
    """Whether a node's turn.done waits until its published audio reached the client before its readers hear it."""
    return publishes_audio and is_published and control_type(chunk) == TURN_DONE


@dataclass(frozen=True)
class AudioPort:
    format: str
    sample_rate_hz: int


def published_audio_port(ports: Iterable[AudioPort]) -> AudioPort | None:
    """The one port all published audio shares, or None without published audio."""
    distinct = sorted(set(ports), key=lambda port: (port.format, port.sample_rate_hz))
    if len(distinct) > 1:
        raise ValueError(
            "published audio must share one format and rate, got "
            f"{[(port.format, port.sample_rate_hz) for port in distinct]}"
        )
    elif distinct:
        return distinct[0]
    else:
        return None


@dataclass(frozen=True)
class ImagePort:
    format: str


@dataclass(frozen=True)
class SessionPorts:
    """What a pipeline's Session accepts and produces; `input` is one audio port and the image ports of the nodes that see; hot_settings are the realtime session fields a CONFIGURE control may change while it is open; turn_detection_types are the `turn_detection.type` values its nodes implement."""

    input: tuple[AudioPort | ImagePort, ...]
    output: AudioPort | None
    unit_ms: int
    controls: frozenset[str] = frozenset()
    hot_settings: frozenset[str] = frozenset()
    sampling_parameters: tuple[str, ...] = ()
    turn_detection_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        audio_ports = [port for port in self.input if isinstance(port, AudioPort)]
        if len(audio_ports) != 1:
            raise ValueError(
                f"a Session takes one audio input port, got {len(audio_ports)}"
            )
        else:
            pass

    @property
    def audio_input(self) -> AudioPort:
        return next(port for port in self.input if isinstance(port, AudioPort))

    @property
    def input_modalities(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                AUDIO if isinstance(port, AudioPort) else IMAGE for port in self.input
            )
        )

    def with_node(self, node: SessionPorts) -> SessionPorts:
        """These ports joined with what another node of the same Session accepts and publishes; the unit stays this one's."""
        if node.audio_input != self.audio_input:
            raise ValueError(
                "the nodes of one Session share its audio input port, got "
                f"{(self.audio_input.format, self.audio_input.sample_rate_hz)} and "
                f"{(node.audio_input.format, node.audio_input.sample_rate_hz)}"
            )
        else:
            pass
        return replace(
            self,
            input=tuple(dict.fromkeys(self.input + node.input)),
            output=published_audio_port(
                port for port in (self.output, node.output) if port is not None
            ),
            controls=self.controls | node.controls,
            hot_settings=self.hot_settings | node.hot_settings,
            sampling_parameters=self.sampling_parameters
            + tuple(
                parameter
                for parameter in node.sampling_parameters
                if parameter not in self.sampling_parameters
            ),
            turn_detection_types=self.turn_detection_types
            + tuple(
                kind
                for kind in node.turn_detection_types
                if kind not in self.turn_detection_types
            ),
        )
