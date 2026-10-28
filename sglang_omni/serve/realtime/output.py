"""Transport independent outputs from realtime interaction adapters."""

from dataclasses import dataclass
from typing import Literal

from sglang_omni.proto.session import ChunkPayload, OutputChunk

ResponseStatus = Literal["completed", "cancelled", "failed", "incomplete"]


@dataclass(frozen=True)
class ResponseEvent:
    response_id: str


@dataclass(frozen=True)
class ResponseStarted(ResponseEvent):
    pass


@dataclass(frozen=True)
class TextDelta(ResponseEvent):
    item_id: str
    text: str


@dataclass(frozen=True)
class TextFinished(ResponseEvent):
    item_id: str
    text: str


@dataclass(frozen=True)
class AudioDelta(ResponseEvent):
    item_id: str
    pcm: bytes


@dataclass(frozen=True)
class ResponseProgress(ResponseEvent):
    """The reply text's first text_end characters have audio, ending audio_ms after the reply's first sample."""

    item_id: str
    text_end: int
    audio_ms: float


@dataclass(frozen=True)
class ResponseHeard(ResponseEvent):
    """After the client's truncate, the history keeps the ended reply's first text_end characters; it follows the response's end."""

    item_id: str
    text_end: int


@dataclass(frozen=True)
class AudioFinished(ResponseEvent):
    item_id: str


@dataclass(frozen=True)
class ResponseFinished(ResponseEvent):
    item_id: str
    text: str
    has_audio: bool
    status: ResponseStatus
    reason: str
    usage: dict[str, int | float | None] | None = None


@dataclass(frozen=True)
class TurnFailure:
    error_type: str
    code: str
    message: str


@dataclass(frozen=True)
class SpeechStarted:
    """Turn-based adapters report where the detector placed the utterance start."""

    item_id: str
    audio_start_ms: float


@dataclass(frozen=True)
class SpeechStopped:
    item_id: str
    audio_end_ms: float


@dataclass(frozen=True)
class PluginChunk:
    """A published chunk of a plugin format, one the realtime protocol does not interpret, sent as the Session produced it."""

    response_id: str | None
    modality: str
    format: str
    t_start_ms: float
    duration_ms: float
    turn: int | None
    payload: ChunkPayload

    @classmethod
    def from_output(cls, output: OutputChunk, response_id: str | None) -> "PluginChunk":
        return cls(
            response_id=response_id,
            modality=output.modality,
            format=output.format,
            t_start_ms=output.t_start_ms,
            duration_ms=output.duration_ms,
            turn=output.turn,
            payload=output.payload,
        )


@dataclass(frozen=True)
class InputTranscription:
    """Transcript of the user's utterance item; is_final closes the item."""

    item_id: str
    text: str
    is_final: bool


OutputEvent = (
    ResponseStarted
    | TextDelta
    | TextFinished
    | AudioDelta
    | ResponseProgress
    | ResponseHeard
    | AudioFinished
    | ResponseFinished
    | TurnFailure
    | SpeechStarted
    | SpeechStopped
    | PluginChunk
    | InputTranscription
)


class ContextLimitError(RuntimeError):
    code = "context_limit"
