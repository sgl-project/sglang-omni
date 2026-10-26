"""Transport independent outputs from realtime interaction adapters."""

from dataclasses import dataclass
from typing import Literal

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
    | InputTranscription
)


class ContextLimitError(RuntimeError):
    code = "context_limit"
