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
class OutputImage(ResponseEvent):
    """An image a tool of the reply made, as base64 of mime_type, with the call that asked for it."""

    item_id: str
    call_id: str
    name: str
    arguments: object
    mime_type: str
    data: str


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
class VideoPiece:
    """One piece of the file of one clip of the scene a video node shoots, with the call that asked for the clip; the pieces of a clip arrive in order."""

    call_id: str
    clip: int
    piece: int
    pieces: int
    mime_type: str
    data: bytes
    duration_s: float
    width: int
    height: int
    fps: float
    prompt: str


@dataclass(frozen=True)
class VideoProgress:
    """How far the clip of a call is: queued, generating with a percentage, done or failed."""

    call_id: str
    clip: int
    status: str
    percent: int


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
    | OutputImage
    | AudioFinished
    | ResponseFinished
    | VideoPiece
    | VideoProgress
    | TurnFailure
    | SpeechStarted
    | SpeechStopped
    | InputTranscription
)


class ContextLimitError(RuntimeError):
    code = "context_limit"
