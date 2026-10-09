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


OutputEvent = (
    ResponseStarted
    | TextDelta
    | TextFinished
    | AudioDelta
    | AudioFinished
    | ResponseFinished
    | TurnFailure
)


class ContextLimitError(RuntimeError):
    code = "context_limit"
