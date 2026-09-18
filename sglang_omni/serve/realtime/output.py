"""Transport independent outputs from realtime interaction adapters."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpeechBoundary:
    started: bool
    item_id: str
    time_ms: float


@dataclass(frozen=True)
class InputCommitted:
    item_id: str


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
    include_audio: bool
    status: str
    reason: str
    usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class TranscriptionDelta:
    item_id: str
    text: str
    segment_id: str | None = None
    start_ms: float = 0
    end_ms: float = 0


@dataclass(frozen=True)
class TranscriptionFinished(TranscriptionDelta):
    revision_id: int | None = None


@dataclass(frozen=True)
class TranscriptionRevision(TranscriptionDelta):
    base_revision_id: int = 0
    revision_id: int = 1


@dataclass(frozen=True)
class TranscriptionSegment(TranscriptionFinished):
    """Final text for one segment; item completion is a separate event."""


@dataclass(frozen=True)
class TranscriptionFailure:
    item_id: str
    code: str
    message: str


@dataclass(frozen=True)
class TurnFailure:
    type: str
    code: str
    message: str


OutputEvent = (
    SpeechBoundary
    | InputCommitted
    | ResponseStarted
    | TextDelta
    | TextFinished
    | AudioDelta
    | AudioFinished
    | ResponseFinished
    | TranscriptionDelta
    | TranscriptionFinished
    | TranscriptionRevision
    | TranscriptionSegment
    | TranscriptionFailure
    | TurnFailure
)


class ContextLimitError(RuntimeError):
    code = "context_limit"
