"""Pydantic models for the subset of OpenAI Realtime WebSocket events
we currently implement.

Reference: https://developers.openai.com/api/docs/guides/realtime
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer


# Forward compatibility for future event types.
class EventBase(BaseModel):
    model_config = ConfigDict(extra="allow")


class TurnDetectionType(str, Enum):
    """``turn_detection.type`` discriminator. ``str`` mixin keeps wire
    values as plain strings and lets handlers compare against either
    the enum member or its literal string."""

    SERVER_VAD = "server_vad"
    SEMANTIC_VAD = "semantic_vad"


class SemanticVADEagerness(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class TurnDetection(EventBase):
    type: TurnDetectionType = TurnDetectionType.SERVER_VAD
    threshold: float | None = None
    prefix_padding_ms: int | None = None
    silence_duration_ms: int | None = None
    eagerness: SemanticVADEagerness | None = None
    interrupt_response: bool | None = None


class SessionConfig(EventBase):
    """session.update payload. All fields optional — only set fields are applied."""

    modalities: list[str] | None = None
    instructions: str | None = None
    input_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] | None = None
    output_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] | None = None
    turn_detection: TurnDetection | None = None
    temperature: float | None = None
    max_response_output_tokens: int | str | None = None


class TranscriptionSessionConfig(EventBase):
    """session.update payload for transcription sessions."""

    model_config = ConfigDict(extra="forbid")

    language: str | None = None
    turn_detection: TurnDetection | None = None
    input_audio_format: Literal["pcm16"] | None = None


class SessionObject(EventBase):
    id: str
    object: Literal["realtime.session"] = "realtime.session"
    model: str
    capabilities: dict[str, Any] = Field(default_factory=dict)
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    instructions: str = ""
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    turn_detection: TurnDetection | None = None
    temperature: float = 0.8
    max_response_output_tokens: int | str = "inf"


# ================================
# Server Events for Transcription Sessions
# ================================


class TranscriptionServerEvent(EventBase):
    event_id: str | None = None
    event_index: int | None = None
    type: str


class TranscriptionSegment(TranscriptionServerEvent):
    type: Literal["transcription.segment"] = "transcription.segment"
    segment_id: int
    text: str
    is_final: bool


class TranscriptionCompleted(TranscriptionServerEvent):
    type: Literal["transcription.completed"] = "transcription.completed"
    text: str


class TranscriptionSpeechStarted(TranscriptionServerEvent):
    type: Literal["input_audio_buffer.speech_started"] = (
        "input_audio_buffer.speech_started"
    )
    audio_start_ms: int
    segment_id: int


class TranscriptionSpeechStopped(TranscriptionServerEvent):
    type: Literal["input_audio_buffer.speech_stopped"] = (
        "input_audio_buffer.speech_stopped"
    )
    audio_end_ms: int
    segment_id: int | None


class TranscriptionCommitted(TranscriptionServerEvent):
    type: Literal["input_audio_buffer.committed"] = "input_audio_buffer.committed"
    segment_id: int


class TranscriptionCleared(TranscriptionServerEvent):
    type: Literal["input_audio_buffer.cleared"] = "input_audio_buffer.cleared"


class TranscriptionErrorBody(EventBase):
    type: Literal["invalid_request_error", "server_error"]
    code: str
    message: str


class TranscriptionError(TranscriptionServerEvent):
    type: Literal["error"] = "error"
    error: TranscriptionErrorBody


class TranscriptionSessionObject(EventBase):
    """session payload echoed in session.created / session.updated."""

    id: str
    object: Literal["realtime.session"] = "realtime.session"
    model: str
    intent: Literal["transcription"] = "transcription"
    input_audio_format: Literal["pcm16"] = "pcm16"
    language: str | None = None
    decode_interval_ms: int
    turn_detection: TurnDetection | None = None

    @field_serializer("turn_detection")
    def _serialize_turn_detection(
        self, value: TurnDetection | None
    ) -> dict[str, Any] | None:
        # Only the settings the client actually set are echoed back.
        return value.model_dump(exclude_none=True) if value is not None else None


class TranscriptionSessionCreated(TranscriptionServerEvent):
    type: Literal["session.created"] = "session.created"
    session: TranscriptionSessionObject


class TranscriptionSessionUpdated(TranscriptionServerEvent):
    type: Literal["session.updated"] = "session.updated"
    session: TranscriptionSessionObject


# ================================
# Client Events
# ================================


class ClientEvent(EventBase):
    event_id: str | None = None
    type: str


class SessionUpdate(ClientEvent):
    type: Literal["session.update"]
    session: SessionConfig


class TranscriptionSessionUpdate(ClientEvent):
    type: Literal["session.update"]
    session: TranscriptionSessionConfig


class InputAudioBufferAppend(ClientEvent):
    type: Literal["input_audio_buffer.append"]
    audio: str  # base64-encoded raw PCM16 (or g711) per session.input_audio_format


class InputAudioBufferCommit(ClientEvent):
    type: Literal["input_audio_buffer.commit"]


class InputAudioBufferClear(ClientEvent):
    type: Literal["input_audio_buffer.clear"]


class TranscriptionDone(ClientEvent):
    type: Literal["transcription.done"]


class ResponseCancel(ClientEvent):
    type: Literal["response.cancel"]


class ConversationItemTruncate(ClientEvent):
    type: Literal["conversation.item.truncate"]
    item_id: str
    content_index: int
    audio_end_ms: int = Field(ge=0)


def make_event(event_type: str, **fields: Any) -> dict[str, Any]:
    """Construct a server event dict. event_id is filled in by the
    session loop so handlers don't have to."""
    payload: dict[str, Any] = {"type": event_type}
    for k, v in fields.items():
        if v is None:
            continue
        payload[k] = v
    return payload


_CONVERSATION_CLIENT_EVENT_TYPES: dict[str, type[ClientEvent]] = {
    "session.update": SessionUpdate,
    "input_audio_buffer.append": InputAudioBufferAppend,
    "input_audio_buffer.clear": InputAudioBufferClear,
    "response.cancel": ResponseCancel,
    "conversation.item.truncate": ConversationItemTruncate,
}

_TRANSCRIPTION_CLIENT_EVENT_TYPES: dict[str, type[ClientEvent]] = {
    "session.update": TranscriptionSessionUpdate,
    "input_audio_buffer.append": InputAudioBufferAppend,
    "input_audio_buffer.commit": InputAudioBufferCommit,
    "input_audio_buffer.clear": InputAudioBufferClear,
    "transcription.done": TranscriptionDone,
}


def _parse(
    raw: dict[str, Any], table: dict[str, type[ClientEvent]]
) -> ClientEvent | None:
    event_type = raw.get("type")
    if not isinstance(event_type, str):
        return None
    cls = table.get(event_type)
    if cls is None:
        return None
    return cls.model_validate(raw)


def parse_conversation_client_event(raw: dict[str, Any]) -> ClientEvent | None:
    """Parse one client event of a conversation session, return None if not part of its protocol."""
    return _parse(raw, _CONVERSATION_CLIENT_EVENT_TYPES)


def parse_transcription_client_event(raw: dict[str, Any]) -> ClientEvent | None:
    """Parse one client event of a transcription session, return None if not part of its protocol."""
    return _parse(raw, _TRANSCRIPTION_CLIENT_EVENT_TYPES)
