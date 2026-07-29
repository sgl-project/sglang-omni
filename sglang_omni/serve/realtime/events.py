"""Pydantic models for the subset of OpenAI Realtime WebSocket events
we currently implement.

Reference: https://developers.openai.com/api/docs/guides/realtime
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Forward compatibility for future event types.
class EventBase(BaseModel):
    model_config = ConfigDict(extra="allow")


class TurnDetectionType(str, Enum):
    """``turn_detection.type`` discriminator. ``str`` mixin keeps wire
    values as plain strings and lets handlers compare against either
    the enum member or its literal string."""

    SERVER_VAD = "server_vad"
    SEMANTIC_VAD = "semantic_vad"
    MODEL_NATIVE = "model_native"


class TurnDetection(EventBase):
    type: TurnDetectionType = TurnDetectionType.SERVER_VAD
    threshold: float | None = None
    prefix_padding_ms: int | None = None
    silence_duration_ms: int | None = None


class SessionConfig(EventBase):
    """``session.update`` payload. All fields optional — only set fields are applied."""

    modalities: list[str] | None = None
    instructions: str | None = None
    input_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"] | None = None
    turn_detection: TurnDetection | None = None
    temperature: float | None = None
    max_response_output_tokens: int | str | None = None
    output_audio_format: Literal["pcm16"] | None = None
    voice: str | dict[str, Any] | None = None


class SessionObject(EventBase):
    id: str
    object: Literal["realtime.session"] = "realtime.session"
    model: str
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    instructions: str = ""
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    voice: str | dict[str, Any] | None = None
    turn_detection: TurnDetection | None = None
    temperature: float = 0.8
    max_response_output_tokens: int | str = "inf"


class ClientEvent(EventBase):
    event_id: str | None = None
    type: str


class SessionUpdate(ClientEvent):
    type: Literal["session.update"]
    session: SessionConfig


class InputAudioBufferAppend(ClientEvent):
    type: Literal["input_audio_buffer.append"]
    audio: str  # base64-encoded raw PCM16 (or g711) per session.input_audio_format
    video_frames: list[str] | None = None
    force_listen: bool = False
    max_slice_nums: int = Field(default=1, ge=1, le=9)
    timestamp_ms: int | None = Field(default=None, ge=0)


class InputAudioBufferClear(ClientEvent):
    type: Literal["input_audio_buffer.clear"]


class ResponseCancel(ClientEvent):
    type: Literal["response.cancel"]


class PlaybackAck(ClientEvent):
    type: Literal["response.audio.playback_ack"]
    audio_end_ms: float = Field(ge=0)


class SessionClose(ClientEvent):
    type: Literal["session.close"]
    reason: str = Field(default="client_closed", min_length=1, max_length=128)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        reason = value.strip()
        if not reason:
            raise ValueError("session close reason must not be blank")
        return reason


def make_event(event_type: str, **fields: Any) -> dict[str, Any]:
    """Construct a server event dict. ``event_id`` is filled in by the
    session loop so handlers don't have to."""
    payload: dict[str, Any] = {"type": event_type}
    for k, v in fields.items():
        if v is None:
            continue
        payload[k] = v
    return payload


CLIENT_EVENT_TYPES: dict[str, type[ClientEvent]] = {
    "session.update": SessionUpdate,
    "input_audio_buffer.append": InputAudioBufferAppend,
    "input_audio_buffer.clear": InputAudioBufferClear,
    "response.cancel": ResponseCancel,
    "response.audio.playback_ack": PlaybackAck,
    "session.close": SessionClose,
}


def parse_client_event(raw: dict[str, Any]) -> ClientEvent | None:
    """Dispatch a raw client event dict to a typed model.

    Returns ``None`` when the ``type`` is unrecognized. A malformed
    payload that fails pydantic validation raises
    :class:`pydantic.ValidationError` — callers don't catch it.
    """
    event_type = raw.get("type")
    if not isinstance(event_type, str):
        return None

    cls = CLIENT_EVENT_TYPES.get(event_type)
    if cls is None:
        return None

    return cls.model_validate(raw)
