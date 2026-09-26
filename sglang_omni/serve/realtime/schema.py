"""Shared realtime configuration and wire value types."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import TypedDict

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
SessionType = Literal["realtime", "transcription"]
SessionState = Literal["CREATED", "OPEN", "CLOSING", "CLOSED"]
MAX_EVENT_ID_LENGTH = 256
Interaction = Literal["native"]
TailPolicy = Literal["flush", "pad", "reject"]
PartialStyle = Literal["append_only", "revising"]


class AudioFormat(TypedDict):
    type: str
    rate: int


class TurnDetectionConfig(TypedDict, total=False):
    type: str
    threshold: float | None
    prefix_padding_ms: int | None
    silence_duration_ms: int | None
    eagerness: str | None
    interrupt_response: bool


class AudioInputConfig(TypedDict, total=False):
    format: AudioFormat
    turn_detection: TurnDetectionConfig | None


class AudioOutputConfig(TypedDict, total=False):
    format: AudioFormat


class AudioConfig(TypedDict, total=False):
    input: AudioInputConfig
    output: AudioOutputConfig


class TimebaseConfig(TypedDict, total=False):
    microturn_ms: Annotated[float, Field(gt=0)] | None
    native_unit_ms: int


class SessionExtension(TypedDict, total=False):
    interaction: Interaction
    tail_policy: TailPolicy
    timebase: TimebaseConfig


class SessionConfiguration(TypedDict, total=False):
    type: SessionType
    model: str
    instructions: str
    output_modalities: list[str]
    audio: AudioConfig
    sglang: SessionExtension


class Rejection(TypedDict):
    field: str
    requested: float | list[str]
    reason: str
    granted: list[str] | None


class GrantedCapabilities(TypedDict, total=False):
    interaction: Interaction
    native_full_duplex: bool
    proactive_output: bool
    turn_control: list[str | None]
    client_commit: bool
    input_modalities: list[str]
    output_modalities: list[str]
    input_audio_format: AudioFormat
    output_audio_format: AudioFormat
    native_unit_ms: int
    first_unit_ms: int
    microturn_ms: str | float | None
    tail_policy: TailPolicy
    supports_server_interrupt: bool
    supports_truncate: bool
    supports_resume: bool
    partial_style: PartialStyle
    pressure_policy: Literal["reject"]
    strict_order: bool
    limits: dict[str, int | float]
    rejections: list[Rejection]


class CapabilityResponse(GrantedCapabilities):
    model: str


class SessionUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    session: SessionConfiguration


class ClientEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    event_id: str = Field(min_length=1, max_length=MAX_EVENT_ID_LENGTH)


class SessionUpdateEvent(ClientEvent):
    type: Literal["session.update"]
    session: dict[str, object]


class AppendMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    seq: int
    t_start_ms: float | None = None


class AudioAppendEvent(ClientEvent):
    type: Literal["input_audio_buffer.append"]
    audio: str
    sglang: AppendMetadata


class SessionCommandEvent(ClientEvent):
    type: Literal[
        "input_audio_buffer.clear",
        "sglang.input_audio.end",
        "session.close",
        "input_audio_buffer.commit",
        "response.create",
    ]


CLIENT_EVENT: TypeAdapter[
    SessionUpdateEvent | AudioAppendEvent | SessionCommandEvent
] = TypeAdapter(
    Annotated[
        SessionUpdateEvent | AudioAppendEvent | SessionCommandEvent,
        Field(discriminator="type"),
    ]
)
