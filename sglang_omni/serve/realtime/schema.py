"""Shared realtime configuration and wire value types."""

from __future__ import annotations

import base64
import binascii
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator
from typing_extensions import TypedDict

from sglang_omni.serve.realtime.reference_audio import (
    MAX_REFERENCE_AUDIO_BYTES,
    normalize_reference_wav,
)

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


class ImageFormat(TypedDict):
    types: list[str]
    max_bytes: int
    max_frames_per_unit: int
    max_slice_nums: int


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


class SamplingConfig(TypedDict, total=False):
    temperature: Annotated[float, Field(ge=0, le=2)]
    top_k: Annotated[int, Field(ge=-1)]
    top_p: Annotated[float, Field(gt=0, le=1)]
    repetition_penalty: Annotated[float, Field(ge=1)]
    listen_prob_scale: Annotated[float, Field(ge=0)]
    greedy: bool
    force_listen_count: Annotated[int, Field(ge=0)]
    max_new_tokens_per_unit: Annotated[int, Field(ge=1)]
    repetition_window_size: Annotated[int, Field(ge=1)]
    talker_temperature: Annotated[float, Field(ge=0, le=2)]
    talker_repetition_penalty: Annotated[float, Field(ge=1)]


class AudioReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    media_type: Literal["audio/wav"]
    data: bytes = Field(repr=False)

    @field_validator("data", mode="before", json_schema_input_type=str)
    @classmethod
    def decode_wav(cls, value: JsonValue) -> bytes:
        if not isinstance(value, str):
            raise ValueError("reference audio must be base64 text")
        elif len(value) > (MAX_REFERENCE_AUDIO_BYTES + 2) // 3 * 4:
            raise ValueError("reference audio exceeds byte limit")
        else:
            pass
        try:
            audio = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("invalid base64 reference audio") from exc
        if len(audio) > MAX_REFERENCE_AUDIO_BYTES:
            raise ValueError("reference audio exceeds byte limit")
        else:
            return normalize_reference_wav(audio)


class SessionExtension(TypedDict, total=False):
    interaction: Interaction
    tail_policy: TailPolicy
    timebase: TimebaseConfig
    sampling: SamplingConfig
    reference_audio: AudioReference
    tts_reference_audio: AudioReference
    max_slice_nums: Annotated[int, Field(ge=1)]


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
    input_image_format: ImageFormat
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
    sampling_parameters: list[str]
    supports_reference_audio: bool
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


class ImageAppendMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    t_ms: float


class ImageAppendEvent(ClientEvent):
    type: Literal["sglang.input_image.append"]
    image: str
    sglang: ImageAppendMetadata


class SessionCommandEvent(ClientEvent):
    type: Literal[
        "input_audio_buffer.clear",
        "sglang.input_audio.end",
        "session.close",
        "input_audio_buffer.commit",
        "response.create",
    ]


CLIENT_EVENT: TypeAdapter[
    SessionUpdateEvent | AudioAppendEvent | ImageAppendEvent | SessionCommandEvent
] = TypeAdapter(
    Annotated[
        SessionUpdateEvent | AudioAppendEvent | ImageAppendEvent | SessionCommandEvent,
        Field(discriminator="type"),
    ]
)
