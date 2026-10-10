# SPDX-License-Identifier: Apache-2.0
"""Wire schemas and state records for native duplex trace replay."""

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

Identifier = Annotated[str, Field(min_length=1)]
Nonnegative = Annotated[float, Field(ge=0)]


class TraceRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
    direction: Literal["send", "receive", "error", "admission"]
    time_s: Nonnegative
    event: dict[str, JsonValue]


class AdmissionEvent(BaseModel):
    """The only diagnostic record allowed before a session exists."""

    model_config = ConfigDict(strict=True, extra="forbid")
    type: Literal["connection_denied"]
    http_status: Literal[503]
    attempt: Annotated[int, Field(ge=1)]


class MediaTime(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    t_start_ms: Nonnegative
    duration_ms: Nonnegative


class Metadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    seq: Annotated[int, Field(ge=0)] | None = None
    t_start_ms: Nonnegative | None = None
    media_time: MediaTime | None = None
    fatal: bool | None = None


class StatusDetails(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    reason: Identifier


class Response(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    id: Identifier
    status: str
    status_details: StatusDetails | None = None


class WireEvent(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow", allow_inf_nan=False)
    type: Identifier
    event_id: Identifier
    sglang: Metadata = Field(default_factory=Metadata)
    session: dict[str, JsonValue] | None = None
    response: Response | None = None
    response_id: Identifier | None = None
    item_id: Identifier | None = None
    unit_id: Identifier | None = None
    content_index: int | None = None
    output_index: int | None = None
    audio: str | None = None
    delta: str | None = None
    tail_policy: str | None = None
    client_event_id: Identifier | None = None
    seq: int | None = None
    accepted_end_ms: Nonnegative | None = None
    consumed_ms: Nonnegative | None = None
    discarded_ms: Nonnegative | None = None
    padding_ms: Nonnegative | None = None
    reason: str | None = None
    error: dict[str, JsonValue] | None = None


@dataclass
class SentCommand:
    event: WireEvent
    time_s: float
    input_end_ms: float


@dataclass
class ResponseState:
    terminal: str | None = None
    reason: str | None = None
    audio_done: bool = False
    audio_seen: bool = False
    item_id: str | None = None
