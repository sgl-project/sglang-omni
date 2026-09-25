"""Typed runtime lifecycle and command results, projected by the transport."""

from dataclasses import dataclass

from sglang_omni.serve.realtime.schema import (
    GrantedCapabilities,
    SessionConfiguration,
    SessionType,
    TailPolicy,
)


@dataclass(frozen=True)
class Created:
    session_id: str
    model: str
    session_type: SessionType = "realtime"


@dataclass(frozen=True)
class Updated:
    session_id: str
    model: str
    session_type: SessionType
    granted: GrantedCapabilities
    client_event_id: str
    config: SessionConfiguration | None = None


@dataclass(frozen=True)
class Accepted:
    sequence: int
    accepted_end_ms: float
    client_event_id: str


@dataclass(frozen=True)
class Cleared:
    discarded_ms: float
    client_event_id: str


@dataclass(frozen=True)
class Ended:
    accepted_end_ms: float
    tail_policy: TailPolicy
    client_event_id: str


@dataclass(frozen=True)
class Drained:
    accepted_end_ms: float
    consumed_ms: float
    discarded_ms: float
    padding_ms: float
    client_event_id: str


@dataclass(frozen=True)
class Closed:
    reason: str
    client_event_id: str | None = None


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    is_fatal: bool
    client_event_id: str | None = None
    param: str | None = None


@dataclass(frozen=True)
class UnitCompleted:
    unit_id: str


ControlEvent = (
    Created
    | Updated
    | Accepted
    | Cleared
    | Ended
    | Drained
    | Closed
    | Failure
    | UnitCompleted
)
