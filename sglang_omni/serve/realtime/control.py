"""Typed runtime lifecycle and command results, projected by the transport."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Created:
    session_id: str
    model: str
    session_type: str = "realtime"


@dataclass(frozen=True)
class Updated:
    session_id: str
    model: str
    session_type: str
    granted: dict
    client_event_id: str
    config: dict | None = None


@dataclass(frozen=True)
class Accepted:
    seq: int
    accepted_end_ms: float
    client_event_id: str


@dataclass(frozen=True)
class Cleared:
    discarded_ms: float
    client_event_id: str


@dataclass(frozen=True)
class Ended:
    accepted_end_ms: float
    tail_policy: str
    client_event_id: str


@dataclass(frozen=True)
class Drained:
    accepted_end_ms: float
    consumed_ms: float
    discarded_ms: float
    padding_ms: float
    client_event_id: str


@dataclass(frozen=True)
class Cancelled:
    old_epoch: int
    epoch: int
    response_ids: tuple[str, ...]
    client_event_id: str | None


@dataclass(frozen=True)
class Closed:
    reason: str
    client_event_id: str | None = None


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    fatal: bool
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
    | Cancelled
    | Closed
    | Failure
    | UnitCompleted
)
