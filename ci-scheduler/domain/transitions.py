from __future__ import annotations

from .models import QueueStatus

VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"reserved", "cancelled", "stale", "skipped"}),
    "reserved": frozenset({"dispatched", "cancelled", "stale"}),
    "dispatched": frozenset({"running", "cancelled", "stale"}),
    "running": frozenset({"passed", "failed", "cancelled", "timed_out", "stale"}),
    "passed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "timed_out": frozenset(),
    "stale": frozenset(),
    "skipped": frozenset(),
}


class InvalidTransitionError(ValueError):
    pass


def can_transition(current: QueueStatus, target: QueueStatus) -> bool:
    return target in VALID_TRANSITIONS.get(current, frozenset())


def validate_transition(current: QueueStatus, target: QueueStatus) -> None:
    if not can_transition(current, target):
        raise InvalidTransitionError(f"invalid transition {current!r} -> {target!r}")
