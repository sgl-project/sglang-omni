from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


QueueStatus = Literal[
    "pending",
    "dispatched",
    "running",
    "passed",
    "failed",
    "cancelled",
    "timed_out",
    "stale",
    "skipped",
]

TerminalStatus = {"passed", "failed", "cancelled", "timed_out", "stale", "skipped"}
ActiveStatus = {"dispatched", "running"}


@dataclass(frozen=True)
class Stage:
    stage_id: str
    check_name: str
    order: int
    depends_on: tuple[str, ...]
    capacity_group: str
    workflow_id: str
    workflow_ref: str
    timeout_minutes: int
    commands: tuple[str, ...]


@dataclass(frozen=True)
class PullRequestState:
    repo: str
    pr_number: int
    head_sha: str
    state: str
    draft: bool
    labels: frozenset[str]
    installation_id: int | None

    def has_label(self, label: str) -> bool:
        return label in self.labels


@dataclass(frozen=True)
class QueueItem:
    id: int
    repo: str
    pr_number: int
    head_sha: str
    stage_id: str
    check_name: str
    check_run_id: int | None
    status: str
    attempt: int
    enqueue_time: str
    dispatch_time: str | None
    workflow_run_id: int | None
    conclusion: str | None
    installation_id: int | None


@dataclass(frozen=True)
class DispatchCandidate:
    item: QueueItem
    stage: Stage
    priority_rank: int
