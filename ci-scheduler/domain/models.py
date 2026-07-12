from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

QueueStatus = Literal[
    "pending",
    "reserved",
    "dispatched",
    "running",
    "passed",
    "failed",
    "cancelled",
    "timed_out",
    "stale",
    "skipped",
]

RunnerClass = Literal["github-hosted", "self-hosted", "unsupported"]

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"passed", "failed", "cancelled", "timed_out", "stale", "skipped"}
)
ACTIVE_STATUSES: frozenset[str] = frozenset({"reserved", "dispatched", "running"})
CAPACITY_CONSUMING_STATUSES: frozenset[str] = frozenset({"reserved", "dispatched", "running"})

CONCLUSION_BY_STATUS: dict[str, str] = {
    "passed": "success",
    "failed": "failure",
    "cancelled": "cancelled",
    "timed_out": "timed_out",
    "stale": "cancelled",
    "skipped": "skipped",
}


@dataclass(frozen=True)
class DagNode:
    key: str
    root_workflow: str
    job_id: str
    source_path: str
    declaration_order: int
    needs: tuple[str, ...]
    is_virtual: bool
    is_executable: bool
    runner_class: RunnerClass
    check_name: str
    generated_workflow_path: str | None
    job_def: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class WorkflowGraph:
    root_workflow: str
    source_path: str
    source_hash: str
    nodes: tuple[DagNode, ...]
    node_by_key: dict[str, DagNode] = field(repr=False)

    def needs(self, key: str) -> tuple[str, ...]:
        return self.node_by_key[key].needs

    def executable_nodes(self) -> tuple[DagNode, ...]:
        return tuple(node for node in self.nodes if node.is_executable)

    def get(self, key: str) -> DagNode:
        return self.node_by_key[key]


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
    stage_key: str
    check_name: str
    check_run_id: int | None
    status: str
    attempt: int
    enqueue_time: str
    dispatch_time: str | None
    workflow_run_id: int | None
    dispatch_id: str | None
    generated_workflow_path: str | None
    source_hash: str | None
    conclusion: str | None
    installation_id: int | None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass(frozen=True)
class NeedState:
    result: str
    outputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SchedulerContext:
    source_event: dict[str, Any]
    needs: dict[str, NeedState]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_event": self.source_event,
            "needs": {
                key: {"result": state.result, "outputs": dict(state.outputs)}
                for key, state in self.needs.items()
            },
        }


@dataclass(frozen=True)
class StageOutputSchema:
    version: int
    outputs: dict[str, str]


@dataclass(frozen=True)
class DispatchSelection:
    item_id: int
    stage_key: str
    priority_rank: int
