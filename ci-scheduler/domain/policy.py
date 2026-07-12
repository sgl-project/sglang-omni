from __future__ import annotations

from .models import (
    CAPACITY_CONSUMING_STATUSES,
    DagNode,
    PullRequestState,
    QueueItem,
    TERMINAL_STATUSES,
    WorkflowGraph,
)

ItemId = int


def is_ready(
    item: QueueItem,
    graph: WorkflowGraph,
    latest: dict[str, QueueItem],
) -> bool:
    for dependency in graph.needs(item.stage_key):
        dep_item = latest.get(dependency)
        if dep_item is None or not dep_item.is_terminal:
            return False
    return True


def _pr_eligible(
    item: QueueItem,
    pr_states: dict[tuple[str, int], PullRequestState],
    *,
    run_label: str,
) -> bool:
    pr = pr_states.get((item.repo, item.pr_number))
    if pr is None:
        return False
    return (
        pr.state == "open"
        and not pr.draft
        and pr.head_sha == item.head_sha
        and pr.has_label(run_label)
    )


def _priority_rank(item: QueueItem, pr_states: dict[tuple[str, int], PullRequestState], *, high_priority_label: str) -> int:
    pr = pr_states.get((item.repo, item.pr_number))
    if pr and pr.has_label(high_priority_label):
        return 0
    return 1


def select_next(
    pending: list[QueueItem],
    pr_states: dict[tuple[str, int], PullRequestState],
    latest_status: dict[str, QueueItem],
    graph: WorkflowGraph,
    *,
    run_label: str,
    high_priority_label: str,
    active_count: int,
    capacity: int,
    self_hosted_only: bool = True,
) -> tuple[ItemId, ...]:
    """Pure selection policy for self-hosted queue items."""

    slots = max(0, capacity - active_count)
    if slots <= 0:
        return ()

    node_by_key = graph.node_by_key
    candidates: list[tuple[tuple, ItemId]] = []

    for item in pending:
        if not _pr_eligible(item, pr_states, run_label=run_label):
            continue
        node = node_by_key.get(item.stage_key)
        if node is None:
            continue
        if self_hosted_only and node.runner_class != "self-hosted":
            continue
        if not is_ready(item, graph, latest_status):
            continue
        sort_key = (
            _priority_rank(item, pr_states, high_priority_label=high_priority_label),
            item.enqueue_time,
            item.pr_number,
            node.declaration_order,
            item.stage_key,
            item.id,
        )
        candidates.append((sort_key, item.id))

    candidates.sort(key=lambda entry: entry[0])
    return tuple(item_id for _, item_id in candidates[:slots])


def select_hosted_ready(
    pending: list[QueueItem],
    pr_states: dict[tuple[str, int], PullRequestState],
    latest_status: dict[str, QueueItem],
    graph: WorkflowGraph,
    *,
    run_label: str,
) -> tuple[ItemId, ...]:
    """Return all dependency-ready github-hosted items."""

    node_by_key = graph.node_by_key
    selected: list[tuple[tuple, ItemId]] = []
    for item in pending:
        if not _pr_eligible(item, pr_states, run_label=run_label):
            continue
        node = node_by_key.get(item.stage_key)
        if node is None or node.runner_class != "github-hosted":
            continue
        if not is_ready(item, graph, latest_status):
            continue
        sort_key = (node.declaration_order, item.stage_key, item.id)
        selected.append((sort_key, item.id))
    selected.sort(key=lambda entry: entry[0])
    return tuple(item_id for _, item_id in selected)


def latest_by_stage(items: list[QueueItem]) -> dict[str, QueueItem]:
    latest: dict[str, QueueItem] = {}
    for item in items:
        current = latest.get(item.stage_key)
        if current is None or item.attempt > current.attempt:
            latest[item.stage_key] = item
    return latest


def count_active(items: list[QueueItem], graph: WorkflowGraph) -> int:
    node_by_key = graph.node_by_key
    return sum(
        1
        for item in items
        if item.status in CAPACITY_CONSUMING_STATUSES
        and (node := node_by_key.get(item.stage_key)) is not None
        and node.runner_class == "self-hosted"
    )


def terminal_result(status: str) -> str:
    if status == "passed":
        return "success"
    if status in {"failed", "timed_out"}:
        return "failure"
    if status == "skipped":
        return "skipped"
    if status in {"cancelled", "stale"}:
        return "cancelled"
    return status
