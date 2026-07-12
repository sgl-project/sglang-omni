from __future__ import annotations

import pytest

from domain.models import DagNode, PullRequestState, QueueItem, WorkflowGraph
from domain.policy import is_ready, select_next
from domain.transitions import InvalidTransitionError, can_transition, validate_transition


REPO = "sgl-project/sglang-omni"


def make_node(
    key: str,
    order: int,
    needs: tuple[str, ...] = (),
    runner: str = "self-hosted",
) -> DagNode:
    return DagNode(
        key=key,
        root_workflow="test",
        job_id=key.split("::")[-1],
        source_path="omni-ci.yaml",
        declaration_order=order,
        needs=needs,
        is_virtual=False,
        is_executable=True,
        runner_class=runner,  # type: ignore[arg-type]
        check_name=f"CI / {key}",
        generated_workflow_path=f".github/workflows/generated/zz_generated_scheduler__{key}.yaml",
        job_def={},
    )


def make_graph(nodes: list[DagNode]) -> WorkflowGraph:
    return WorkflowGraph(
        root_workflow="test",
        source_path="omni-ci.yaml",
        source_hash="abc",
        nodes=tuple(nodes),
        node_by_key={node.key: node for node in nodes},
    )


def item(
    item_id: int,
    stage_key: str,
    *,
    pr_number: int = 1,
    sha: str = "sha-1",
    status: str = "pending",
    enqueue_time: str = "2026-01-01T00:00:00+00:00",
) -> QueueItem:
    return QueueItem(
        id=item_id,
        repo=REPO,
        pr_number=pr_number,
        head_sha=sha,
        stage_key=stage_key,
        check_name=f"CI / {stage_key}",
        check_run_id=None,
        status=status,
        attempt=1,
        enqueue_time=enqueue_time,
        dispatch_time=None,
        workflow_run_id=None,
        dispatch_id=None,
        generated_workflow_path=None,
        source_hash="abc",
        conclusion=None,
        installation_id=1,
    )


def pr(number: int, sha: str, *, labels: set[str] | None = None) -> PullRequestState:
    return PullRequestState(
        repo=REPO,
        pr_number=number,
        head_sha=sha,
        state="open",
        draft=False,
        labels=frozenset({"run-ci"} if labels is None else labels),
        installation_id=1,
    )


def test_high_priority_sorts_before_normal() -> None:
    graph = make_graph([make_node("test::setup", 100)])
    pending = [
        item(1, "test::setup", pr_number=10, sha="sha-10", enqueue_time="2026-01-01T01:00:00+00:00"),
        item(2, "test::setup", pr_number=20, sha="sha-20", enqueue_time="2026-01-01T02:00:00+00:00"),
    ]
    pr_states = {
        (REPO, 10): pr(10, "sha-10"),
        (REPO, 20): pr(20, "sha-20", labels={"run-ci", "high-priority"}),
    }
    selected = select_next(
        pending,
        pr_states,
        {},
        graph,
        run_label="run-ci",
        high_priority_label="high-priority",
        active_count=0,
        capacity=1,
    )
    assert selected == (2,)


def test_capacity_one_returns_single_item() -> None:
    graph = make_graph(
        [
            make_node("test::setup", 100),
            make_node("test::stage-a", 200, ("test::setup",)),
        ]
    )
    pending = [
        item(1, "test::setup"),
        item(2, "test::stage-a"),
    ]
    pr_states = {(REPO, 1): pr(1, "sha-1")}
    selected = select_next(
        pending,
        pr_states,
        {"test::setup": item(1, "test::setup", status="passed")},
        graph,
        run_label="run-ci",
        high_priority_label="high-priority",
        active_count=0,
        capacity=1,
    )
    assert len(selected) == 1


def test_is_ready_requires_terminal_dependencies() -> None:
    graph = make_graph([make_node("test::stage-a", 200, ("test::setup",))])
    queue_item = item(2, "test::stage-a")
    assert is_ready(queue_item, graph, {}) is False
    assert is_ready(
        queue_item,
        graph,
        {"test::setup": item(1, "test::setup", status="passed")},
    )


def test_valid_and_invalid_transitions() -> None:
    assert can_transition("pending", "reserved")
    assert can_transition("running", "passed")
    assert not can_transition("passed", "pending")
    with pytest.raises(InvalidTransitionError):
        validate_transition("passed", "pending")
