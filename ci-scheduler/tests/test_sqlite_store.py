from __future__ import annotations

import threading
from pathlib import Path

import pytest

from adapters.sqlite_store import SqliteQueueStore
from domain.models import DagNode, PullRequestState, WorkflowGraph


REPO = "sgl-project/sglang-omni"


def make_node(key: str, order: int, needs: tuple[str, ...] = ()) -> DagNode:
    return DagNode(
        key=key,
        root_workflow="test",
        job_id=key,
        source_path="test.yaml",
        declaration_order=order,
        needs=needs,
        is_virtual=False,
        is_executable=True,
        runner_class="self-hosted",
        check_name=f"CI / {key}",
        generated_workflow_path=f".github/workflows/generated/zz_{key}.yaml",
        job_def={},
    )


@pytest.fixture
def graph() -> WorkflowGraph:
    return WorkflowGraph(
        root_workflow="test",
        source_path="test.yaml",
        source_hash="hash",
        nodes=(
            make_node("setup", 100),
            make_node("stage-a", 200, ("setup",)),
            make_node("stage-b", 300, ("setup",)),
        ),
        node_by_key={
            "setup": make_node("setup", 100),
            "stage-a": make_node("stage-a", 200, ("setup",)),
            "stage-b": make_node("stage-b", 300, ("setup",)),
        },
    )


@pytest.fixture
def store(tmp_path: Path) -> SqliteQueueStore:
    return SqliteQueueStore(tmp_path / "queue.sqlite3")


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


def test_reserve_next_respects_capacity_one(store: SqliteQueueStore, graph: WorkflowGraph) -> None:
    store.enqueue_pr(pr(1, "sha-1"), graph, run_label="run-ci")
    first = store.reserve_next(
        graph,
        run_label="run-ci",
        high_priority_label="high-priority",
        capacity=1,
    )
    second = store.reserve_next(
        graph,
        run_label="run-ci",
        high_priority_label="high-priority",
        capacity=1,
    )
    assert first is not None
    assert first.stage_key == "setup"
    assert second is None


def test_concurrent_reserve_only_one_winner(store: SqliteQueueStore, graph: WorkflowGraph) -> None:
    store.enqueue_pr(pr(1, "sha-1"), graph, run_label="run-ci")
    results: list[int | None] = []
    lock = threading.Lock()

    def worker() -> None:
        reserved = store.reserve_next(
            graph,
            run_label="run-ci",
            high_priority_label="high-priority",
            capacity=1,
        )
        with lock:
            results.append(reserved.id if reserved else None)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    winners = [item_id for item_id in results if item_id is not None]
    assert len(winners) == 1


def test_stale_sha_marks_old_items_stale(store: SqliteQueueStore, graph: WorkflowGraph) -> None:
    store.enqueue_pr(pr(1, "old-sha"), graph, run_label="run-ci")
    reserved = store.reserve_next(
        graph,
        run_label="run-ci",
        high_priority_label="high-priority",
        capacity=1,
    )
    assert reserved is not None
    store.enqueue_pr(pr(1, "new-sha"), graph, run_label="run-ci")
    old_items = store.list_items_for_pr(REPO, 1)
    old = next(item for item in old_items if item.id == reserved.id)
    assert old.status == "stale"


def test_high_priority_pr_dispatches_first(store: SqliteQueueStore, graph: WorkflowGraph) -> None:
    store.enqueue_pr(pr(10, "sha-10"), graph, run_label="run-ci")
    store.enqueue_pr(pr(20, "sha-20", labels={"run-ci", "high-priority"}), graph, run_label="run-ci")
    reserved = store.reserve_next(
        graph,
        run_label="run-ci",
        high_priority_label="high-priority",
        capacity=1,
    )
    assert reserved is not None
    assert reserved.pr_number == 20


def test_draft_prs_do_not_participate(store: SqliteQueueStore, graph: WorkflowGraph) -> None:
    draft = PullRequestState(
        repo=REPO,
        pr_number=1,
        head_sha="sha-1",
        state="open",
        draft=True,
        labels=frozenset({"run-ci"}),
        installation_id=1,
    )
    items = store.enqueue_pr(draft, graph, run_label="run-ci")
    assert items == []
    assert store.reserve_next(
        graph,
        run_label="run-ci",
        high_priority_label="high-priority",
        capacity=1,
    ) is None


def test_prs_without_run_ci_do_not_participate(store: SqliteQueueStore, graph: WorkflowGraph) -> None:
    items = store.enqueue_pr(pr(1, "sha-1", labels=set()), graph, run_label="run-ci")
    assert items == []
    assert store.reserve_next(
        graph,
        run_label="run-ci",
        high_priority_label="high-priority",
        capacity=1,
    ) is None
