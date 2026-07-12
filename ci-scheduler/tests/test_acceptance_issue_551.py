"""Acceptance tests mapping issue #551 requirements."""

from __future__ import annotations

from pathlib import Path

from adapters.sqlite_store import SqliteQueueStore
from compiler.graph import compile_workflow
from compiler.generator import check_generated
from domain.transitions import VALID_TRANSITIONS


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
GENERATED_DIR = WORKFLOWS_DIR / "generated"


def test_issue_551_capacity_one_invariant() -> None:
    """Only one self-hosted stage may be reserved at a time."""

    store = SqliteQueueStore(Path("test-capacity.sqlite3"))
    try:
        graph = compile_workflow(WORKFLOWS_DIR / "test-asr-ci.yaml", workflows_dir=WORKFLOWS_DIR)
        from domain.models import PullRequestState

        pr = PullRequestState(
            repo="sgl-project/sglang-omni",
            pr_number=1,
            head_sha="sha",
            state="open",
            draft=False,
            labels=frozenset({"run-ci"}),
            installation_id=1,
        )
        store.enqueue_pr(pr, graph, run_label="run-ci")
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
        assert second is None
    finally:
        Path("test-capacity.sqlite3").unlink(missing_ok=True)


def test_issue_551_generated_workflow_one_job_per_file() -> None:
    workflows = list(GENERATED_DIR.glob("zz_generated_scheduler__*.yaml"))
    assert workflows
    for path in workflows[:5]:
        text = path.read_text(encoding="utf-8")
        assert "run-stage:" in text
        assert text.count("\njobs:\n") == 1


def test_issue_551_dag_from_yaml_not_stages_json() -> None:
    assert not (REPO_ROOT / "ci-scheduler" / "stages.json").exists()
    assert not (REPO_ROOT / "ci-scheduler" / "scheduler_app").exists()
    assert not (REPO_ROOT / ".github" / "scheduler-workflows").exists()


def test_issue_551_state_machine_documented_transitions() -> None:
    assert "pending" in VALID_TRANSITIONS
    assert "reserved" in VALID_TRANSITIONS["pending"]
    assert "running" in VALID_TRANSITIONS["dispatched"]


def test_issue_551_workflow_gen_check_passes() -> None:
    graph = compile_workflow(WORKFLOWS_DIR / "omni-ci.yaml", workflows_dir=WORKFLOWS_DIR)
    errors = check_generated(graph, GENERATED_DIR)
    assert errors == []
