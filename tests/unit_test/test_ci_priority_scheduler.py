from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_scheduler_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / ".github" / "scripts" / "ci_priority_scheduler.py"
    spec = importlib.util.spec_from_file_location("ci_priority_scheduler", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scheduler = _load_scheduler_module()


def pr(
    number: int,
    *,
    labels: set[str] | None = None,
    draft: bool = False,
    head_sha: str | None = None,
    opened_at: str | None = None,
):
    return scheduler.PullRequestState(
        number=number,
        labels=frozenset(labels or {"run-ci"}),
        draft=draft,
        head_sha=head_sha or f"sha-{number}",
        opened_at=opened_at or f"2026-05-24T00:{number:02d}:00Z",
    )


def stage(
    stage_id: str,
    *,
    workflow: str = "qwen3",
    order: int = 0,
    depends_on: tuple[str, ...] = (),
):
    return scheduler.StageSpec(
        id=stage_id,
        workflow=workflow,
        name=stage_id,
        order=order,
        depends_on=depends_on,
    )


def run(
    pr_number: int,
    stage_id: str,
    *,
    head_sha: str | None = None,
    status: str = "completed",
    conclusion: str | None = "success",
    created_at: str = "2026-05-24T01:00:00Z",
    rerun_request_id: int | None = None,
):
    return scheduler.StageRunState(
        pr_number=pr_number,
        head_sha=head_sha or f"sha-{pr_number}",
        stage_id=stage_id,
        status=status,
        conclusion=conclusion,
        created_at=created_at,
        rerun_request_id=rerun_request_id,
    )


def decision(prs, stages, runs=(), rerun_requests=()):
    return scheduler.select_next_stage(
        pull_requests=list(prs),
        stages=list(stages),
        runs=list(runs),
        rerun_requests=list(rerun_requests),
    )


def test_run_ci_label_is_required_and_draft_prs_are_not_queued():
    stages = [stage("stage-1")]

    assert decision([pr(1, labels={"high-priority"})], stages).selected_stage is None
    assert (
        decision(
            [pr(2, labels={"run-ci", "high-priority"}, draft=True)], stages
        ).selected_stage
        is None
    )
    assert (
        decision([pr(3, labels={"run-ci"})], stages).selected_stage.stage.id
        == "stage-1"
    )


def test_only_one_stage_is_released_and_active_stage_blocks_the_queue():
    stages = [stage("stage-1", order=1), stage("stage-2", order=2)]

    selected = decision([pr(1), pr(2)], stages).selected_stage
    assert (selected.pr.number, selected.stage.id) == (1, "stage-1")

    active = run(1, "stage-1", status="in_progress", conclusion=None)
    blocked = decision([pr(1), pr(2)], stages, runs=[active])

    assert blocked.selected_stage is None
    assert blocked.active_stage.pr_number == 1
    assert blocked.active_stage.stage_id == "stage-1"


def test_high_priority_stages_are_released_before_older_low_priority_stages():
    stages = [stage("stage-1", order=1), stage("stage-2", order=2)]
    low = pr(1, labels={"run-ci"}, opened_at="2026-05-24T00:01:00Z")
    high = pr(
        2,
        labels={"run-ci", "high-priority"},
        opened_at="2026-05-24T00:02:00Z",
    )

    selected = decision([low, high], stages).selected_stage

    assert (selected.pr.number, selected.stage.id) == (2, "stage-1")


def test_late_high_priority_label_promotes_queued_but_unreleased_stages():
    stages = [stage("stage-1", order=1), stage("stage-2", order=2)]
    b = pr(2, labels={"run-ci"}, opened_at="2026-05-24T00:02:00Z")
    c = pr(
        3,
        labels={"run-ci", "high-priority"},
        opened_at="2026-05-24T00:03:00Z",
    )
    completed_b_stage_1 = run(2, "stage-1", created_at="2026-05-24T01:00:00Z")

    selected = decision([b, c], stages, runs=[completed_b_stage_1]).selected_stage

    assert (selected.pr.number, selected.stage.id) == (3, "stage-1")


def test_running_low_priority_stage_is_not_preempted_by_new_high_priority_pr():
    stages = [stage("stage-1", order=1), stage("stage-2", order=2)]
    b = pr(2, labels={"run-ci"})
    c = pr(3, labels={"run-ci", "high-priority"})
    running_b = run(2, "stage-1", status="in_progress", conclusion=None)

    blocked = decision([b, c], stages, runs=[running_b])

    assert blocked.selected_stage is None
    assert blocked.active_stage.pr_number == 2

    completed_b = run(2, "stage-1", created_at="2026-05-24T01:00:00Z")
    selected = decision([b, c], stages, runs=[completed_b]).selected_stage

    assert (selected.pr.number, selected.stage.id) == (3, "stage-1")


def test_queued_low_priority_stage_is_preempted_by_new_high_priority_pr():
    stages = [stage("stage-1", order=1), stage("stage-2", order=2)]
    b = pr(2, labels={"run-ci"})
    c = pr(3, labels={"run-ci", "high-priority"})
    queued_b = run(2, "stage-1", status="queued", conclusion=None)

    selected = decision([b, c], stages, runs=[queued_b])

    assert selected.preempt_stage.pr_number == 2
    assert (selected.selected_stage.pr.number, selected.selected_stage.stage.id) == (
        3,
        "stage-1",
    )


def test_queued_stage_blocks_peers_when_no_higher_priority_stage_exists():
    stages = [stage("stage-1", order=1), stage("stage-2", order=2)]
    queued = run(1, "stage-1", status="queued", conclusion=None)

    blocked = decision([pr(1), pr(2)], stages, runs=[queued])

    assert blocked.selected_stage is None
    assert blocked.active_stage.stage_id == "stage-1"


def test_rerun_failed_ci_requeues_failed_stages_once_with_current_priority():
    stages = [
        stage("stage-1", order=1),
        stage("stage-2", order=2),
        stage("stage-3", order=3),
    ]
    high = pr(1, labels={"run-ci", "high-priority"})
    low = pr(2, labels={"run-ci"})
    request = scheduler.RerunRequest(
        pr_number=1,
        request_id=9001,
        created_at="2026-05-24T02:00:00Z",
        stage_ids=frozenset({"stage-2", "stage-3"}),
    )
    previous_runs = [
        run(1, "stage-1", conclusion="success", created_at="2026-05-24T01:01:00Z"),
        run(1, "stage-2", conclusion="failure", created_at="2026-05-24T01:02:00Z"),
        run(1, "stage-3", conclusion="cancelled", created_at="2026-05-24T01:03:00Z"),
    ]

    first = decision([high, low], stages, runs=previous_runs, rerun_requests=[request])
    assert (first.selected_stage.pr.number, first.selected_stage.stage.id) == (
        1,
        "stage-2",
    )
    assert first.selected_stage.rerun_request_id == 9001

    attempted_stage_2 = run(
        1,
        "stage-2",
        conclusion="failure",
        created_at="2026-05-24T02:05:00Z",
        rerun_request_id=9001,
    )
    second = decision(
        [high, low],
        stages,
        runs=[*previous_runs, attempted_stage_2],
        rerun_requests=[request],
    )
    assert (second.selected_stage.pr.number, second.selected_stage.stage.id) == (
        1,
        "stage-3",
    )

    attempted_stage_3 = run(
        1,
        "stage-3",
        conclusion="failure",
        created_at="2026-05-24T02:06:00Z",
        rerun_request_id=9001,
    )
    after_request_consumed = decision(
        [high, low],
        stages,
        runs=[*previous_runs, attempted_stage_2, attempted_stage_3],
        rerun_requests=[request],
    )
    assert (
        after_request_consumed.selected_stage.pr.number,
        after_request_consumed.selected_stage.stage.id,
    ) == (
        2,
        "stage-1",
    )


def test_successful_stages_are_not_queued_again_without_a_rerun_request():
    stages = [stage("stage-1", order=1), stage("stage-2", order=2)]
    runs = [run(1, "stage-1"), run(1, "stage-2")]

    assert decision([pr(1)], stages, runs=runs).selected_stage is None


def test_failed_stage_is_not_retried_without_rerun_request_but_independent_stages_continue():
    stages = [stage("stage-1", order=1), stage("stage-2", order=2)]
    failed = run(1, "stage-1", conclusion="failure")

    selected = decision([pr(1)], stages, runs=[failed]).selected_stage

    assert (selected.pr.number, selected.stage.id) == (1, "stage-2")


def test_stage_dependencies_must_succeed_before_dependent_stages_are_queued():
    stages = [
        stage("docs", workflow="s2pro", order=1),
        stage("non-streaming", workflow="s2pro", order=2, depends_on=("docs",)),
        stage("streaming", workflow="s2pro", order=3, depends_on=("docs",)),
        stage(
            "consistency",
            workflow="s2pro",
            order=4,
            depends_on=("non-streaming", "streaming"),
        ),
    ]
    pull_request = pr(1)

    assert decision([pull_request], stages).selected_stage.stage.id == "docs"

    docs_done = run(1, "docs")
    assert (
        decision([pull_request], stages, runs=[docs_done]).selected_stage.stage.id
        == "non-streaming"
    )

    non_streaming_done = run(1, "non-streaming")
    assert (
        decision(
            [pull_request],
            stages,
            runs=[docs_done, non_streaming_done],
        ).selected_stage.stage.id
        == "streaming"
    )

    streaming_done = run(1, "streaming")
    assert (
        decision(
            [pull_request],
            stages,
            runs=[docs_done, non_streaming_done, streaming_done],
        ).selected_stage.stage.id
        == "consistency"
    )
