"""Pure priority queue decisions for self-hosted CI stages.

This module intentionally has no GitHub API calls. The workflow-facing layer is
responsible for collecting PR/stage/run state; this module decides which single
stage, if any, may be dispatched next.
"""

from __future__ import annotations

from dataclasses import dataclass

RUNNING_STATUSES = frozenset({"in_progress"})
QUEUED_STATUSES = frozenset({"queued", "waiting", "pending", "requested"})
ACTIVE_STATUSES = RUNNING_STATUSES | QUEUED_STATUSES
FAILED_CONCLUSIONS = frozenset({"failure", "cancelled", "timed_out"})
RUN_LABEL = "run-ci"
HIGH_PRIORITY_LABEL = "high-priority"


@dataclass(frozen=True)
class PullRequestState:
    number: int
    labels: frozenset[str]
    draft: bool
    head_sha: str
    opened_at: str


@dataclass(frozen=True)
class StageSpec:
    id: str
    workflow: str
    name: str
    order: int
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageRunState:
    pr_number: int
    head_sha: str
    stage_id: str
    status: str
    conclusion: str | None
    created_at: str
    rerun_request_id: int | None = None
    run_id: int | None = None


@dataclass(frozen=True)
class RerunRequest:
    pr_number: int
    request_id: int
    created_at: str
    stage_ids: frozenset[str]


@dataclass(frozen=True)
class StageCandidate:
    pr: PullRequestState
    stage: StageSpec
    rerun_request_id: int | None = None
    queued_at: str = ""


@dataclass(frozen=True)
class QueueDecision:
    selected_stage: StageCandidate | None
    active_stage: StageRunState | None = None
    preempt_stage: StageRunState | None = None


def ci_enabled(
    pr: PullRequestState,
    *,
    run_label: str = RUN_LABEL,
) -> bool:
    return run_label in pr.labels and not pr.draft


def high_priority(
    pr: PullRequestState,
    *,
    high_priority_label: str = HIGH_PRIORITY_LABEL,
) -> bool:
    return high_priority_label in pr.labels


def select_next_stage(
    *,
    pull_requests: list[PullRequestState],
    stages: list[StageSpec],
    runs: list[StageRunState],
    rerun_requests: list[RerunRequest] | None = None,
    run_label: str = RUN_LABEL,
    high_priority_label: str = HIGH_PRIORITY_LABEL,
) -> QueueDecision:
    """Return the one stage that may start next.

    The scheduler has exactly one global stage slot. If any stage run is already
    active, no new stage is selected. Otherwise, high-priority PR stages are
    selected before normal-priority PR stages. Priority is evaluated from the
    current PR labels, so labels added while a stage is still queued can promote
    that stage before it is dispatched.
    """

    running_stage = _stage_with_status(runs, RUNNING_STATUSES)
    if running_stage is not None:
        return QueueDecision(selected_stage=None, active_stage=running_stage)

    stages_by_id = {stage.id: stage for stage in stages}
    rerun_requests = rerun_requests or []
    candidates: list[StageCandidate] = []

    for pr in sorted(pull_requests, key=lambda item: (item.opened_at, item.number)):
        if not ci_enabled(pr, run_label=run_label):
            continue

        candidates.extend(
            _rerun_candidates_for_pr(
                pr=pr,
                stages_by_id=stages_by_id,
                runs=runs,
                rerun_requests=rerun_requests,
            )
        )
        candidates.extend(_initial_candidates_for_pr(pr, stages, runs))

    queued_stage = _stage_with_status(runs, QUEUED_STATUSES)
    if not candidates:
        if queued_stage is not None and not _queued_stage_is_still_valid(
            queued_stage,
            pull_requests,
        ):
            return QueueDecision(selected_stage=None, preempt_stage=queued_stage)
        return QueueDecision(selected_stage=None)

    selected = min(
        candidates,
        key=lambda candidate: _candidate_sort_key(
            candidate,
            high_priority_label=high_priority_label,
        ),
    )
    if queued_stage is not None:
        queued_candidate = _candidate_for_queued_stage(
            queued_stage,
            pull_requests,
            stages_by_id,
        )
        if queued_candidate is None:
            return QueueDecision(
                selected_stage=selected,
                preempt_stage=queued_stage,
            )
        if _candidate_sort_key(
            selected,
            high_priority_label=high_priority_label,
        ) < _candidate_sort_key(
            queued_candidate,
            high_priority_label=high_priority_label,
        ):
            return QueueDecision(
                selected_stage=selected,
                preempt_stage=queued_stage,
            )
        return QueueDecision(selected_stage=None, active_stage=queued_stage)

    return QueueDecision(selected_stage=selected)


def _stage_with_status(
    runs: list[StageRunState],
    statuses: frozenset[str],
) -> StageRunState | None:
    matches = [run for run in runs if run.status in statuses]
    if not matches:
        return None
    return min(matches, key=lambda run: (run.created_at, run.pr_number, run.stage_id))


def _initial_candidates_for_pr(
    pr: PullRequestState,
    stages: list[StageSpec],
    runs: list[StageRunState],
) -> list[StageCandidate]:
    candidates: list[StageCandidate] = []
    for stage in sorted(stages, key=lambda item: (item.order, item.id)):
        if not _dependencies_succeeded(pr, stage, runs):
            continue

        latest = _latest_run(pr, stage.id, runs)
        if latest is None:
            candidates.append(
                StageCandidate(pr=pr, stage=stage, queued_at=pr.opened_at)
            )
            continue

        if latest.status == "completed" and latest.conclusion == "success":
            continue
        if latest.status == "completed" and latest.conclusion == "cancelled":
            candidates.append(
                StageCandidate(pr=pr, stage=stage, queued_at=latest.created_at)
            )
            continue
        # Failed/timed-out stages are not retried implicitly. Cancelled stages
        # are allowed back in because low-priority queued runs can be cancelled
        # by the dispatcher when a higher-priority stage arrives.
    return candidates


def _rerun_candidates_for_pr(
    *,
    pr: PullRequestState,
    stages_by_id: dict[str, StageSpec],
    runs: list[StageRunState],
    rerun_requests: list[RerunRequest],
) -> list[StageCandidate]:
    candidates: list[StageCandidate] = []
    for request in sorted(
        (request for request in rerun_requests if request.pr_number == pr.number),
        key=lambda item: (item.created_at, item.request_id),
    ):
        for stage_id in sorted(
            request.stage_ids,
            key=lambda item: (
                stages_by_id[item].order if item in stages_by_id else 10**9,
                item,
            ),
        ):
            stage = stages_by_id.get(stage_id)
            if stage is None:
                continue
            if _has_attempt_for_rerun_request(pr, stage.id, runs, request):
                continue
            if not _dependencies_succeeded(pr, stage, runs):
                continue
            failed_run = _latest_run_before_or_at(
                pr, stage.id, runs, request.created_at
            )
            if (
                failed_run is None
                or failed_run.status != "completed"
                or failed_run.conclusion not in FAILED_CONCLUSIONS
            ):
                continue
            candidates.append(
                StageCandidate(
                    pr=pr,
                    stage=stage,
                    rerun_request_id=request.request_id,
                    queued_at=request.created_at,
                )
            )
    return candidates


def _dependencies_succeeded(
    pr: PullRequestState,
    stage: StageSpec,
    runs: list[StageRunState],
) -> bool:
    for dependency_id in stage.depends_on:
        latest = _latest_run(pr, dependency_id, runs)
        if (
            latest is None
            or latest.status != "completed"
            or latest.conclusion != "success"
        ):
            return False
    return True


def _latest_run(
    pr: PullRequestState,
    stage_id: str,
    runs: list[StageRunState],
) -> StageRunState | None:
    matches = [
        run
        for run in runs
        if run.pr_number == pr.number
        and run.head_sha == pr.head_sha
        and run.stage_id == stage_id
    ]
    if not matches:
        return None
    return max(matches, key=lambda run: (run.created_at, run.rerun_request_id or -1))


def _latest_run_before_or_at(
    pr: PullRequestState,
    stage_id: str,
    runs: list[StageRunState],
    timestamp: str,
) -> StageRunState | None:
    matches = [
        run
        for run in runs
        if run.pr_number == pr.number
        and run.head_sha == pr.head_sha
        and run.stage_id == stage_id
        and run.created_at <= timestamp
    ]
    if not matches:
        return None
    return max(matches, key=lambda run: (run.created_at, run.rerun_request_id or -1))


def _has_attempt_for_rerun_request(
    pr: PullRequestState,
    stage_id: str,
    runs: list[StageRunState],
    request: RerunRequest,
) -> bool:
    return any(
        run.pr_number == pr.number
        and run.head_sha == pr.head_sha
        and run.stage_id == stage_id
        and run.rerun_request_id == request.request_id
        for run in runs
    )


def _candidate_sort_key(
    candidate: StageCandidate,
    *,
    high_priority_label: str,
) -> tuple[int, str, int, int, str, int, int]:
    priority = (
        0 if high_priority(candidate.pr, high_priority_label=high_priority_label) else 1
    )
    return (
        priority,
        candidate.pr.opened_at,
        candidate.pr.number,
        candidate.stage.order,
        candidate.stage.id,
        0 if candidate.rerun_request_id is not None else 1,
        candidate.rerun_request_id or -1,
    )


def _queued_stage_is_still_valid(
    queued_stage: StageRunState,
    pull_requests: list[PullRequestState],
) -> bool:
    return any(
        pr.number == queued_stage.pr_number
        and pr.head_sha == queued_stage.head_sha
        and ci_enabled(pr)
        for pr in pull_requests
    )


def _candidate_for_queued_stage(
    queued_stage: StageRunState,
    pull_requests: list[PullRequestState],
    stages_by_id: dict[str, StageSpec],
) -> StageCandidate | None:
    pr = next(
        (
            pr
            for pr in pull_requests
            if pr.number == queued_stage.pr_number
            and pr.head_sha == queued_stage.head_sha
            and ci_enabled(pr)
        ),
        None,
    )
    stage = stages_by_id.get(queued_stage.stage_id)
    if pr is None or stage is None:
        return None
    return StageCandidate(
        pr=pr,
        stage=stage,
        rerun_request_id=queued_stage.rerun_request_id,
        queued_at=queued_stage.created_at,
    )
