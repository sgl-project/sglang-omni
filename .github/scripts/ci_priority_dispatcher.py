"""Dispatch exactly one priority CI stage when the global queue is free."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ci_priority_scheduler import (
    FAILED_CONCLUSIONS,
    QUEUED_STATUSES,
    RUNNING_STATUSES,
    PullRequestState,
    RerunRequest,
    StageRunState,
    ci_enabled,
    select_next_stage,
)
from ci_priority_stage_config import dispatch_config_for_stage, stage_specs

STAGE_RUNNER_WORKFLOW = "ci-priority-stage-runner.yaml"
RUN_TITLE_RE = re.compile(
    r"^omni-ci pr=(?P<pr>\d+) sha=(?P<sha>\S+) "
    r"stage=(?P<stage>\S+) rerun=(?P<rerun>\S+)$"
)
RERUN_COMMAND = "/rerun-failed-ci"
DISPATCH_VISIBILITY_TIMEOUT_SECONDS = 60
DISPATCH_VISIBILITY_POLL_SECONDS = 3


@dataclass(frozen=True)
class PullRequestMeta:
    state: PullRequestState
    head_repo: str


class GitHubClient:
    def __init__(self, repo: str, token: str) -> None:
        self.repo = repo
        self.token = token

    def get(self, path: str, params: dict[str, str | int] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, payload=payload)

    def patch(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request("PATCH", path, payload=payload)

    def paginate(
        self,
        path: str,
        params: dict[str, str | int] | None = None,
    ):
        page = 1
        while True:
            request_params = dict(params or {})
            request_params.update({"per_page": 100, "page": page})
            data = self.get(path, request_params)
            if isinstance(data, dict):
                items = next((v for v in data.values() if isinstance(v, list)), [])
            else:
                items = data
            yield from items
            if len(items) < 100:
                break
            page += 1

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"https://api.github.com/repos/{self.repo}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            if not body:
                return None
            return json.loads(body)


def main() -> int:
    event = _load_event()
    if _is_irrelevant_comment_event(event):
        print("Comment does not request /rerun-failed-ci; nothing to schedule.")
        return 0
    settle_seconds = int(os.environ.get("OMNI_CI_SCHEDULER_SETTLE_SECONDS", "5"))
    if settle_seconds:
        time.sleep(settle_seconds)

    client = GitHubClient(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
    pull_requests = _open_pull_requests(client)
    if not pull_requests:
        print("No open pull requests found.")
        return 0

    runs = _stage_runs(client)
    rerun_requests = _rerun_requests(client, pull_requests, runs)
    _sync_stage_check_runs(client, pull_requests, runs)
    decision = select_next_stage(
        pull_requests=[meta.state for meta in pull_requests],
        stages=stage_specs(),
        runs=runs,
        rerun_requests=rerun_requests,
    )

    if decision.preempt_stage is not None:
        _cancel_stage_run(client, decision.preempt_stage)

    if decision.active_stage is not None:
        print(
            "A CI stage is already active: "
            f"PR #{decision.active_stage.pr_number} "
            f"{decision.active_stage.stage_id} ({decision.active_stage.status})."
        )
        return 0

    if decision.selected_stage is None:
        print("No queued CI stage is ready to dispatch.")
        return 0

    meta_by_pr = {meta.state.number: meta for meta in pull_requests}
    meta = meta_by_pr[decision.selected_stage.pr.number]
    _mark_selected_check_in_progress(client, decision.selected_stage)
    _dispatch_stage(client, meta, decision.selected_stage)
    return 0


def _load_event() -> dict[str, Any]:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return {}
    with open(event_path) as f:
        return json.load(f)


def _is_irrelevant_comment_event(event: dict[str, Any]) -> bool:
    if os.environ.get("GITHUB_EVENT_NAME") != "issue_comment":
        return False
    if not event.get("issue", {}).get("pull_request"):
        return True
    return RERUN_COMMAND not in event.get("comment", {}).get("body", "")


def _open_pull_requests(client: GitHubClient) -> list[PullRequestMeta]:
    pull_requests: list[PullRequestMeta] = []
    for pull in client.paginate("/pulls", {"state": "open"}):
        if pull.get("base", {}).get("ref") != "main":
            continue
        number = int(pull["number"])
        issue = client.get(f"/issues/{number}")
        labels = frozenset(label["name"] for label in issue.get("labels", []))
        head = pull["head"]
        state = PullRequestState(
            number=number,
            labels=labels,
            draft=bool(pull.get("draft")),
            head_sha=head["sha"],
            opened_at=pull["created_at"],
        )
        pull_requests.append(
            PullRequestMeta(
                state=state,
                head_repo=head["repo"]["full_name"],
            )
        )
    return pull_requests


def _stage_runs(client: GitHubClient) -> list[StageRunState]:
    workflow_id = urllib.parse.quote(STAGE_RUNNER_WORKFLOW, safe="")
    runs: list[StageRunState] = []
    for run in client.paginate(f"/actions/workflows/{workflow_id}/runs"):
        parsed = _parse_stage_run_title(run.get("display_title") or "")
        if parsed is None:
            continue
        runs.append(
            StageRunState(
                pr_number=parsed["pr_number"],
                head_sha=parsed["head_sha"],
                stage_id=parsed["stage_id"],
                status=run["status"],
                conclusion=run.get("conclusion"),
                created_at=run["created_at"],
                rerun_request_id=parsed["rerun_request_id"],
                run_id=int(run["id"]),
            )
        )
    return runs


def _sync_stage_check_runs(
    client: GitHubClient,
    pull_requests: list[PullRequestMeta],
    runs: list[StageRunState],
) -> None:
    stages = stage_specs()
    for meta in pull_requests:
        existing_checks = _stage_check_runs_for_sha(client, meta.state.head_sha)
        for stage in stages:
            latest = _latest_stage_run(meta.state, stage.id, runs)
            if ci_enabled(meta.state):
                payload = _check_payload_for_stage_run(
                    meta.state,
                    stage,
                    latest,
                    status="queued",
                )
                _upsert_check_run(client, existing_checks, payload)
                continue

            existing = existing_checks.get(_stage_check_name(stage))
            if existing and existing.get("status") != "completed":
                _update_check_run(
                    client,
                    existing,
                    {
                        "name": _stage_check_name(stage),
                        "status": "completed",
                        "conclusion": "skipped",
                        "completed_at": _now_utc(),
                    },
                )


def _mark_selected_check_in_progress(client: GitHubClient, selected) -> None:
    existing_checks = _stage_check_runs_for_sha(client, selected.pr.head_sha)
    payload = {
        "name": _stage_check_name(selected.stage),
        "head_sha": selected.pr.head_sha,
        "status": "in_progress",
        "started_at": _now_utc(),
        "external_id": _stage_external_id(
            selected.pr.number,
            selected.pr.head_sha,
            selected.stage.id,
        ),
    }
    _upsert_check_run(client, existing_checks, payload)


def _check_payload_for_stage_run(
    pr: PullRequestState,
    stage,
    run: StageRunState | None,
    *,
    status: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": _stage_check_name(stage),
        "head_sha": pr.head_sha,
        "status": status,
        "external_id": _stage_external_id(pr.number, pr.head_sha, stage.id),
    }
    if run is None:
        return payload

    if run.status in QUEUED_STATUSES:
        payload["status"] = "queued"
    elif run.status in RUNNING_STATUSES:
        payload["status"] = "in_progress"
        payload["started_at"] = run.created_at
    elif run.status == "completed":
        payload["status"] = "completed"
        payload["conclusion"] = run.conclusion or "failure"
        payload["completed_at"] = _now_utc()

    if run.run_id is not None:
        payload["details_url"] = (
            f"https://github.com/{client_repo()}/actions/runs/{run.run_id}"
        )
    return payload


def _stage_check_runs_for_sha(
    client: GitHubClient,
    head_sha: str,
) -> dict[str, dict[str, Any]]:
    checks_by_name: dict[str, dict[str, Any]] = {}
    for check in client.paginate(
        f"/commits/{head_sha}/check-runs",
        {"filter": "latest"},
    ):
        name = check.get("name")
        if not name:
            continue
        existing = checks_by_name.get(name)
        if existing is None or _check_sort_key(check) > _check_sort_key(existing):
            checks_by_name[name] = check
    return checks_by_name


def _check_sort_key(check: dict[str, Any]) -> tuple[str, int]:
    return (
        check.get("started_at")
        or check.get("completed_at")
        or check.get("created_at")
        or "",
        int(check.get("id") or 0),
    )


def _upsert_check_run(
    client: GitHubClient,
    existing_checks: dict[str, dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    existing = existing_checks.get(payload["name"])
    if existing is None:
        created = client.post("/check-runs", payload)
        if isinstance(created, dict):
            existing_checks[payload["name"]] = created
        return
    if existing.get("status") == "completed" and payload.get("status") != "completed":
        created = client.post("/check-runs", payload)
        if isinstance(created, dict):
            existing_checks[payload["name"]] = created
        return
    _update_check_run(client, existing, payload)


def _update_check_run(
    client: GitHubClient,
    existing: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    update_payload = dict(payload)
    update_payload.pop("head_sha", None)
    client.patch(f"/check-runs/{existing['id']}", update_payload)


def _stage_check_name(stage) -> str:
    return f"{stage.workflow} / {stage.name}"


def _stage_external_id(pr_number: int, head_sha: str, stage_id: str) -> str:
    return f"pr={pr_number} sha={head_sha} stage={stage_id}"


def _latest_stage_run(
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
    return max(matches, key=lambda run: (run.created_at, run.run_id or 0))


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def client_repo() -> str:
    return os.environ["GITHUB_REPOSITORY"]


def _parse_stage_run_title(title: str) -> dict[str, Any] | None:
    match = RUN_TITLE_RE.match(title)
    if match is None:
        return None
    rerun = match.group("rerun")
    return {
        "pr_number": int(match.group("pr")),
        "head_sha": match.group("sha"),
        "stage_id": match.group("stage"),
        "rerun_request_id": None if rerun == "-" else int(rerun),
    }


def _rerun_requests(
    client: GitHubClient,
    pull_requests: list[PullRequestMeta],
    runs: list[StageRunState],
) -> list[RerunRequest]:
    requests: list[RerunRequest] = []
    for meta in pull_requests:
        for comment in client.paginate(f"/issues/{meta.state.number}/comments"):
            if RERUN_COMMAND not in comment.get("body", ""):
                continue
            stage_ids = _failed_stage_ids_before_request(
                meta.state,
                runs,
                comment["created_at"],
            )
            if not stage_ids:
                continue
            requests.append(
                RerunRequest(
                    pr_number=meta.state.number,
                    request_id=int(comment["id"]),
                    created_at=comment["created_at"],
                    stage_ids=frozenset(stage_ids),
                )
            )
    return requests


def _failed_stage_ids_before_request(
    pr: PullRequestState,
    runs: list[StageRunState],
    timestamp: str,
) -> set[str]:
    latest_by_stage: dict[str, StageRunState] = {}
    for run in runs:
        if run.pr_number != pr.number or run.head_sha != pr.head_sha:
            continue
        if run.created_at > timestamp:
            continue
        current = latest_by_stage.get(run.stage_id)
        if current is None or (run.created_at, run.run_id or 0) > (
            current.created_at,
            current.run_id or 0,
        ):
            latest_by_stage[run.stage_id] = run
    return {
        stage_id
        for stage_id, run in latest_by_stage.items()
        if run.status == "completed" and run.conclusion in FAILED_CONCLUSIONS
    }


def _cancel_stage_run(client: GitHubClient, run: StageRunState) -> None:
    if run.run_id is None:
        print(
            f"Cannot cancel preempted PR #{run.pr_number} {run.stage_id}: "
            "missing run_id."
        )
        return
    print(
        f"Cancelling queued lower-priority stage run {run.run_id}: "
        f"PR #{run.pr_number} {run.stage_id}."
    )
    client.post(f"/actions/runs/{run.run_id}/cancel")


def _dispatch_stage(client: GitHubClient, meta: PullRequestMeta, selected) -> None:
    workflow_id = urllib.parse.quote(STAGE_RUNNER_WORKFLOW, safe="")
    rerun_request_id = selected.rerun_request_id
    print(
        f"Dispatching PR #{meta.state.number} {selected.stage.id} "
        f"at {meta.state.head_sha}."
    )
    payload = {
        # Dispatch the trusted workflow definition from the default branch; the
        # runner job checks out the PR head SHA explicitly.
        "ref": "main",
        "inputs": {
            "pr_number": str(meta.state.number),
            "head_sha": meta.state.head_sha,
            "head_repo": meta.head_repo,
            "stage_id": selected.stage.id,
            "stage_name": selected.stage.name,
            "stage_config": dispatch_config_for_stage(selected.stage.id),
            "rerun_request_id": (
                "-" if rerun_request_id is None else str(rerun_request_id)
            ),
        },
    }
    client.post(f"/actions/workflows/{workflow_id}/dispatches", payload)
    _wait_for_dispatched_stage(client, selected)


def _wait_for_dispatched_stage(client: GitHubClient, selected) -> None:
    timeout_seconds = int(
        os.environ.get(
            "OMNI_CI_DISPATCH_VISIBILITY_TIMEOUT_SECONDS",
            str(DISPATCH_VISIBILITY_TIMEOUT_SECONDS),
        )
    )
    poll_seconds = int(
        os.environ.get(
            "OMNI_CI_DISPATCH_VISIBILITY_POLL_SECONDS",
            str(DISPATCH_VISIBILITY_POLL_SECONDS),
        )
    )
    deadline = time.monotonic() + timeout_seconds
    while True:
        for run in _stage_runs(client):
            if (
                run.pr_number == selected.pr.number
                and run.head_sha == selected.pr.head_sha
                and run.stage_id == selected.stage.id
                and run.rerun_request_id == selected.rerun_request_id
            ):
                print(
                    "Dispatched stage is now visible to the scheduler: "
                    f"run_id={run.run_id} status={run.status}."
                )
                return

        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for the dispatched CI stage runner to become "
                "visible through the GitHub Actions API."
            )
        time.sleep(poll_seconds)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        print(f"GitHub API request failed: HTTP {exc.code} {exc.reason}")
        print(exc.read().decode("utf-8", errors="replace"))
        sys.exit(1)
