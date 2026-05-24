#!/usr/bin/env python3

"""Wait normal-priority CI behind active high-priority PR CI work."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ACTIVE_STATUSES = ("queued", "in_progress", "waiting", "pending", "requested")
BLOCKING_COMPLETED_STATUSES = ("failure", "timed_out", "cancelled")
BLOCKING_JOB_CONCLUSIONS = {"failure", "timed_out", "cancelled"}
PRIORITY_GATE_JOB_NAME = "priority gate"
REUSABLE_JOB_SEPARATOR = " / "
DEFAULT_PRIORITY_WORKFLOWS = (
    "PR Test",
    "PR Test (Examples)",
    "Qwen3-Omni CI",
    "S2-Pro CI",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer; got {raw!r}") from None
    if value < 0:
        raise SystemExit(f"{name} must be non-negative; got {value}")
    return value


class GitHubClient:
    def __init__(self, repo: str, token: str) -> None:
        self.repo = repo
        self.token = token

    def get(self, path: str, params: dict[str, str | int] | None = None):
        url = f"https://api.github.com/repos/{self.repo}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    def paginate(self, path: str, params: dict[str, str | int] | None = None):
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


def _load_event() -> dict:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return {}
    with open(event_path) as f:
        return json.load(f)


def _label_names_from_payload(event: dict) -> set[str]:
    pull_request = event.get("pull_request") or {}
    return {label["name"] for label in pull_request.get("labels", [])}


def _pr_number_from_payload(event: dict) -> int | None:
    pull_request = event.get("pull_request") or {}
    number = pull_request.get("number") or event.get("number")
    if number is None:
        return None
    return int(number)


def _priority_workflows() -> set[str]:
    raw = os.environ.get("OMNI_CI_PRIORITY_WORKFLOWS")
    if raw:
        return {name.strip() for name in raw.split(",") if name.strip()}
    return set(DEFAULT_PRIORITY_WORKFLOWS)


def _current_run_priority_state(
    client: GitHubClient,
    event: dict,
    *,
    run_label: str,
    high_priority_label: str,
    stage_name: str,
) -> str:
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request" and not event.get(
        "pull_request"
    ):
        print("Non-PR event; priority gate is bypassed.")
        return "bypass"

    pr_number = _pr_number_from_payload(event)
    labels = _label_names_from_payload(event)
    if pr_number is not None:
        try:
            labels = _labels_for_pr(client, {}, pr_number)
        except urllib.error.HTTPError as exc:
            print(
                "Failed to refresh current PR labels from GitHub API; "
                f"falling back to event payload labels: HTTP {exc.code} {exc.reason}"
            )
    if high_priority_label in labels and run_label in labels:
        stage_message = f" for `{stage_name}`" if stage_name else ""
        print(
            f"Current PR has both `{run_label}` and `{high_priority_label}`; "
            f"entering high-priority stage gate{stage_message}."
        )
        return "high"

    stage_message = f" `{stage_name}`" if stage_name else ""
    print(
        f"Current PR is normal priority; waiting behind active{stage_message} "
        f"`{run_label}` + `{high_priority_label}` CI work."
    )
    return "normal"


def _labels_for_pr(
    client: GitHubClient,
    label_cache: dict[int, set[str]],
    pr_number: int,
) -> set[str]:
    if pr_number not in label_cache:
        labels = client.paginate(f"/issues/{pr_number}/labels")
        label_cache[pr_number] = {label["name"] for label in labels}
    return label_cache[pr_number]


def _head_sha_for_pr(
    client: GitHubClient,
    head_sha_cache: dict[int, str | None],
    pr_number: int,
) -> str | None:
    if pr_number not in head_sha_cache:
        pull_request = client.get(f"/pulls/{pr_number}")
        head = pull_request.get("head") or {}
        head_sha_cache[pr_number] = head.get("sha")
    return head_sha_cache[pr_number]


def _job_matches_stage(job_name: str | None, stage_name: str) -> bool:
    if not stage_name:
        return True
    if not job_name:
        return False
    if job_name == stage_name:
        return True
    # Matrix jobs append their matrix values to the configured job name.
    if job_name.startswith(f"{stage_name} ("):
        return True
    # Reusable workflow jobs can be reported as "<caller job> / <called job>".
    if job_name.endswith(f" / {stage_name}"):
        return True
    return f" / {stage_name} (" in job_name


def _is_priority_gate_job(job_name: str | None) -> bool:
    if not job_name:
        return False
    return job_name == PRIORITY_GATE_JOB_NAME or job_name.endswith(
        f"{REUSABLE_JOB_SEPARATOR}{PRIORITY_GATE_JOB_NAME}"
    )


def _job_matches_priority_gate(job_name: str | None, stage_name: str) -> bool:
    if not job_name:
        return False
    if not stage_name:
        return job_name == PRIORITY_GATE_JOB_NAME
    return job_name == f"{stage_name}{REUSABLE_JOB_SEPARATOR}{PRIORITY_GATE_JOB_NAME}"


def _reusable_caller_name(job_name: str | None) -> str:
    if not job_name or REUSABLE_JOB_SEPARATOR not in job_name:
        return ""
    return job_name.split(REUSABLE_JOB_SEPARATOR, 1)[0]


def _is_reusable_stage_runner_job(job_name: str | None) -> bool:
    return bool(
        job_name
        and REUSABLE_JOB_SEPARATOR in job_name
        and not _is_priority_gate_job(job_name)
    )


def _job_sort_key(job: dict) -> tuple[str, int]:
    job_id = job.get("job_id", job["id"])
    return (str(job.get("started_at") or job.get("created_at") or ""), int(job_id))


def _stage_gate_sort_key(job: dict) -> tuple[int, str, int]:
    return (int(job.get("run_priority", 1)), *_job_sort_key(job))


def _active_matching_jobs(
    client: GitHubClient,
    *,
    run_id: int,
    stage_name: str,
) -> list[dict]:
    matches: list[dict] = []
    for job in client.paginate(f"/actions/runs/{run_id}/jobs"):
        if job.get("status") not in ACTIVE_STATUSES:
            continue
        if not _job_matches_stage(job.get("name"), stage_name):
            continue
        matches.append(job)
    return matches


def _blocking_completed_jobs(
    client: GitHubClient,
    *,
    run_id: int,
    stage_name: str,
) -> list[dict]:
    matches: list[dict] = []
    for job in client.paginate(f"/actions/runs/{run_id}/jobs"):
        if job.get("status") != "completed":
            continue
        if job.get("conclusion") not in BLOCKING_JOB_CONCLUSIONS:
            continue
        if not _job_matches_stage(job.get("name"), stage_name):
            continue
        matches.append(job)
    return matches


def _run_priority_from_labels(labels: set[str], *, high_priority_label: str) -> int:
    return 0 if high_priority_label in labels else 1


def _stage_gate_blockers(
    client: GitHubClient,
    *,
    current_run_id: int,
    priority_workflows: set[str],
    run_label: str,
    high_priority_label: str,
    stage_name: str,
) -> list[dict]:
    label_cache: dict[int, set[str]] = {}
    blockers: list[dict] = []
    gate_contenders: list[dict] = []
    seen_run_ids: set[int] = set()

    for status in ACTIVE_STATUSES:
        for run in client.paginate("/actions/runs", {"status": status}):
            run_id = int(run["id"])
            if run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)

            if priority_workflows and run.get("name") not in priority_workflows:
                continue
            if run.get("event") != "pull_request":
                continue

            is_ci_run = False
            run_priority = 1
            pr_number = None
            for pr in run.get("pull_requests") or []:
                pr_number = int(pr["number"])
                labels = _labels_for_pr(client, label_cache, pr_number)
                if run_label in labels:
                    is_ci_run = True
                    run_priority = _run_priority_from_labels(
                        labels,
                        high_priority_label=high_priority_label,
                    )
                    break
            if not is_ci_run:
                continue

            jobs = list(client.paginate(f"/actions/runs/{run_id}/jobs"))
            active_runner_stages = set()
            completed_runner_stages = set()

            for job in jobs:
                job_name = job.get("name")
                if not _is_reusable_stage_runner_job(job_name):
                    continue
                caller_name = _reusable_caller_name(job_name)
                if job.get("status") in ACTIVE_STATUSES:
                    active_runner_stages.add(caller_name)
                    blockers.append(
                        {
                            "id": run_id,
                            "name": run.get("name"),
                            "pr": pr_number,
                            "status": run.get("status"),
                            "url": run.get("html_url"),
                            "job": job_name,
                            "job_status": job.get("status"),
                            "job_conclusion": job.get("conclusion"),
                            "run_priority": run_priority,
                        }
                    )
                elif job.get("status") == "completed":
                    completed_runner_stages.add(caller_name)

            for job in jobs:
                job_name = job.get("name")
                if not _is_priority_gate_job(job_name):
                    continue
                gate_stage = _reusable_caller_name(job_name)
                if job.get("status") in ACTIVE_STATUSES:
                    gate_contenders.append(
                        {
                            "id": run_id,
                            "name": run.get("name"),
                            "pr": pr_number,
                            "status": run.get("status"),
                            "url": run.get("html_url"),
                            "job": job_name,
                            "job_status": job.get("status"),
                            "job_conclusion": job.get("conclusion"),
                            "job_id": int(job["id"]),
                            "started_at": job.get("started_at"),
                            "created_at": job.get("created_at"),
                            "run_priority": run_priority,
                        }
                    )
                elif (
                    job.get("status") == "completed"
                    and job.get("conclusion") == "success"
                    and gate_stage
                    and gate_stage not in active_runner_stages
                    and gate_stage not in completed_runner_stages
                ):
                    blockers.append(
                        {
                            "id": run_id,
                            "name": run.get("name"),
                            "pr": pr_number,
                            "status": run.get("status"),
                            "url": run.get("html_url"),
                            "job": job_name,
                            "job_status": "waiting for stage job",
                            "job_conclusion": None,
                            "run_priority": run_priority,
                        }
                    )

    if blockers:
        return blockers
    if not gate_contenders:
        return []

    current_gates = [
        job
        for job in gate_contenders
        if job["id"] == current_run_id
        and _job_matches_priority_gate(job.get("job"), stage_name)
    ]
    if not current_gates:
        winner = min(gate_contenders, key=_stage_gate_sort_key)
        return [winner]

    current_gate = min(current_gates, key=_stage_gate_sort_key)
    winner = min(gate_contenders, key=_stage_gate_sort_key)
    if current_gate["job_id"] == winner["job_id"]:
        return []
    return [winner]


def _active_high_priority_runs(
    client: GitHubClient,
    *,
    current_run_id: int,
    priority_workflows: set[str],
    run_label: str,
    high_priority_label: str,
    stage_name: str,
) -> list[dict]:
    label_cache: dict[int, set[str]] = {}
    head_sha_cache: dict[int, str | None] = {}
    matches: list[dict] = []
    seen_run_ids: set[int] = set()

    statuses = ACTIVE_STATUSES
    if stage_name:
        statuses += BLOCKING_COMPLETED_STATUSES

    for status in statuses:
        for run in client.paginate("/actions/runs", {"status": status}):
            run_id = int(run["id"])
            if run_id == current_run_id or run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)

            if priority_workflows and run.get("name") not in priority_workflows:
                continue
            if run.get("event") != "pull_request":
                continue

            for pr in run.get("pull_requests") or []:
                pr_number = int(pr["number"])
                labels = _labels_for_pr(client, label_cache, pr_number)
                if run_label not in labels or high_priority_label not in labels:
                    continue

                if stage_name:
                    if status in BLOCKING_COMPLETED_STATUSES:
                        current_head_sha = _head_sha_for_pr(
                            client,
                            head_sha_cache,
                            pr_number,
                        )
                        if current_head_sha != run.get("head_sha"):
                            continue
                        jobs = _blocking_completed_jobs(
                            client,
                            run_id=run_id,
                            stage_name=stage_name,
                        )
                    else:
                        jobs = _active_matching_jobs(
                            client,
                            run_id=run_id,
                            stage_name=stage_name,
                        )
                        if not jobs:
                            jobs = _blocking_completed_jobs(
                                client,
                                run_id=run_id,
                                stage_name=stage_name,
                            )
                        if not jobs:
                            # A freshly queued run can briefly have no materialized
                            # jobs. Block conservatively until GitHub exposes them.
                            if run.get("status") in {"queued", "requested", "pending"}:
                                matches.append(
                                    {
                                        "id": run_id,
                                        "name": run.get("name"),
                                        "pr": pr_number,
                                        "status": run.get("status"),
                                        "url": run.get("html_url"),
                                        "job": stage_name,
                                        "job_status": "not materialized",
                                        "job_conclusion": None,
                                    }
                                )
                                break
                            continue

                    for job in jobs:
                        matches.append(
                            {
                                "id": run_id,
                                "name": run.get("name"),
                                "pr": pr_number,
                                "status": run.get("status"),
                                "url": run.get("html_url"),
                                "job": job.get("name"),
                                "job_status": job.get("status"),
                                "job_conclusion": job.get("conclusion"),
                            }
                        )
                    break

                else:
                    matches.append(
                        {
                            "id": run_id,
                            "name": run.get("name"),
                            "pr": pr_number,
                            "status": run.get("status"),
                            "url": run.get("html_url"),
                        }
                    )
                    break

    return matches


def main() -> int:
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    current_run_id = int(os.environ["GITHUB_RUN_ID"])
    run_label = os.environ.get("OMNI_CI_RUN_LABEL", "run-ci")
    high_priority_label = os.environ.get("OMNI_CI_HIGH_PRIORITY_LABEL", "high-priority")
    stage_name = os.environ.get("OMNI_CI_PRIORITY_STAGE", "")
    poll_seconds = _env_int("OMNI_CI_PRIORITY_POLL_SECONDS", 30)
    timeout_seconds = _env_int("OMNI_CI_PRIORITY_TIMEOUT_SECONDS", 6 * 60 * 60)
    settle_seconds = _env_int("OMNI_CI_PRIORITY_SETTLE_SECONDS", 15)
    workflows = _priority_workflows()
    event = _load_event()
    client = GitHubClient(repo, token)

    priority_state = _current_run_priority_state(
        client,
        event,
        run_label=run_label,
        high_priority_label=high_priority_label,
        stage_name=stage_name,
    )
    if priority_state == "bypass":
        return 0

    deadline = time.monotonic() + timeout_seconds
    if stage_name and settle_seconds:
        print(
            "Waiting "
            f"{settle_seconds} seconds for GitHub to materialize peer stage gates."
        )
        time.sleep(settle_seconds)

    while True:
        try:
            if stage_name:
                active_runs = _stage_gate_blockers(
                    client,
                    current_run_id=current_run_id,
                    priority_workflows=workflows,
                    run_label=run_label,
                    high_priority_label=high_priority_label,
                    stage_name=stage_name,
                )
            elif priority_state == "high":
                active_runs = []
            else:
                active_runs = _active_high_priority_runs(
                    client,
                    current_run_id=current_run_id,
                    priority_workflows=workflows,
                    run_label=run_label,
                    high_priority_label=high_priority_label,
                    stage_name=stage_name,
                )
        except urllib.error.HTTPError as exc:
            print(f"GitHub API request failed: HTTP {exc.code} {exc.reason}")
            print(exc.read().decode("utf-8", errors="replace"))
            return 1

        if not active_runs:
            stage_message = f" `{stage_name}`" if stage_name else ""
            if priority_state == "high":
                print(
                    f"High-priority{stage_message} gate acquired; "
                    "stage CI can start."
                )
            elif stage_name:
                print(f"Stage gate acquired for{stage_message}; stage CI can start.")
            else:
                print(
                    f"No active high-priority{stage_message} CI work found; "
                    "normal-priority CI can start."
                )
            return 0

        if stage_name:
            print("Waiting for the active CI stage slot:")
        else:
            print("Waiting for active high-priority CI work:")
        for run in active_runs:
            job_state = run.get("job_conclusion") or run.get("job_status")
            job_message = f", job={run['job']} ({job_state})" if "job" in run else ""
            print(
                f"  - {run['name']} for PR #{run['pr']} "
                f"({run['status']}, run_id={run['id']}{job_message}): "
                f"{run['url']}"
            )

        if time.monotonic() >= deadline:
            print(
                "Timed out waiting for high-priority CI runs after "
                f"{timeout_seconds} seconds."
            )
            return 1

        time.sleep(poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
