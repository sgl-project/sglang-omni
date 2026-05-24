#!/usr/bin/env python3

"""Wait normal-priority CI behind active high-priority PR CI runs."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ACTIVE_STATUSES = ("queued", "in_progress", "waiting", "pending", "requested")
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


def _priority_workflows() -> set[str]:
    raw = os.environ.get("OMNI_CI_PRIORITY_WORKFLOWS")
    if raw:
        return {name.strip() for name in raw.split(",") if name.strip()}
    return set(DEFAULT_PRIORITY_WORKFLOWS)


def _is_current_run_high_priority(
    event: dict,
    *,
    run_label: str,
    high_priority_label: str,
) -> bool:
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        print("Non-PR event; priority gate is bypassed.")
        return True

    labels = _label_names_from_payload(event)
    if high_priority_label in labels and run_label in labels:
        print(
            f"Current PR has both `{run_label}` and `{high_priority_label}`; "
            "priority gate is bypassed."
        )
        return True

    print(
        f"Current PR is normal priority; waiting behind active `{run_label}` + "
        f"`{high_priority_label}` CI runs."
    )
    return False


def _labels_for_pr(
    client: GitHubClient,
    label_cache: dict[int, set[str]],
    pr_number: int,
) -> set[str]:
    if pr_number not in label_cache:
        labels = client.paginate(f"/issues/{pr_number}/labels")
        label_cache[pr_number] = {label["name"] for label in labels}
    return label_cache[pr_number]


def _active_high_priority_runs(
    client: GitHubClient,
    *,
    current_run_id: int,
    priority_workflows: set[str],
    run_label: str,
    high_priority_label: str,
) -> list[dict]:
    label_cache: dict[int, set[str]] = {}
    matches: list[dict] = []
    seen_run_ids: set[int] = set()

    for status in ACTIVE_STATUSES:
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
                if run_label in labels and high_priority_label in labels:
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
    poll_seconds = _env_int("OMNI_CI_PRIORITY_POLL_SECONDS", 30)
    timeout_seconds = _env_int("OMNI_CI_PRIORITY_TIMEOUT_SECONDS", 6 * 60 * 60)
    workflows = _priority_workflows()
    event = _load_event()

    if _is_current_run_high_priority(
        event,
        run_label=run_label,
        high_priority_label=high_priority_label,
    ):
        return 0

    client = GitHubClient(repo, token)
    deadline = time.monotonic() + timeout_seconds

    while True:
        try:
            active_runs = _active_high_priority_runs(
                client,
                current_run_id=current_run_id,
                priority_workflows=workflows,
                run_label=run_label,
                high_priority_label=high_priority_label,
            )
        except urllib.error.HTTPError as exc:
            print(f"GitHub API request failed: HTTP {exc.code} {exc.reason}")
            print(exc.read().decode("utf-8", errors="replace"))
            return 1

        if not active_runs:
            print(
                "No active high-priority CI runs found; normal-priority CI can start."
            )
            return 0

        print("Waiting for active high-priority CI runs:")
        for run in active_runs:
            print(
                f"  - {run['name']} for PR #{run['pr']} "
                f"({run['status']}, run_id={run['id']}): {run['url']}"
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
