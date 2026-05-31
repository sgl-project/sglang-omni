# SPDX-License-Identifier: Apache-2.0
"""Gate expensive PR CI on lint success and mergeability."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def _github_json(
    token: str,
    repo: str,
    path: str,
    query: dict[str, str] | None = None,
) -> dict:
    url = f"https://api.github.com/repos/{repo}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {exc.code} for {path}: {body}") from exc


def _load_event() -> dict:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise RuntimeError("GITHUB_EVENT_PATH is not set")
    with open(event_path, encoding="utf-8") as event_file:
        return json.load(event_file)


def _wait_for_mergeability(
    token: str,
    repo: str,
    pull_number: int,
    timeout_seconds: int,
    poll_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_state = None
    while time.monotonic() < deadline:
        pull = _github_json(token, repo, f"/pulls/{pull_number}")
        mergeable = pull.get("mergeable")
        mergeable_state = pull.get("mergeable_state")
        last_state = mergeable_state
        print(
            "PR mergeability:",
            f"mergeable={mergeable}",
            f"mergeable_state={mergeable_state}",
        )
        if mergeable_state == "dirty":
            raise RuntimeError(
                "PR has merge conflicts with the base branch; skip staged CI."
            )
        if mergeable_state not in (None, "unknown"):
            return
        time.sleep(poll_seconds)
    raise RuntimeError(
        "Could not determine PR mergeability before timeout "
        f"(last mergeable_state={last_state!r})."
    )


def _latest_check_run(check_runs: list[dict], check_name: str) -> dict | None:
    matching_runs = [run for run in check_runs if run.get("name") == check_name]
    matching_runs.sort(
        key=lambda run: run.get("started_at") or run.get("created_at") or "",
        reverse=True,
    )
    return matching_runs[0] if matching_runs else None


def _wait_for_lint_success(
    token: str,
    repo: str,
    head_sha: str,
    check_name: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        checks = _github_json(
            token,
            repo,
            f"/commits/{head_sha}/check-runs",
            {
                "check_name": check_name,
                "filter": "latest",
                "per_page": "100",
            },
        )
        check_run = _latest_check_run(checks.get("check_runs", []), check_name)
        if check_run is None:
            print(f"Waiting for {check_name!r} check on {head_sha}...")
            time.sleep(poll_seconds)
            continue

        status = check_run.get("status")
        conclusion = check_run.get("conclusion")
        details_url = check_run.get("details_url")
        print(
            f"Found {check_name!r} check:",
            f"status={status}",
            f"conclusion={conclusion}",
            f"url={details_url}",
        )
        if status == "completed":
            if conclusion == "success":
                return
            raise RuntimeError(
                f"{check_name!r} check completed with conclusion={conclusion}; "
                "skip staged CI."
            )
        time.sleep(poll_seconds)

    raise RuntimeError(
        f"{check_name!r} check did not complete before timeout for {head_sha}."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-name", default="lint")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()

    event_name = os.environ.get("GITHUB_EVENT_NAME")
    if event_name != "pull_request":
        print(f"{event_name} event bypasses PR lint/mergeability gate.")
        return 0

    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        raise RuntimeError("GITHUB_TOKEN and GITHUB_REPOSITORY must be set")

    event = _load_event()
    pull_request = event["pull_request"]
    pull_number = int(pull_request["number"])
    head_sha = pull_request["head"]["sha"]
    base_branch = pull_request["base"]["ref"]
    print(
        "Checking PR CI gate:",
        f"pr={pull_number}",
        f"base={base_branch}",
        f"head_sha={head_sha}",
    )

    _wait_for_mergeability(
        token,
        repo,
        pull_number,
        args.timeout_seconds,
        args.poll_seconds,
    )
    _wait_for_lint_success(
        token,
        repo,
        head_sha,
        args.check_name,
        args.timeout_seconds,
        args.poll_seconds,
    )
    print("PR CI gate passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PR CI gate failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
