# SPDX-License-Identifier: Apache-2.0
"""Read frozen pull request revisions and check summaries from GitHub."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, Field, ValidationError

from scripts.docs_review.models import CheckFreshness, CheckRecord, CheckStatus


class PilotError(Exception):
    """A safe-to-display operational failure."""


class Revision(BaseModel):
    sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    ref: str = ""


class PullRequest(BaseModel):
    state: Literal["open", "closed"]
    body: str | None
    base: Revision
    head: Revision


def github_api(
    endpoint: str, *, body: str | None = None, method: str = "GET"
) -> str:
    arguments = ["gh", "api", endpoint]
    input_text: str | None = None
    if body is not None:
        arguments.extend(["--method", method, "--input", "-"])
        input_text = body
    completed = subprocess.run(
        arguments, input=input_text, capture_output=True, text=True, check=False
    )
    if completed.returncode:
        status = re.search(r"HTTP (\d{3})", completed.stderr)
        if status is None:
            raise PilotError("GitHub API request failed; check repository permissions")
        else:
            raise PilotError(f"GitHub API request failed (HTTP {status.group(1)})")
    else:
        return completed.stdout


def fetch_pull_request(repository_name: str, pull_request_number: int) -> PullRequest:
    response = github_api(f"repos/{repository_name}/pulls/{pull_request_number}")
    return PullRequest.model_validate_json(response)


def fetch_revision(repository: Path, revision: str) -> None:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            "fetch",
            "--no-tags",
            "origin",
            revision,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise PilotError("Could not fetch a pull request revision")
    else:
        pass


class GitHubCheckRun(BaseModel):
    name: str
    head_sha: str
    status: str
    conclusion: str | None


class GitHubCheckRuns(BaseModel):
    check_runs: list[GitHubCheckRun]


GITHUB_CHECK_STATUSES = {
    "success": CheckStatus.PASSED,
    "failure": CheckStatus.FAILED,
    "skipped": CheckStatus.SKIPPED,
    "cancelled": CheckStatus.CANCELLED,
    "timed_out": CheckStatus.TIMED_OUT,
}


def fetch_unit_test_checks(
    repository_name: str, head_commit_sha: str
) -> list[CheckRecord]:
    try:
        response = github_api(
            f"repos/{repository_name}/commits/{head_commit_sha}/check-runs?per_page=100"
        )
        checks = GitHubCheckRuns.model_validate_json(response).check_runs
    except (PilotError, ValidationError):
        return []
    else:
        return [
            CheckRecord(
                name=check.name,
                status=(
                    GITHUB_CHECK_STATUSES.get(
                        check.conclusion or "", CheckStatus.UNKNOWN
                    )
                    if check.status == "completed"
                    else CheckStatus.PENDING
                ),
                freshness=(
                    CheckFreshness.CURRENT
                    if check.head_sha == head_commit_sha
                    else CheckFreshness.STALE
                ),
                execution_commit_sha=check.head_sha,
                coverage="GitHub check summary. Per-test coverage is unknown.",
            )
            for check in checks
        ]
