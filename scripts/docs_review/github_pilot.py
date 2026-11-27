# SPDX-License-Identifier: Apache-2.0
"""Run an authorized GitHub documentation review without executing PR code."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from scripts.docs_review.context import run_git
from scripts.docs_review.github_comment import (
    ADVISORY_BOT_LOGIN,
    ADVISORY_REVIEW_MARKER,
    render_advisory_comment,
    render_artifact,
)
from scripts.docs_review.models import (
    BudgetConfig,
    ModelSettings,
    ReviewArtifact,
    ReviewRequest,
)
from scripts.docs_review.reporting import render_bundle
from scripts.docs_review.review import SystemClock, prepare_review, run_review

LOGGER = logging.getLogger(__name__)

STATE_BRANCH = "docs-review-ledger"
STATE_PATH = "docs-review-ledger.json"
MAX_INPUT_BYTES = 100_000
MAX_REVIEWS_PER_PULL_REQUEST = 2
LEDGER_WRITE_ATTEMPTS = 5


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


class GitHubContent(BaseModel):
    sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    content: str
    encoding: Literal["base64"]


def github_api(
    endpoint: str,
    *,
    method: Literal["GET", "PUT", "POST", "PATCH"],
    body: str | None = None,
) -> str:
    arguments = ["gh", "api", "--method", method, endpoint]
    if body is not None:
        arguments.extend(["--input", "-"])
    else:
        pass
    completed = subprocess.run(
        arguments, input=body, capture_output=True, text=True, check=False
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
    response = github_api(
        f"repos/{repository_name}/pulls/{pull_request_number}", method="GET"
    )
    return PullRequest.model_validate_json(response)


def fetch_revision(repository: Path, revision: str) -> None:
    if run_git(repository, ["fetch", "--no-tags", "origin", revision]).returncode:
        raise PilotError("Could not fetch a pull request revision")
    else:
        pass


def publish_report(repository_name: str, artifact: ReviewArtifact) -> None:
    repository_url = f"{os.environ['GITHUB_SERVER_URL']}/{repository_name}"
    endpoint = f"repos/{repository_name}/issues/{artifact.pull_request_number}/comments"
    page = 1
    while True:
        comments = json.loads(
            github_api(f"{endpoint}?per_page=100&page={page}", method="GET")
        )
        comment_id = next(
            (
                entry["id"]
                for entry in comments
                if (entry.get("user") or {}).get("login") == ADVISORY_BOT_LOGIN
                and ADVISORY_REVIEW_MARKER in (entry.get("body") or "")
            ),
            None,
        )
        if comment_id is not None or len(comments) < 100:
            break
        else:
            pass
        page += 1
    if not artifact.report.findings and comment_id is None:
        return
    else:
        pass
    body = render_advisory_comment(repository_url, artifact)
    if comment_id is not None:
        endpoint = f"repos/{repository_name}/issues/comments/{comment_id}"
    else:
        pass
    github_api(
        endpoint,
        method="PATCH" if comment_id is not None else "POST",
        body=json.dumps({"body": body}),
    )


class Reservation(BaseModel):
    pull_request_number: int = Field(gt=0)
    head_commit_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    review_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimated_cost_cents: int = Field(gt=0)
    date_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    run_id: str = Field(pattern=r"^\d+$")
    run_attempt: int = Field(default=1, gt=0)
    execution: Literal["reserved", "complete", "incomplete"] = "reserved"


class Reservations(BaseModel):
    version: Literal[1] = 1
    reservations: list[Reservation] = Field(default_factory=list)


def load_reservations(repository_name: str) -> tuple[Reservations, str]:
    endpoint = f"repos/{repository_name}/contents/{STATE_PATH}?ref={STATE_BRANCH}"
    content = GitHubContent.model_validate_json(github_api(endpoint, method="GET"))
    raw = base64.b64decode(content.content, validate=False)
    return Reservations.model_validate_json(raw), content.sha


def authorize_reservation(
    reservations: Reservations,
    reservation: Reservation,
    budget: BudgetConfig,
    follow_up_authorized: bool,
) -> None:
    pull_request_runs = [
        entry
        for entry in reservations.reservations
        if entry.pull_request_number == reservation.pull_request_number
    ]
    matching_runs = [
        entry
        for entry in pull_request_runs
        if entry.review_input_sha256 == reservation.review_input_sha256
    ]
    cost = reservation.estimated_cost_cents
    pr_spend = sum(entry.estimated_cost_cents for entry in pull_request_runs)
    day_spend = sum(
        entry.estimated_cost_cents
        for entry in reservations.reservations
        if entry.date_utc == reservation.date_utc
    )
    if matching_runs and (
        not follow_up_authorized
        or any(entry.execution == "complete" for entry in matching_runs)
    ):
        raise PilotError("This exact review was already reserved; see its earlier run")
    elif len(pull_request_runs) >= MAX_REVIEWS_PER_PULL_REQUEST:
        raise PilotError("Pull request review limit reached")
    elif pull_request_runs and not follow_up_authorized:
        raise PilotError("A follow-up review must be explicitly authorized")
    elif reservation.estimated_cost_cents > budget.max_cost_cents_per_call:
        raise PilotError("Estimated call cost exceeds the per-call budget")
    elif pr_spend + cost > budget.max_cost_cents_per_pull_request:
        raise PilotError("Pull request budget exceeded")
    elif day_spend + cost > budget.max_cost_cents_per_day:
        raise PilotError("Daily budget exceeded")
    else:
        pass


def update_reservation(
    repository_name: str,
    reservation: Reservation,
    execution: Literal["reserved", "complete", "incomplete"],
    *,
    budget: BudgetConfig | None = None,
    follow_up: bool = False,
    automatic: bool = False,
) -> bool:
    for attempt in range(LEDGER_WRITE_ATTEMPTS):
        ledger, sha = load_reservations(repository_name)
        if execution == "reserved":
            if automatic and any(
                entry.pull_request_number == reservation.pull_request_number
                and entry.review_input_sha256 == reservation.review_input_sha256
                and entry.execution == "complete"
                for entry in ledger.reservations
            ):
                return False
            else:
                pass
            assert budget is not None
            authorize_reservation(ledger, reservation, budget, follow_up)
            ledger.reservations.append(reservation)
        else:
            matches = [
                entry
                for entry in ledger.reservations
                if (entry.run_id, entry.run_attempt, entry.review_input_sha256)
                == (
                    reservation.run_id,
                    reservation.run_attempt,
                    reservation.review_input_sha256,
                )
            ]
            if len(matches) != 1:
                raise PilotError("The review reservation could not be located")
            else:
                pass
            matches[0].execution = execution
        body = json.dumps(
            {
                "message": f"Mark advisory review {reservation.run_id} as {execution}",
                "content": base64.b64encode(ledger.model_dump_json().encode()).decode(),
                "sha": sha,
                "branch": STATE_BRANCH,
            }
        )
        try:
            github_api(
                f"repos/{repository_name}/contents/{STATE_PATH}",
                body=body,
                method="PUT",
            )
            return True
        except PilotError as error:
            if attempt + 1 == LEDGER_WRITE_ATTEMPTS or not re.search(
                r"HTTP (409|422)", str(error)
            ):
                raise
            else:
                pass
    raise AssertionError("Unreachable ledger retry state")


def run_github_review() -> None:
    repository_name = os.environ["GITHUB_REPOSITORY"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository_name):
        raise PilotError("Invalid repository name")
    else:
        pass
    try:
        pull_request_number = int(os.environ["PULL_REQUEST_NUMBER"])
        run_attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    except (KeyError, ValueError):
        raise PilotError("Expected an integer PR number and run attempt") from None
    if pull_request_number < 1:
        raise PilotError("Pull request number must be positive")
    else:
        pass
    run_id = os.environ["GITHUB_RUN_ID"]
    automatic_review = os.environ.get("AUTOMATIC_REVIEW", "false") == "true"
    pull_request = fetch_pull_request(repository_name, pull_request_number)
    if pull_request.state != "open":
        raise PilotError("The pull request is not open")
    else:
        pass
    repository = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    fetch_revision(repository, pull_request.base.sha)
    fetch_revision(repository, pull_request.head.sha)
    output_directory = Path(os.environ["RUNNER_TEMP"]) / "docs-review"
    artifact_directory = output_directory / "artifact"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    offline = os.environ["OFFLINE_REVIEW"] == "true"
    if offline:
        model_settings = None
        budget = None
    else:
        try:
            model_settings = ModelSettings.model_validate_json(
                os.environ["DOCS_REVIEW_MODEL"]
            )
            budget = BudgetConfig.model_validate_json(os.environ["DOCS_REVIEW_BUDGET"])
        except (KeyError, ValueError):
            raise PilotError(
                "Invalid DOCS_REVIEW_MODEL or DOCS_REVIEW_BUDGET configuration"
            ) from None
        if model_settings.api_key_environment_variable != "DOCS_REVIEW_API_KEY":
            raise PilotError("The model must use DOCS_REVIEW_API_KEY")
        elif not os.environ.get("DOCS_REVIEW_API_KEY"):
            raise PilotError("The model API key is not configured")
        elif budget.call_cost_cents <= 0:
            raise PilotError("The estimated call cost must be positive")
        else:
            pass
    request = ReviewRequest(
        repository=repository,
        base_commit_sha=pull_request.base.sha,
        head_commit_sha=pull_request.head.sha,
        pull_request_number=pull_request_number,
        pull_request_text=pull_request.body or "",
        model_settings=model_settings,
        budget=budget,
        ledger_path=output_directory / "local.sqlite",
        max_bundle_bytes=MAX_INPUT_BYTES,
    )
    prepared = prepare_review(request)
    bundle = prepared.bundle
    if not bundle.reviewed_paths:
        if automatic_review:
            print("No documentation changes; review skipped.")
            return
        else:
            raise PilotError("No changed documentation files were found")
    elif not offline and len(render_bundle(bundle).encode()) > MAX_INPUT_BYTES:
        raise PilotError("Review input exceeds the configured byte limit")
    else:
        pass
    reservation = None
    if budget is not None:
        reservation = Reservation(
            pull_request_number=pull_request_number,
            head_commit_sha=pull_request.head.sha,
            review_input_sha256=prepared.input_hash,
            estimated_cost_cents=budget.call_cost_cents,
            date_utc=datetime.now(timezone.utc).date().isoformat(),
            run_id=run_id,
            run_attempt=run_attempt,
        )
        if not update_reservation(
            repository_name,
            reservation,
            "reserved",
            budget=budget,
            follow_up=automatic_review or os.environ["FOLLOW_UP"] == "true",
            automatic=automatic_review,
        ):
            print("Review input unchanged since the last complete run; skipped.")
            return
        else:
            pass
    else:
        pass
    try:
        result = run_review(request, None, SystemClock(), prepared=prepared)
    except Exception:
        if reservation is not None:
            try:
                update_reservation(repository_name, reservation, "incomplete")
            except PilotError as finish_error:
                LOGGER.warning(
                    f"Could not mark reservation incomplete after review error: {finish_error}"
                )
        else:
            pass
        raise
    if reservation is not None:
        update_reservation(repository_name, reservation, result.report.execution)
    else:
        pass
    current = fetch_pull_request(repository_name, pull_request_number)
    if current.state != "open" or current.head.sha != pull_request.head.sha:
        stale = True
    else:
        if current.base.sha != pull_request.base.sha:
            fetch_revision(repository, current.base.sha)
        else:
            pass
        current_merge_base = run_git(
            repository, ["merge-base", current.base.sha, current.head.sha]
        )
        if current_merge_base.returncode:
            raise PilotError("Could not compute the pull request merge base")
        else:
            pass
        stale = current_merge_base.stdout.strip() != bundle.merge_base_commit_sha
    artifact = ReviewArtifact(
        pull_request_number=pull_request_number,
        run_id=run_id,
        run_attempt=run_attempt,
        stale=stale,
        publishable=(
            not offline and not stale and result.report.execution == "complete"
        ),
        report=result.report,
    )
    (artifact_directory / "report.json").write_text(
        artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (artifact_directory / "report.md").write_text(
        render_artifact(artifact, offline=offline, automatic=automatic_review),
        encoding="utf-8",
    )
    if automatic_review and artifact.publishable:
        publish_report(repository_name, artifact)
    else:
        pass
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"execution={result.report.execution}\n")
        output.write(f"stale={str(stale).lower()}\n")


def main() -> int:
    try:
        run_github_review()
    except PilotError as error:
        print(str(error), file=sys.stderr)
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
