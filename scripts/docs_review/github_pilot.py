# SPDX-License-Identifier: Apache-2.0
"""Run an authorized GitHub documentation review without executing PR code."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from scripts.docs_review.github import PilotError as GitHubPilotError
from scripts.docs_review.github import (
    fetch_pull_request,
    fetch_revision,
    fetch_unit_test_checks,
)
from scripts.docs_review.models import (
    BudgetConfig,
    ModelSettings,
    ReviewArtifact,
    ReviewBundle,
    ReviewRequest,
    ReviewScope,
)
from scripts.docs_review.reporting import render_bundle, render_markdown
from scripts.docs_review.review import (
    SystemClock,
    assemble_report,
    prepare_review,
    run_review,
)

STATE_BRANCH = "docs-review-ledger"
STATE_PATH = "docs-review-ledger.json"
MAX_INPUT_BYTES = 100_000
MAX_REVIEWS_PER_PULL_REQUEST = 2
REVIEW_WORKFLOW_PATH = ".github/workflows/pr-advisory-review.yaml"
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


class GitHubContent(BaseModel):
    sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    content: str
    encoding: Literal["base64"]


def github_api(endpoint: str, *, body: str | None = None, method: str = "GET") -> str:
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


def merge_base_commit_sha(
    repository: Path, base_commit_sha: str, head_commit_sha: str
) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            "merge-base",
            base_commit_sha,
            head_commit_sha,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise PilotError("Could not compute the pull request merge base")
    else:
        merge_base = completed.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", merge_base):
            raise PilotError("The merge base is not a commit SHA")
        else:
            return merge_base


def load_reservations(repository_name: str) -> tuple[Reservations, str]:
    endpoint = f"repos/{repository_name}/contents/{STATE_PATH}?ref={STATE_BRANCH}"
    content = GitHubContent.model_validate_json(github_api(endpoint))
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
    day_runs = [
        entry
        for entry in reservations.reservations
        if entry.date_utc == reservation.date_utc
    ]
    matching_runs = [
        entry
        for entry in pull_request_runs
        if entry.review_input_sha256 == reservation.review_input_sha256
    ]
    if matching_runs and (
        not follow_up_authorized
        or any(entry.execution == "complete" for entry in matching_runs)
    ):
        raise PilotError("This exact review was already reserved; see its earlier run")
    elif len(pull_request_runs) >= MAX_REVIEWS_PER_PULL_REQUEST:
        raise PilotError("Pull request review limit reached")
    elif (
        pull_request_runs
        and not follow_up_authorized
        and any(
            entry.run_id != reservation.run_id
            or entry.run_attempt != reservation.run_attempt
            or entry.head_commit_sha != reservation.head_commit_sha
            for entry in pull_request_runs
        )
    ):
        raise PilotError("A follow-up review must be explicitly authorized")
    elif reservation.estimated_cost_cents > budget.max_cost_cents_per_call:
        raise PilotError("Estimated call cost exceeds the per-call budget")
    elif (
        sum(entry.estimated_cost_cents for entry in pull_request_runs)
        + reservation.estimated_cost_cents
        > budget.max_cost_cents_per_pull_request
    ):
        raise PilotError("Pull request budget exceeded")
    elif (
        sum(entry.estimated_cost_cents for entry in day_runs)
        + reservation.estimated_cost_cents
        > budget.max_cost_cents_per_day
    ):
        raise PilotError("Daily budget exceeded")
    else:
        pass


def has_complete_reservation(
    reservations: Reservations, pull_request_number: int, review_input_sha256: str
) -> bool:
    return any(
        entry.pull_request_number == pull_request_number
        and entry.review_input_sha256 == review_input_sha256
        and entry.execution == "complete"
        for entry in reservations.reservations
    )


def save_reservation(
    repository_name: str,
    reservations: Reservations,
    content_sha: str,
    reservation: Reservation,
) -> None:
    reservations.reservations.append(reservation)
    write_reservations(repository_name, reservations, content_sha)


def save_reservation_with_retry(
    repository_name: str,
    reservation: Reservation,
    budget: BudgetConfig,
    follow_up_authorized: bool,
) -> None:
    for attempt in range(LEDGER_WRITE_ATTEMPTS):
        reservations, content_sha = load_reservations(repository_name)
        authorize_reservation(reservations, reservation, budget, follow_up_authorized)
        reservations.reservations.append(reservation)
        try:
            write_reservations(repository_name, reservations, content_sha)
        except PilotError as error:
            if attempt + 1 == LEDGER_WRITE_ATTEMPTS or not re.search(
                r"HTTP (409|422)", str(error)
            ):
                raise
            else:
                pass
        else:
            return
    raise PilotError("The review ledger could not be updated after repeated conflicts")


def write_reservations(
    repository_name: str, reservations: Reservations, content_sha: str
) -> None:
    content = base64.b64encode(reservations.model_dump_json().encode()).decode()
    body = json.dumps(
        {
            "message": "Reserve an advisory documentation review",
            "content": content,
            "sha": content_sha,
            "branch": STATE_BRANCH,
        }
    )
    github_api(
        f"repos/{repository_name}/contents/{STATE_PATH}", body=body, method="PUT"
    )


def finish_reservation(
    repository_name: str,
    reservation: Reservation,
    execution: Literal["complete", "incomplete"],
) -> None:
    for attempt in range(LEDGER_WRITE_ATTEMPTS):
        reservations, content_sha = load_reservations(repository_name)
        matching = [
            entry
            for entry in reservations.reservations
            if entry.run_id == reservation.run_id
            and entry.run_attempt == reservation.run_attempt
            and entry.review_input_sha256 == reservation.review_input_sha256
        ]
        if len(matching) != 1:
            raise PilotError("The review reservation could not be located")
        else:
            matching[0].execution = execution
        try:
            write_reservations(repository_name, reservations, content_sha)
        except PilotError as error:
            if attempt + 1 == LEDGER_WRITE_ATTEMPTS or not re.search(
                r"HTTP (409|422)", str(error)
            ):
                raise
            else:
                pass
        else:
            return
    raise PilotError("The review ledger could not be updated after repeated conflicts")


def list_changed_paths(
    repository: Path, base_commit_sha: str, head_commit_sha: str
) -> list[str]:
    merge_base = merge_base_commit_sha(repository, base_commit_sha, head_commit_sha)
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            "diff",
            "--name-only",
            merge_base,
            head_commit_sha,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise PilotError("Could not list changed paths for the pull request")
    else:
        return [line for line in completed.stdout.splitlines() if line.strip()]


def detect_review_scopes(changed_paths: list[str]) -> list[ReviewScope]:
    scopes: list[ReviewScope] = []
    if any(
        path.startswith("docs/") or path in {"README.md", "CONTRIBUTING.md"}
        for path in changed_paths
    ):
        scopes.append("docs")
    else:
        pass
    if any(
        (path.startswith("tests/unit_test/") and path.endswith(".py"))
        or path in {"conftest.py", "tests/conftest.py"}
        for path in changed_paths
    ):
        scopes.append("unit_tests")
    else:
        pass
    return scopes


def resolve_review_scopes(
    repository: Path,
    base_commit_sha: str,
    head_commit_sha: str,
    *,
    automatic_review: bool,
) -> list[ReviewScope]:
    if automatic_review:
        return detect_review_scopes(
            list_changed_paths(repository, base_commit_sha, head_commit_sha)
        )
    else:
        scope_input = os.environ.get("REVIEW_SCOPE", "docs")
        if scope_input == "unit_tests":
            return ["unit_tests"]
        else:
            return ["docs"]


def model_settings_for_scope(
    review_scope: ReviewScope, offline: bool
) -> ModelSettings | None:
    if offline:
        return None
    else:
        pass
    if review_scope == "unit_tests":
        raw = (
            os.environ.get("UNIT_TEST_REVIEW_MODEL") or os.environ["DOCS_REVIEW_MODEL"]
        )
    else:
        raw = os.environ["DOCS_REVIEW_MODEL"]
    return ModelSettings.model_validate_json(raw)


def review_one_scope(
    repository_name: str,
    pull_request: PullRequest,
    pull_request_number: int,
    repository: Path,
    review_scope: ReviewScope,
    *,
    offline: bool,
    budget: BudgetConfig | None,
    run_id: str,
    run_attempt: int,
    automatic_review: bool,
    output_directory: Path,
) -> ReviewArtifact | None:
    include_paths = (
        [
            path.strip()
            for path in os.environ.get("UNIT_TEST_CONTEXT_PATHS", "").splitlines()
            if path.strip()
        ]
        if review_scope == "unit_tests"
        else []
    )
    checks = (
        fetch_unit_test_checks(repository_name, pull_request.head.sha)
        if review_scope == "unit_tests"
        else []
    )
    request = ReviewRequest(
        repository=repository,
        base_commit_sha=pull_request.base.sha,
        head_commit_sha=pull_request.head.sha,
        pull_request_number=pull_request_number,
        pull_request_text=pull_request.body or "",
        model_settings=model_settings_for_scope(review_scope, offline),
        budget=budget,
        ledger_path=output_directory / f"local-{review_scope}.sqlite",
        review_scope=review_scope,
        include_paths=include_paths,
        checks=checks,
    )
    prepared = prepare_review(request)
    bundle = prepared.bundle
    if review_scope == "docs" and not bundle.reviewed_paths:
        return None
    elif (
        review_scope == "unit_tests"
        and not bundle.unit_test_cases
        and not bundle.test_hygiene_notes
        and not bundle.coverage_limitations
    ):
        return None
    else:
        pass
    reservation: Reservation | None = None
    reservation_error = ""
    input_too_large = (
        not offline
        and bundle.needs_model
        and len(render_bundle(bundle).encode()) > MAX_INPUT_BYTES
    )
    if input_too_large:
        reservation_error = "Review input exceeds the byte limit. Select a smaller pull request or fewer context files."
    elif offline or not bundle.needs_model:
        reservation = None
    elif budget is None:
        reservation_error = "The review budget is not configured"
    else:
        try:
            reservations, _content_sha = load_reservations(repository_name)
        except PilotError as error:
            reservation_error = str(error)
        else:
            if automatic_review and has_complete_reservation(
                reservations, pull_request_number, prepared.input_hash
            ):
                reservation_error = (
                    "This input was reviewed previously; the earlier report "
                    "was not loaded in this run."
                )
            else:
                reservation = Reservation(
                    pull_request_number=pull_request_number,
                    head_commit_sha=pull_request.head.sha,
                    review_input_sha256=prepared.input_hash,
                    estimated_cost_cents=budget.call_cost_cents,
                    date_utc=datetime.now(timezone.utc).date().isoformat(),
                    run_id=run_id,
                    run_attempt=run_attempt,
                )
                try:
                    save_reservation_with_retry(
                        repository_name,
                        reservation,
                        budget,
                        os.environ["FOLLOW_UP"] == "true",
                    )
                except PilotError as error:
                    reservation = None
                    reservation_error = str(error)
                else:
                    pass
    if reservation_error:
        report = assemble_report(
            bundle=bundle,
            prompt_version=prepared.prompt_version,
            input_hash=prepared.input_hash,
            model_findings=None,
            model_failed=True,
            incomplete_reason=reservation_error,
            usage_cost_cents=0,
        )
    else:
        result = run_review(request, None, SystemClock(), prepared=prepared)
        report = result.report
    if reservation is not None:
        try:
            finish_reservation(repository_name, reservation, report.execution)
        except PilotError as error:
            report = report.model_copy(
                update={
                    "execution": "incomplete",
                    "outcome": "incomplete",
                    "incomplete_reason": "; ".join(
                        filter(None, [report.incomplete_reason, str(error)])
                    ),
                }
            )
        else:
            pass
    else:
        pass
    reviewed_merge_base = bundle.merge_base_commit_sha
    current = fetch_pull_request(repository_name, pull_request_number)
    if current.state != "open" or current.head.sha != pull_request.head.sha:
        stale = True
    else:
        current_merge_base = merge_base_commit_sha(
            repository, current.base.sha, current.head.sha
        )
        stale = current_merge_base != reviewed_merge_base
    publishable = not offline and not stale and report.execution == "complete"
    return ReviewArtifact(
        pull_request_number=pull_request_number,
        run_id=run_id,
        run_attempt=run_attempt,
        stale=stale,
        publishable=publishable,
        review_scope=review_scope,
        report=report,
    )


def run_github_review() -> None:
    repository_name = os.environ["GITHUB_REPOSITORY"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository_name):
        raise PilotError("Invalid repository name")
    else:
        pass
    pull_request_number = int(os.environ["PULL_REQUEST_NUMBER"])
    if pull_request_number < 1:
        raise PilotError("Pull request number must be positive")
    else:
        pass
    run_id = os.environ["GITHUB_RUN_ID"]
    run_attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    automatic_review = os.environ.get("AUTOMATIC_REVIEW", "false") == "true"
    pull_request = fetch_pull_request(repository_name, pull_request_number)
    if pull_request.state != "open":
        raise PilotError("The pull request is not open")
    else:
        pass
    repository = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    fetch_revision(repository, pull_request.base.sha)
    fetch_revision(repository, f"refs/pull/{pull_request_number}/head")
    output_directory = Path(os.environ["RUNNER_TEMP"]) / "docs-review"
    artifact_directory = output_directory / "artifact"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    offline = os.environ["OFFLINE_REVIEW"] == "true"
    if offline:
        budget = None
    else:
        budget = BudgetConfig.model_validate_json(os.environ["DOCS_REVIEW_BUDGET"])
        model_settings = model_settings_for_scope("docs", offline=False)
        if model_settings is None or model_settings.api_key_environment_variable != (
            "DOCS_REVIEW_API_KEY"
        ):
            raise PilotError("The model must use DOCS_REVIEW_API_KEY")
        elif not os.environ.get("DOCS_REVIEW_API_KEY"):
            raise PilotError("The model API key is not configured")
        elif budget.call_cost_cents <= 0:
            raise PilotError("The estimated call cost must be positive")
        else:
            pass
    review_scopes = resolve_review_scopes(
        repository,
        pull_request.base.sha,
        pull_request.head.sha,
        automatic_review=automatic_review,
    )
    if not review_scopes:
        if automatic_review:
            print("No in-scope changes; advisory review skipped.")
            return
        else:
            raise PilotError("Choose documentation or unit-test review scope")
    else:
        pass
    artifacts: list[ReviewArtifact] = []
    for review_scope in review_scopes:
        try:
            artifact = review_one_scope(
                repository_name,
                pull_request,
                pull_request_number,
                repository,
                review_scope,
                offline=offline,
                budget=budget,
                run_id=run_id,
                run_attempt=run_attempt,
                automatic_review=automatic_review,
                output_directory=output_directory,
            )
        except (
            PilotError,
            GitHubPilotError,
            ValidationError,
            RuntimeError,
            OSError,
        ) as error:
            error_message = (
                "Review configuration failed validation"
                if isinstance(error, ValidationError)
                else str(error)
            )
            incomplete_bundle = ReviewBundle(
                repository_identity=repository_name,
                base_commit_sha=pull_request.base.sha,
                head_commit_sha=pull_request.head.sha,
                merge_base_commit_sha="",
                policy_version="",
                pull_request_text=pull_request.body or "",
                comment_text="",
                evidence_text="",
                checks=[],
                files=[],
                reviewed_paths=[],
                added_documentation_text="",
                style_guide_limitation="",
                review_scope=review_scope,
                collection_limitations=[error_message],
            )
            report = assemble_report(incomplete_bundle, "", "", None, True, "", 0)
            artifact = ReviewArtifact(
                pull_request_number=pull_request_number,
                run_id=run_id,
                run_attempt=run_attempt,
                stale=False,
                publishable=False,
                review_scope=review_scope,
                report=report,
            )
        else:
            pass
        if artifact is None:
            continue
        else:
            artifacts.append(artifact)
    if not artifacts:
        if automatic_review:
            print("No in-scope changes produced a report; skipped.")
            return
        else:
            raise PilotError("No changed files matched the selected review scope")
    else:
        pass
    if automatic_review:
        current = fetch_pull_request(repository_name, pull_request_number)
        if (
            current.state != "open"
            or current.head.sha != pull_request.head.sha
            or current.base.sha != pull_request.base.sha
        ):
            for artifact in artifacts:
                artifact.stale = True
                artifact.publishable = False
        else:
            pass
        if not any(artifact.stale for artifact in artifacts):
            repository_url = f'{os.environ["GITHUB_SERVER_URL"]}/{repository_name}'
            from scripts.docs_review.github_comment import publish_reports

            published = publish_reports(
                repository_name,
                repository_url,
                run_id,
                run_attempt,
                artifacts,
                expected_base_sha=pull_request.base.sha,
                expected_head_sha=pull_request.head.sha,
            )
            if published is False:
                for artifact in artifacts:
                    artifact.stale = True
                    artifact.publishable = False
            else:
                pass
        else:
            pass
    else:
        pass
    (artifact_directory / "report.json").write_text(
        json.dumps(
            [artifact.model_dump(mode="json") for artifact in artifacts], indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    if automatic_review:
        publication_note = (
            "Findings post automatically to this pull request when complete."
        )
    else:
        publication_note = "Download the report artifact to inspect numbered findings."
    notices = [
        f"Pull request #{pull_request_number}",
        f"Review run {run_id}, attempt {run_attempt}.",
        publication_note,
    ]
    if offline:
        notices.append("Collection-only run without a model call.")
    else:
        pass
    markdown_sections = []
    for artifact in artifacts:
        if artifact.stale:
            notices.append(
                f"Stale {artifact.review_scope} report. The pull request changed "
                "or closed during review."
            )
        else:
            pass
        markdown_sections.append(
            f"## {artifact.review_scope}\n\n"
            + render_markdown(artifact.report, number_findings=True)
        )
    (artifact_directory / "report.md").write_text(
        "\n\n".join(notices) + "\n\n" + "\n\n".join(markdown_sections),
        encoding="utf-8",
    )
    execution = (
        "complete"
        if all(artifact.report.execution == "complete" for artifact in artifacts)
        else "incomplete"
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"execution={execution}\n")
        output.write(
            f"stale={str(any(artifact.stale for artifact in artifacts)).lower()}\n"
        )
    print(
        f"Advisory review: scopes={[artifact.review_scope for artifact in artifacts]}, "
        f"execution={execution}."
    )


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
