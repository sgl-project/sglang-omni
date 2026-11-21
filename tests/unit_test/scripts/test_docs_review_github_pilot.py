# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the advisory review GitHub integration."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.docs_review import github, github_comment, github_pilot
from scripts.docs_review.github_pilot import (
    PilotError,
    Reservation,
    Reservations,
    authorize_reservation,
)
from scripts.docs_review.models import BudgetConfig, Finding, Report, ReviewArtifact
from scripts.docs_review.transport import OpenAICompatibleReviewer, ReviewerError


def test_pull_request_keeps_api_base_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def api(endpoint: str, **kwargs: object) -> str:
        assert endpoint.endswith("/pulls/1")
        return json.dumps(
            {
                "state": "open",
                "body": "",
                "base": {"sha": "a" * 40, "ref": "main"},
                "head": {"sha": "b" * 40},
            }
        )

    monkeypatch.setattr(github, "github_api", api)
    pull_request = github_pilot.fetch_pull_request("owner/repo", 1)
    assert pull_request.base.sha == "a" * 40
    assert pull_request.head.sha == "b" * 40


def test_reservations_bound_duplicates_followups_and_daily_spend() -> None:
    budget = BudgetConfig(
        max_cost_cents_per_call=25,
        max_cost_cents_per_pull_request=50,
        max_cost_cents_per_day=50,
        call_cost_cents=25,
    )
    first = Reservation(
        pull_request_number=1,
        head_commit_sha="a" * 40,
        review_input_sha256="b" * 64,
        estimated_cost_cents=25,
        date_utc="2026-11-06",
        run_id="1",
    )
    second = first.model_copy(update={"review_input_sha256": "c" * 64, "run_id": "2"})
    other = first.model_copy(update={"pull_request_number": 2, "run_id": "3"})
    reservations = Reservations(reservations=[first])
    with pytest.raises(PilotError, match="already reserved"):
        authorize_reservation(reservations, first, budget, False)
    authorize_reservation(reservations, first, budget, True)
    completed = first.model_copy(update={"execution": "complete"})
    with pytest.raises(PilotError, match="already reserved"):
        authorize_reservation(
            Reservations(reservations=[completed]), first, budget, True
        )
    with pytest.raises(PilotError, match="follow-up"):
        authorize_reservation(reservations, second, budget, False)
    authorize_reservation(reservations, second, budget, True)
    reservations.reservations.append(second)
    with pytest.raises(PilotError, match="Daily budget"):
        authorize_reservation(reservations, other, budget, False)
    third = first.model_copy(update={"review_input_sha256": "d" * 64, "run_id": "4"})
    with pytest.raises(PilotError, match="review limit"):
        authorize_reservation(reservations, third, budget, True)


def test_two_scopes_in_one_run_respect_the_same_budget() -> None:
    budget = BudgetConfig(
        max_cost_cents_per_call=25,
        max_cost_cents_per_pull_request=50,
        max_cost_cents_per_day=50,
        call_cost_cents=25,
    )
    first = Reservation(
        pull_request_number=1,
        head_commit_sha="a" * 40,
        review_input_sha256="b" * 64,
        estimated_cost_cents=25,
        date_utc="2026-11-16",
        run_id="123",
        run_attempt=1,
        execution="complete",
    )
    second = first.model_copy(update={"review_input_sha256": "c" * 64})
    ledger = Reservations(reservations=[first])
    authorize_reservation(ledger, second, budget, False)
    ledger.reservations.append(second)
    third = first.model_copy(update={"review_input_sha256": "d" * 64})
    with pytest.raises(PilotError, match="review limit"):
        authorize_reservation(ledger, third, budget, False)


def test_clean_review_updates_and_clears_old_sticky_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []
    old_comment = github_comment.ADVISORY_REVIEW_MARKER + "\nprevious finding"

    def api(endpoint: str, *, body: str | None = None, method: str = "GET") -> str:
        if method == "GET":
            return (
                json.dumps(
                    [
                        {
                            "id": 10,
                            "body": old_comment,
                            "user": {"login": "github-actions[bot]"},
                        }
                    ]
                )
                if endpoint.endswith("&page=1")
                else "[]"
            )
        else:
            assert method == "PATCH" and body is not None
            writes.append(json.loads(body)["body"])
            return "{}"

    monkeypatch.setattr(github_comment, "github_api", api)
    report = Report.model_construct(
        execution="complete",
        head_commit_sha="a" * 40,
        findings=[],
        test_hygiene_notes=[],
        review_scope="unit_tests",
    )
    artifact = ReviewArtifact.model_construct(
        pull_request_number=1,
        run_id="123",
        run_attempt=1,
        stale=False,
        publishable=True,
        review_scope="unit_tests",
        report=report,
    )
    github_comment.publish_reports(
        "owner/repo", "https://github.com/owner/repo", "123", 1, [artifact]
    )
    assert len(writes) == 1
    assert "previous finding" not in writes[0]
    assert "unit-test" in writes[0].lower()


def test_incomplete_scope_does_not_relabel_old_findings_as_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []
    previous = (
        github_comment.ADVISORY_REVIEW_MARKER
        + f"\n<!-- advisory-head:{'a' * 40} -->\n"
        + "<!-- advisory-scope:unit_tests:start -->\nold unit finding\n"
        + "<!-- advisory-scope:unit_tests:end -->"
    )

    def api(endpoint: str, *, body: str | None = None, method: str = "GET") -> str:
        if method == "GET":
            return (
                json.dumps(
                    [
                        {
                            "id": 10,
                            "body": previous,
                            "user": {"login": "github-actions[bot]"},
                        }
                    ]
                )
                if endpoint.endswith("&page=1")
                else "[]"
            )
        else:
            assert body is not None
            writes.append(json.loads(body)["body"])
            return "{}"

    monkeypatch.setattr(github_comment, "github_api", api)
    artifacts = []
    for scope, execution in (("docs", "complete"), ("unit_tests", "incomplete")):
        report = Report.model_construct(
            execution=execution,
            head_commit_sha="b" * 40,
            findings=[],
            test_hygiene_notes=[],
            review_scope=scope,
        )
        artifacts.append(
            ReviewArtifact.model_construct(
                pull_request_number=1,
                run_id="123",
                run_attempt=1,
                stale=False,
                publishable=execution == "complete",
                review_scope=scope,
                report=report,
            )
        )
    github_comment.publish_reports(
        "owner/repo", "https://github.com/owner/repo", "123", 1, artifacts
    )
    assert len(writes) == 1
    assert "old unit finding" not in writes[0]
    assert "Review incomplete" in writes[0]
    assert "documentation review" in writes[0]


def test_comment_is_not_written_if_head_changes_before_the_api_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []

    def api(endpoint: str, *, body: str | None = None, method: str = "GET") -> str:
        if endpoint.endswith("/pulls/1"):
            return json.dumps(
                {"state": "open", "base": {"sha": "a" * 40}, "head": {"sha": "c" * 40}}
            )
        elif method == "GET":
            return "[]"
        else:
            writes.append(endpoint)
            return "{}"

    monkeypatch.setattr(github_comment, "github_api", api)
    report = Report.model_construct(
        execution="complete",
        head_commit_sha="b" * 40,
        findings=[],
        test_hygiene_notes=[],
        review_scope="docs",
    )
    artifact = ReviewArtifact.model_construct(
        pull_request_number=1,
        run_id="123",
        run_attempt=1,
        stale=False,
        publishable=True,
        review_scope="docs",
        report=report,
    )
    published = github_comment.publish_reports(
        "owner/repo",
        "https://github.com/owner/repo",
        "123",
        1,
        [artifact],
        expected_base_sha="a" * 40,
        expected_head_sha="b" * 40,
    )
    assert published is False
    assert writes == []


def test_older_run_does_not_overwrite_a_newer_same_revision_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []
    previous = (
        github_comment.ADVISORY_REVIEW_MARKER
        + "\n<!-- advisory-run:124:1 -->\nnewer finding"
    )

    def api(endpoint: str, *, body: str | None = None, method: str = "GET") -> str:
        if method == "GET":
            return (
                json.dumps(
                    [
                        {
                            "id": 10,
                            "body": previous,
                            "user": {"login": "github-actions[bot]"},
                        }
                    ]
                )
                if endpoint.endswith("&page=1")
                else "[]"
            )
        else:
            writes.append(endpoint)
            return "{}"

    monkeypatch.setattr(github_comment, "github_api", api)
    report = Report.model_construct(
        execution="complete",
        head_commit_sha="b" * 40,
        findings=[],
        test_hygiene_notes=[],
        review_scope="docs",
    )
    artifact = ReviewArtifact.model_construct(
        pull_request_number=1,
        run_id="123",
        run_attempt=1,
        stale=False,
        publishable=True,
        review_scope="docs",
        report=report,
    )
    assert not github_comment.publish_reports(
        "owner/repo", "https://github.com/owner/repo", "123", 1, [artifact]
    )
    assert writes == []


def test_comment_marker_from_another_user_is_not_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[tuple[str, str]] = []
    forged = github_comment.ADVISORY_REVIEW_MARKER + "\n@all forged finding"

    def api(endpoint: str, *, body: str | None = None, method: str = "GET") -> str:
        if method == "GET":
            return (
                json.dumps(
                    [{"id": 10, "body": forged, "user": {"login": "contributor"}}]
                )
                if endpoint.endswith("&page=1")
                else "[]"
            )
        else:
            assert body is not None
            writes.append((method, json.loads(body)["body"]))
            return "{}"

    monkeypatch.setattr(github_comment, "github_api", api)
    report = Report.model_construct(
        execution="complete",
        head_commit_sha="b" * 40,
        findings=[],
        test_hygiene_notes=[],
        review_scope="docs",
    )
    artifact = ReviewArtifact.model_construct(
        pull_request_number=1,
        run_id="123",
        run_attempt=1,
        stale=False,
        publishable=True,
        review_scope="docs",
        report=report,
    )
    github_comment.publish_reports(
        "owner/repo", "https://github.com/owner/repo", "123", 1, [artifact]
    )
    assert len(writes) == 1
    assert writes[0][0] == "POST"
    assert "forged finding" not in writes[0][1]


@pytest.mark.parametrize("incomplete_scope", ["docs", "unit_tests"])
def test_overall_execution_reflects_each_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, incomplete_scope: str
) -> None:
    base, head = configure_run(tmp_path, monkeypatch, offline=False)
    monkeypatch.setenv("AUTOMATIC_REVIEW", "true")
    pull_request = github_pilot.PullRequest(
        state="open", body="", base={"sha": base}, head={"sha": head}
    )
    monkeypatch.setattr(
        github_pilot, "fetch_pull_request", lambda name, number: pull_request
    )
    monkeypatch.setattr(
        github_pilot,
        "resolve_review_scopes",
        lambda *args, **kwargs: ["docs", "unit_tests"],
    )

    def review_scope(*args: object, **kwargs: object) -> ReviewArtifact:
        scope = args[4]
        execution = "incomplete" if scope == incomplete_scope else "complete"
        report = Report.model_construct(
            execution=execution,
            outcome="incomplete" if execution == "incomplete" else "no_findings",
            head_commit_sha=head,
            base_commit_sha=base,
            findings=[],
            test_hygiene_notes=[],
            review_scope=scope,
        )
        return ReviewArtifact.model_construct(
            pull_request_number=1,
            run_id="123",
            run_attempt=2,
            stale=False,
            publishable=execution == "complete",
            review_scope=scope,
            report=report,
        )

    monkeypatch.setattr(github_pilot, "review_one_scope", review_scope)
    monkeypatch.setattr(
        github_pilot, "render_markdown", lambda report, **kwargs: report.execution
    )
    monkeypatch.setattr(github_comment, "publish_reports", lambda *args, **kwargs: None)
    github_pilot.run_github_review()
    assert "execution=incomplete" in (tmp_path / "outputs").read_text()


@pytest.mark.parametrize("allow_both", [True, False])
def test_mixed_scope_reviews_reserve_calls_and_report_budget_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_both: bool
) -> None:
    base, _head = configure_run(tmp_path, monkeypatch, offline=False)
    repository = tmp_path / "repository"
    (repository / "sglang_omni").mkdir()
    (repository / "sglang_omni" / "api.py").write_text(
        "def speak():\n    return b'RIFF'\n"
    )
    test_directory = repository / "tests" / "unit_test"
    test_directory.mkdir(parents=True)
    (test_directory / "test_audio.py").write_text(
        "from sglang_omni.api import speak\n\n"
        "def test_audio():\n    assert speak() == b'RIFF'\n"
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "Add unit test"], check=True
    )
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    pull_request = github_pilot.PullRequest(
        state="open", body="", base={"sha": base}, head={"sha": head}
    )
    monkeypatch.setenv("AUTOMATIC_REVIEW", "true")
    monkeypatch.setattr(github_pilot, "fetch_pull_request", lambda *args: pull_request)
    monkeypatch.setattr(github_pilot, "fetch_unit_test_checks", lambda *args: [])
    monkeypatch.setattr(github_comment, "publish_reports", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        OpenAICompatibleReviewer,
        "review",
        lambda reviewer, bundle: (
            '{"findings": []}' if "Changed test cases" in bundle else "[]"
        ),
    )
    ledger = Reservations()
    monkeypatch.setattr(
        github_pilot, "load_reservations", lambda name: (ledger, "a" * 40)
    )
    monkeypatch.setattr(github_pilot, "write_reservations", lambda *args: None)
    if not allow_both:
        budget = json.loads(os.environ["DOCS_REVIEW_BUDGET"])
        budget["max_cost_cents_per_pull_request"] = 25
        monkeypatch.setenv("DOCS_REVIEW_BUDGET", json.dumps(budget))
    else:
        pass
    github_pilot.run_github_review()
    reports = json.loads(
        (tmp_path / "docs-review" / "artifact" / "report.json").read_text()
    )
    assert [report["review_scope"] for report in reports] == ["docs", "unit_tests"]
    assert reports[0]["report"]["execution"] == "complete"
    assert reports[1]["report"]["execution"] == (
        "complete" if allow_both else "incomplete"
    )
    if not allow_both:
        assert "budget exceeded" in reports[1]["report"]["incomplete_reason"]
    else:
        pass
    assert len(ledger.reservations) == (2 if allow_both else 1)


def test_new_revision_during_second_scope_cannot_replace_the_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, head = configure_run(tmp_path, monkeypatch, offline=False)
    monkeypatch.setenv("AUTOMATIC_REVIEW", "true")
    initial = github_pilot.PullRequest(
        state="open", body="", base={"sha": base}, head={"sha": head}
    )
    changed = initial.model_copy(update={"head": github_pilot.Revision(sha="f" * 40)})
    monkeypatch.setattr(github_pilot, "fetch_pull_request", lambda *args: initial)
    monkeypatch.setattr(
        github_pilot,
        "resolve_review_scopes",
        lambda *args, **kwargs: ["docs", "unit_tests"],
    )
    publications: list[str] = []
    monkeypatch.setattr(
        github_comment,
        "publish_reports",
        lambda *args, **kwargs: publications.append("published"),
    )

    def review_scope(*args: object, **kwargs: object) -> ReviewArtifact:
        scope = args[4]
        if scope == "unit_tests":
            monkeypatch.setattr(
                github_pilot, "fetch_pull_request", lambda *args: changed
            )
        else:
            pass
        report = Report.model_construct(
            execution="complete",
            outcome="no_findings",
            head_commit_sha=head,
            base_commit_sha=base,
            findings=[],
            test_hygiene_notes=[],
            review_scope=scope,
        )
        return ReviewArtifact.model_construct(
            pull_request_number=1,
            run_id="123",
            run_attempt=2,
            stale=False,
            publishable=True,
            review_scope=scope,
            report=report,
        )

    monkeypatch.setattr(github_pilot, "review_one_scope", review_scope)
    monkeypatch.setattr(
        github_pilot, "render_markdown", lambda report, **kwargs: report.execution
    )
    github_pilot.run_github_review()
    assert publications == []
    assert "stale=true" in (tmp_path / "outputs").read_text()


def test_scope_operational_error_does_not_discard_the_other_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, head = configure_run(tmp_path, monkeypatch, offline=False)
    monkeypatch.setenv("AUTOMATIC_REVIEW", "true")
    pull_request = github_pilot.PullRequest(
        state="open", body="", base={"sha": base}, head={"sha": head}
    )
    monkeypatch.setattr(github_pilot, "fetch_pull_request", lambda *args: pull_request)
    monkeypatch.setattr(
        github_pilot,
        "resolve_review_scopes",
        lambda *args, **kwargs: ["docs", "unit_tests"],
    )
    monkeypatch.setattr(github_comment, "publish_reports", lambda *args, **kwargs: True)

    def review_scope(*args: object, **kwargs: object) -> ReviewArtifact:
        if args[4] == "docs":
            raise PilotError("The docs input could not be collected")
        else:
            report = Report.model_construct(
                execution="complete",
                outcome="no_findings",
                head_commit_sha=head,
                base_commit_sha=base,
                findings=[],
                test_hygiene_notes=[],
                review_scope="unit_tests",
            )
            return ReviewArtifact.model_construct(
                pull_request_number=1,
                run_id="123",
                run_attempt=2,
                stale=False,
                publishable=True,
                review_scope="unit_tests",
                report=report,
            )

    monkeypatch.setattr(github_pilot, "review_one_scope", review_scope)
    monkeypatch.setattr(
        github_pilot, "render_markdown", lambda report, **kwargs: report.execution
    )
    github_pilot.run_github_review()
    reports = json.loads(
        (tmp_path / "docs-review" / "artifact" / "report.json").read_text()
    )
    assert [report["report"]["execution"] for report in reports] == [
        "incomplete",
        "complete",
    ]
    assert (
        "docs input could not be collected" in reports[0]["report"]["incomplete_reason"]
    )
    assert "execution=incomplete" in (tmp_path / "outputs").read_text()


def create_repository(root: Path) -> tuple[str, str]:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Review Test"], check=True
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "review@example.com"],
        check=True,
    )
    for name, value in (("core.hooksPath", "/dev/null"), ("commit.gpgsign", "false")):
        subprocess.run(["git", "-C", str(root), "config", name, value], check=True)
    (root / "README.md").write_text("Base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "Base"], check=True)
    base = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    (root / "docs" / "api_reference").mkdir(parents=True)
    (root / "docs" / "api_reference" / "demo.md").write_text(
        "# API\nThe response is WAV.\n", encoding="utf-8"
    )
    (root / "demo.py").write_text('"""Return raw PCM."""\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "Head"], check=True)
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    return base, head


def configure_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: bool
) -> tuple[str, str]:
    repository = tmp_path / "repository"
    base, head = create_repository(repository)
    variables = {
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_SERVER_URL": "https://github.com",
        "PULL_REQUEST_NUMBER": "1",
        "GITHUB_WORKSPACE": str(repository),
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "AUTOMATIC_REVIEW": "false",
        "OFFLINE_REVIEW": str(offline).lower(),
        "FOLLOW_UP": "false",
        "DOCS_REVIEW_API_KEY": "test-only-key",
        "DOCS_REVIEW_MODEL": json.dumps(
            {
                "model_name": "gpt-6.1-terra",
                "endpoint_url": "https://example.invalid/v1/chat/completions",
                "max_output_tokens": 8192,
                "api_key_environment_variable": "DOCS_REVIEW_API_KEY",
            }
        ),
        "DOCS_REVIEW_BUDGET": json.dumps(
            {
                "max_cost_cents_per_call": 25,
                "max_cost_cents_per_pull_request": 50,
                "max_cost_cents_per_day": 200,
                "call_cost_cents": 25,
            }
        ),
    }
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        github_pilot, "fetch_revision", lambda repository, revision: None
    )
    return base, head


def test_offline_report_contains_results_and_excludes_full_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, head = configure_run(tmp_path, monkeypatch, offline=True)
    initial = github_pilot.PullRequest(
        state="open", body="Expected answer", base={"sha": base}, head={"sha": head}
    )
    updated = initial.model_copy(update={"head": github_pilot.Revision(sha="f" * 40)})
    responses = iter([initial, updated])
    monkeypatch.setattr(
        github_pilot, "fetch_pull_request", lambda name, number: next(responses)
    )
    github_pilot.run_github_review()
    artifact = tmp_path / "docs-review" / "artifact"
    assert {file.name for file in artifact.iterdir()} == {"report.json", "report.md"}
    reports = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    report = reports[0]
    assert report["stale"] is True
    assert (report["run_id"], report["run_attempt"]) == ("123", 2)
    assert report["report"]["execution"] == "incomplete"
    assert report["publishable"] is False
    assert "input_bundle" not in report
    markdown = (artifact / "report.md").read_text(encoding="utf-8")
    assert "stale" in markdown.lower()
    assert "collection-only" in markdown.lower()
    assert "Expected answer" not in markdown
    assert "stale=true" in (tmp_path / "outputs").read_text(encoding="utf-8")


def test_failed_model_call_is_reserved_before_invocation_and_not_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, head = configure_run(tmp_path, monkeypatch, offline=False)
    pull_request = github_pilot.PullRequest(
        state="open", body="Expected answer", base={"sha": base}, head={"sha": head}
    )
    monkeypatch.setattr(
        github_pilot, "fetch_pull_request", lambda name, number: pull_request
    )
    reservations = Reservations()
    monkeypatch.setattr(
        github_pilot, "load_reservations", lambda name: (reservations, "a" * 40)
    )

    def save_reservation(
        name: str, ledger: Reservations, sha: str, reservation: Reservation
    ) -> None:
        assert sha == "a" * 40
        ledger.reservations.append(reservation)

    def fail_model(reviewer: OpenAICompatibleReviewer, bundle_text: str) -> str:
        assert len(reservations.reservations) == 1
        raise ReviewerError("Model unavailable")

    def reserve_with_retry(
        name: str,
        reservation: Reservation,
        budget: BudgetConfig,
        follow_up_authorized: bool,
    ) -> None:
        save_reservation(name, reservations, "a" * 40, reservation)

    monkeypatch.setattr(github_pilot, "save_reservation_with_retry", reserve_with_retry)
    monkeypatch.setattr(
        github_pilot, "write_reservations", lambda name, ledger, sha: None
    )
    monkeypatch.setattr(OpenAICompatibleReviewer, "review", fail_model)
    github_pilot.run_github_review()
    artifact = tmp_path / "docs-review" / "artifact"
    reports = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    report = reports[0]
    assert report["report"]["execution"] == "incomplete"
    assert reservations.reservations[0].execution == "incomplete"
    budget = BudgetConfig(
        max_cost_cents_per_call=25,
        max_cost_cents_per_pull_request=50,
        max_cost_cents_per_day=200,
        call_cost_cents=25,
    )
    duplicate = reservations.reservations[0].model_copy(update={"run_id": "124"})
    with pytest.raises(PilotError, match="already reserved"):
        authorize_reservation(
            reservations, duplicate, budget, follow_up_authorized=False
        )


def test_automatic_review_posts_sticky_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, head = configure_run(tmp_path, monkeypatch, offline=False)
    monkeypatch.setenv("AUTOMATIC_REVIEW", "true")
    pull_request = github_pilot.PullRequest(
        state="open", body="", base={"sha": base}, head={"sha": head}
    )
    monkeypatch.setattr(
        github_pilot, "fetch_pull_request", lambda name, number: pull_request
    )
    reservations = Reservations()
    monkeypatch.setattr(
        github_pilot, "load_reservations", lambda name: (reservations, "a" * 40)
    )
    monkeypatch.setattr(
        github_pilot, "write_reservations", lambda name, ledger, sha: None
    )

    def reserve_with_retry(
        name: str,
        reservation: Reservation,
        budget: BudgetConfig,
        follow_up_authorized: bool,
    ) -> None:
        reservations.reservations.append(reservation)

    monkeypatch.setattr(github_pilot, "save_reservation_with_retry", reserve_with_retry)
    model_finding = Finding(
        rule_id="API_CONTRACT",
        file_path="docs/api_reference/demo.md",
        line_start=2,
        line_end=2,
        quote="The response is WAV.",
        claim="documentation says WAV, but code returns PCM.",
        evidence_file_path="demo.py",
        evidence_line_start=1,
        evidence_line_end=1,
        evidence_quote="Return raw PCM.",
        impact="Clients decode the wrong format.",
        suggested_fix="Document PCM.",
        change_relation="introduced",
        certainty="supported",
    )
    monkeypatch.setattr(
        OpenAICompatibleReviewer,
        "review",
        lambda reviewer, bundle: json.dumps([model_finding.model_dump()]),
    )
    posted: list[tuple[str, str]] = []

    def api(endpoint: str, **kwargs: object) -> str:
        body = kwargs.get("body")
        method = kwargs.get("method", "GET")
        if endpoint.endswith("/pulls/1"):
            return json.dumps(
                {
                    "state": "open",
                    "body": "",
                    "base": {"sha": base, "ref": "main"},
                    "head": {"sha": head},
                }
            )
        elif endpoint.endswith("/issues/1/comments?per_page=100&page=1"):
            return "[]"
        elif method == "POST" and endpoint.endswith("/issues/1/comments"):
            assert body is not None
            posted.append((method, json.loads(str(body))["body"]))
            return json.dumps({"id": 42})
        else:
            raise AssertionError(endpoint)

    monkeypatch.setattr(github_comment, "github_api", api)
    github_pilot.run_github_review()
    assert len(posted) == 1
    assert github_comment.ADVISORY_REVIEW_MARKER in posted[0][1]
    assert "Document PCM." in posted[0][1]
    assert head in posted[0][1]


def test_safe_text_strips_mentions() -> None:
    assert "@" not in github_comment.safe_text("@team [click](https://example.invalid)")


@pytest.mark.parametrize("publish_incomplete", [True, False])
def test_incomplete_only_review_replaces_prior_same_revision_status(
    monkeypatch: pytest.MonkeyPatch, publish_incomplete: bool
) -> None:
    writes: list[str] = []
    head = "a" * 40
    previous = (
        github_comment.ADVISORY_REVIEW_MARKER
        + f"\n<!-- advisory-head:{head} -->\n"
        + "<!-- advisory-scope:docs:start -->\nprevious documentation finding\n"
        + "<!-- advisory-scope:docs:end -->\n"
        + "<!-- advisory-scope:unit_tests:start -->\nNo findings.\n"
        + "<!-- advisory-scope:unit_tests:end -->"
    )

    def api(endpoint: str, *, body: str | None = None, method: str = "GET") -> str:
        if method == "GET":
            return (
                json.dumps(
                    [
                        {
                            "id": 10,
                            "body": previous,
                            "user": {"login": "github-actions[bot]"},
                        }
                    ]
                )
                if endpoint.endswith("&page=1")
                else "[]"
            )
        else:
            assert method == "PATCH" and body is not None
            writes.append(json.loads(body)["body"])
            return "{}"

    monkeypatch.setattr(github_comment, "github_api", api)
    report = Report.model_construct(
        execution="incomplete",
        head_commit_sha=head,
        findings=[],
        test_hygiene_notes=[],
        review_scope="unit_tests",
    )
    artifact = ReviewArtifact.model_construct(
        pull_request_number=1,
        run_id="123",
        run_attempt=1,
        stale=False,
        publishable=False,
        review_scope="unit_tests",
        report=report,
    )
    github_comment.publish_reports(
        "owner/repo",
        "https://github.com/owner/repo",
        "123",
        1,
        [artifact],
        publish_incomplete=publish_incomplete,
    )
    if publish_incomplete:
        assert len(writes) == 1
        assert "Review incomplete" in writes[0]
        assert "No findings." not in writes[0]
        assert "previous documentation finding" in writes[0]
    else:
        assert not writes
