# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the advisory review GitHub integration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.docs_review import github_comment, github_pilot
from scripts.docs_review.github_pilot import (
    PilotError,
    Reservation,
    Reservations,
    authorize_reservation,
)
from scripts.docs_review.models import BudgetConfig, Finding
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

    monkeypatch.setattr(github_pilot, "github_api", api)
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
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
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
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
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