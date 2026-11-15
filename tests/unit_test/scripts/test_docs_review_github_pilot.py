# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the advisory review GitHub integration."""

from __future__ import annotations

import base64
import json
import subprocess
import urllib.error
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request

import pytest

from scripts.docs_review import github_comment, github_pilot
from scripts.docs_review.github_pilot import (
    PilotError,
    PullRequest,
    Reservation,
    Reservations,
    Revision,
)
from scripts.docs_review.models import Finding

REAL_SUBPROCESS_RUN = subprocess.run
LEDGER_ENDPOINT = "repos/owner/repo/contents/docs-review-ledger.json"
COMMENTS_ENDPOINT = "repos/owner/repo/issues/1/comments"
COMMENT_ID = 42


class FakeGitHub:
    """Answers gh CLI calls and records every write to GitHub."""

    def __init__(self, pull_requests: list[PullRequest]) -> None:
        self.pull_requests = pull_requests
        self.ledger = Reservations()
        self.ledger_revision = 1
        self.ledger_conflicts = 0
        self.issue_comments: list[dict[str, object]] = []
        self.writes: list[tuple[str, str, str]] = []

    def ledger_sha(self) -> str:
        return f"{self.ledger_revision:040x}"

    def run(
        self,
        arguments: list[str],
        *,
        input: str | None = None,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if arguments[0] != "gh":
            return REAL_SUBPROCESS_RUN(
                arguments,
                input=input,
                capture_output=capture_output,
                text=text,
                check=check,
            )
        else:
            assert arguments[1:3] == ["api", "--method"]
            method, endpoint = arguments[3], arguments[4]
            if method == "PUT" and self.ledger_conflicts:
                self.ledger_conflicts -= 1
                return subprocess.CompletedProcess(
                    arguments, 1, "", "gh: Conflict (HTTP 409)"
                )
            else:
                return subprocess.CompletedProcess(
                    arguments, 0, self.api(method, endpoint, input), ""
                )

    def api(self, method: str, endpoint: str, body: str | None) -> str:
        if method == "GET" and endpoint == "repos/owner/repo/pulls/1":
            pull_request = (
                self.pull_requests.pop(0)
                if len(self.pull_requests) > 1
                else self.pull_requests[0]
            )
            return pull_request.model_dump_json()
        elif (
            method == "GET" and endpoint == f"{LEDGER_ENDPOINT}?ref=docs-review-ledger"
        ):
            content = base64.b64encode(self.ledger.model_dump_json().encode()).decode()
            return json.dumps(
                {"sha": self.ledger_sha(), "content": content, "encoding": "base64"}
            )
        elif method == "PUT" and endpoint == LEDGER_ENDPOINT:
            assert body is not None
            update = json.loads(body)
            assert update["branch"] == "docs-review-ledger"
            assert update["sha"] == self.ledger_sha()
            self.ledger = Reservations.model_validate_json(
                base64.b64decode(update["content"])
            )
            self.ledger_revision += 1
            self.writes.append((method, endpoint, body))
            return "{}"
        elif method == "GET" and endpoint.startswith(f"{COMMENTS_ENDPOINT}?"):
            first_page = endpoint.endswith("&page=1")
            return json.dumps(self.issue_comments if first_page else [])
        elif (method, endpoint) in {
            ("POST", COMMENTS_ENDPOINT),
            ("PATCH", f"repos/owner/repo/issues/comments/{COMMENT_ID}"),
        }:
            assert body is not None
            comment = json.loads(body)["body"]
            self.writes.append((method, endpoint, comment))
            self.issue_comments = [{"id": COMMENT_ID, "body": comment}]
            return json.dumps({"id": COMMENT_ID})
        else:
            raise AssertionError(f"Unexpected GitHub call: {method} {endpoint}")

    def comments(self) -> list[tuple[str, str]]:
        return [
            (method, body)
            for method, endpoint, body in self.writes
            if endpoint != LEDGER_ENDPOINT
        ]


class FakeModel:
    """Answers chat completion requests; a None reply is a network failure."""

    def __init__(self, github: FakeGitHub, reply: str | None) -> None:
        self.github = github
        self.reply = reply
        self.ledger_sizes_at_call: list[int] = []

    def urlopen(self, request: Request, timeout: int) -> BytesIO:
        self.ledger_sizes_at_call.append(len(self.github.ledger.reservations))
        if self.reply is None:
            raise urllib.error.URLError("model unavailable")
        else:
            return BytesIO(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {"content": self.reply},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ).encode()
            )


def install_fakes(
    monkeypatch: pytest.MonkeyPatch, github: FakeGitHub, model: FakeModel
) -> None:
    monkeypatch.setattr(subprocess, "run", github.run)
    monkeypatch.setattr(
        "scripts.docs_review.transport.urllib.request.urlopen", model.urlopen
    )


def git(root: Path, *arguments: str) -> str:
    return REAL_SUBPROCESS_RUN(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def create_repository(root: Path, documentation_change: bool) -> tuple[str, str]:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name, value in (
        ("user.name", "Review Test"),
        ("user.email", "review@example.com"),
        ("core.hooksPath", "/dev/null"),
        ("commit.gpgsign", "false"),
    ):
        git(root, "config", name, value)
    (root / "README.md").write_text("Base\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "Base")
    base = git(root, "rev-parse", "HEAD").strip()
    if documentation_change:
        (root / "docs" / "api_reference").mkdir(parents=True)
        (root / "docs" / "api_reference" / "demo.md").write_text(
            "# API\nThe response is WAV.\n", encoding="utf-8"
        )
    else:
        pass
    (root / "demo.py").write_text('"""Return raw PCM."""\n', encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "Head")
    head = git(root, "rev-parse", "HEAD").strip()
    git(root, "remote", "add", "origin", str(root))
    git(root, "update-ref", "refs/pull/1/head", head)
    return base, head


def start_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_id: str) -> Path:
    runner_directory = tmp_path / f"runner-{run_id}"
    runner_directory.mkdir()
    monkeypatch.setenv("GITHUB_RUN_ID", run_id)
    monkeypatch.setenv("RUNNER_TEMP", str(runner_directory))
    monkeypatch.setenv("GITHUB_OUTPUT", str(runner_directory / "outputs"))
    return runner_directory / "docs-review" / "artifact"


def configure_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline: bool,
    automatic: bool = False,
    documentation_change: bool = True,
) -> PullRequest:
    repository = tmp_path / "repository"
    base, head = create_repository(repository, documentation_change)
    variables = {
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_SERVER_URL": "https://github.com",
        "PULL_REQUEST_NUMBER": "1",
        "GITHUB_WORKSPACE": str(repository),
        "GITHUB_RUN_ATTEMPT": "2",
        "AUTOMATIC_REVIEW": str(automatic).lower(),
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
    return PullRequest(
        state="open",
        body="Expected answer",
        base={"sha": base, "ref": "main"},
        head={"sha": head},
    )


def documentation_finding(claim: str) -> Finding:
    return Finding(
        rule_id="API_CONTRACT",
        file_path="docs/api_reference/demo.md",
        line_start=2,
        line_end=2,
        quote="The response is WAV.",
        claim=claim,
        evidence_file_path="demo.py",
        evidence_line_start=1,
        evidence_line_end=1,
        evidence_quote="Return raw PCM.",
        impact="Clients decode the wrong format.",
        suggested_fix="Document PCM.",
        change_relation="introduced",
        certainty="supported",
    )


def test_offline_report_contains_results_and_excludes_full_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = configure_run(tmp_path, monkeypatch, offline=True)
    artifact = start_run(monkeypatch, tmp_path, "123")
    updated = initial.model_copy(update={"head": Revision(sha="f" * 40)})
    github = FakeGitHub([initial, updated])
    model = FakeModel(github, "[]")
    install_fakes(monkeypatch, github, model)
    github_pilot.run_github_review()
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
    outputs = (artifact.parents[1] / "outputs").read_text(encoding="utf-8")
    assert "stale=true" in outputs
    assert model.ledger_sizes_at_call == []
    assert github.writes == []


def test_reservations_precede_the_model_call_and_bound_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pull_request = configure_run(tmp_path, monkeypatch, offline=False)
    github = FakeGitHub([pull_request])
    model = FakeModel(github, None)
    install_fakes(monkeypatch, github, model)

    artifact = start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    assert report["report"]["execution"] == "incomplete"
    assert model.ledger_sizes_at_call == [1]
    assert [entry.execution for entry in github.ledger.reservations] == ["incomplete"]

    start_run(monkeypatch, tmp_path, "124")
    with pytest.raises(PilotError, match="already reserved"):
        github_pilot.run_github_review()
    assert model.ledger_sizes_at_call == [1]
    assert len(github.ledger.reservations) == 1

    monkeypatch.setenv("FOLLOW_UP", "true")
    model.reply = "[]"
    artifact = start_run(monkeypatch, tmp_path, "125")
    github_pilot.run_github_review()
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    assert report["report"]["execution"] == "complete"
    assert [entry.execution for entry in github.ledger.reservations] == [
        "incomplete",
        "complete",
    ]

    start_run(monkeypatch, tmp_path, "126")
    with pytest.raises(PilotError, match="already reserved"):
        github_pilot.run_github_review()
    assert model.ledger_sizes_at_call == [1, 2]
    assert github.comments() == []


@pytest.mark.parametrize(
    ("limit", "expected_error"),
    [
        ("follow_up", "follow-up review must be explicitly authorized"),
        ("review_count", "review limit reached"),
        ("pull_request_budget", "Pull request budget exceeded"),
        ("daily_budget", "Daily budget exceeded"),
        ("call_budget", "exceeds the per-call budget"),
    ],
)
def test_review_limits_stop_before_the_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str, expected_error: str
) -> None:
    pull_request = configure_run(tmp_path, monkeypatch, offline=False)
    start_run(monkeypatch, tmp_path, "123")
    github = FakeGitHub([pull_request])
    model = FakeModel(github, "[]")
    install_fakes(monkeypatch, github, model)
    today = datetime.now(timezone.utc).date().isoformat()
    earlier_review = Reservation(
        pull_request_number=1,
        head_commit_sha="a" * 40,
        review_input_sha256="b" * 64,
        estimated_cost_cents=25,
        date_utc=today,
        run_id="1",
        execution="complete",
    )
    if limit == "follow_up":
        github.ledger.reservations = [earlier_review]
    elif limit == "review_count":
        monkeypatch.setenv("FOLLOW_UP", "true")
        github.ledger.reservations = [
            earlier_review,
            earlier_review.model_copy(
                update={"review_input_sha256": "c" * 64, "run_id": "2"}
            ),
        ]
    elif limit == "pull_request_budget":
        monkeypatch.setenv("FOLLOW_UP", "true")
        github.ledger.reservations = [
            earlier_review.model_copy(update={"estimated_cost_cents": 50})
        ]
    elif limit == "daily_budget":
        github.ledger.reservations = [
            earlier_review.model_copy(
                update={"pull_request_number": number, "run_id": str(number)}
            )
            for number in range(2, 10)
        ]
    else:
        monkeypatch.setenv(
            "DOCS_REVIEW_BUDGET",
            json.dumps(
                {
                    "max_cost_cents_per_call": 25,
                    "max_cost_cents_per_pull_request": 50,
                    "max_cost_cents_per_day": 200,
                    "call_cost_cents": 30,
                }
            ),
        )
    with pytest.raises(PilotError, match=expected_error):
        github_pilot.run_github_review()
    assert model.ledger_sizes_at_call == []
    assert github.writes == []


@pytest.mark.parametrize(
    ("conflicts", "expected_error"),
    [(2, None), (5, "HTTP 409")],
)
def test_ledger_conflicts_are_retried_a_bounded_number_of_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    conflicts: int,
    expected_error: str | None,
) -> None:
    pull_request = configure_run(tmp_path, monkeypatch, offline=False)
    start_run(monkeypatch, tmp_path, "123")
    github = FakeGitHub([pull_request])
    github.ledger_conflicts = conflicts
    model = FakeModel(github, "[]")
    install_fakes(monkeypatch, github, model)
    if expected_error is None:
        github_pilot.run_github_review()
        assert model.ledger_sizes_at_call == [1]
        assert [entry.execution for entry in github.ledger.reservations] == ["complete"]
    else:
        with pytest.raises(PilotError, match=expected_error):
            github_pilot.run_github_review()
        assert model.ledger_sizes_at_call == []
        assert github.ledger.reservations == []


def test_automatic_review_keeps_one_sticky_comment_across_pushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_push = configure_run(tmp_path, monkeypatch, offline=False, automatic=True)
    github = FakeGitHub([first_push])
    finding = documentation_finding(
        "@someone [see](https://example.invalid) documentation says WAV, "
        "but code returns PCM."
    )
    model = FakeModel(github, json.dumps([finding.model_dump()]))
    install_fakes(monkeypatch, github, model)

    start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    comments = github.comments()
    assert [method for method, body in comments] == ["POST"]
    posted = comments[0][1]
    assert github_comment.ADVISORY_REVIEW_MARKER in posted
    assert "@someone" not in posted
    assert "](https://example.invalid)" not in posted
    assert "Document PCM." in posted
    assert first_push.head.sha in posted
    assert f"/blob/{first_push.head.sha}/demo.py#L1" in posted

    repository = tmp_path / "repository"
    (repository / "docs" / "api_reference" / "demo.md").write_text(
        "# API\nThe response is WAV audio.\n", encoding="utf-8"
    )
    git(repository, "commit", "-qam", "Second push")
    second_head = git(repository, "rev-parse", "HEAD").strip()
    git(repository, "update-ref", "refs/pull/1/head", second_head)
    github.pull_requests = [
        first_push.model_copy(update={"head": Revision(sha=second_head)})
    ]
    start_run(monkeypatch, tmp_path, "124")
    github_pilot.run_github_review()
    comments = github.comments()
    assert [method for method, body in comments] == ["POST", "PATCH"]
    assert second_head in comments[1][1]
    assert len(model.ledger_sizes_at_call) == 2


def test_automatic_review_skips_without_a_model_call_when_nothing_to_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code_only = configure_run(
        tmp_path,
        monkeypatch,
        offline=False,
        automatic=True,
        documentation_change=False,
    )
    github = FakeGitHub([code_only])
    model = FakeModel(github, "[]")
    install_fakes(monkeypatch, github, model)
    start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    assert model.ledger_sizes_at_call == []
    assert github.writes == []


def test_automatic_review_skips_input_that_was_already_reviewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pull_request = configure_run(tmp_path, monkeypatch, offline=False, automatic=True)
    github = FakeGitHub([pull_request])
    model = FakeModel(github, "[]")
    install_fakes(monkeypatch, github, model)
    start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    writes_after_first_review = list(github.writes)

    start_run(monkeypatch, tmp_path, "124")
    github_pilot.run_github_review()
    assert model.ledger_sizes_at_call == [1]
    assert github.writes == writes_after_first_review


@pytest.mark.parametrize("change", ["head_pushed", "closed"])
def test_automatic_review_does_not_comment_when_the_pull_request_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    original = configure_run(tmp_path, monkeypatch, offline=False, automatic=True)
    if change == "head_pushed":
        changed = original.model_copy(update={"head": Revision(sha="f" * 40)})
    else:
        changed = original.model_copy(update={"state": "closed"})
    github = FakeGitHub([original, changed])
    finding = documentation_finding("documentation says WAV, but code returns PCM.")
    model = FakeModel(github, json.dumps([finding.model_dump()]))
    install_fakes(monkeypatch, github, model)
    artifact = start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    assert report["stale"] is True
    assert report["publishable"] is False
    assert github.comments() == []
    outputs = (artifact.parents[1] / "outputs").read_text(encoding="utf-8")
    assert "stale=true" in outputs


def test_unexpected_review_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    with pytest.raises(KeyError):
        github_pilot.main()
