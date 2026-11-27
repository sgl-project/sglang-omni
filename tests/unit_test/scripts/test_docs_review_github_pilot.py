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
from typing import Protocol
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


class ModelCallHook(Protocol):
    def __call__(self) -> None:
        """Run custom state transitions while the model request is in flight."""


CHECKLIST = """## Contributor checklist

- Purpose: Update API documentation.
- Reviewed the final diff: yes
- Checks: pytest passed
- Untested: none
- Performance: not applicable"""


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
            method, endpoint = arguments[3:5]
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
        elif method == "POST" and endpoint == COMMENTS_ENDPOINT:
            assert body is not None
            comment = json.loads(body)["body"]
            self.writes.append((method, endpoint, comment))
            self.issue_comments.append(
                {
                    "id": COMMENT_ID,
                    "user": {"login": github_comment.ADVISORY_BOT_LOGIN},
                    "body": comment,
                }
            )
            return json.dumps({"id": COMMENT_ID})
        elif method == "PATCH" and endpoint.startswith(
            "repos/owner/repo/issues/comments/"
        ):
            assert body is not None
            comment_id = int(endpoint.rsplit("/", maxsplit=1)[1])
            comment = json.loads(body)["body"]
            matching = [
                existing
                for existing in self.issue_comments
                if existing.get("id") == comment_id
            ]
            assert matching, f"Unknown comment id {comment_id}"
            owner = matching[0].get("user")
            assert (
                isinstance(owner, dict)
                and owner.get("login") == github_comment.ADVISORY_BOT_LOGIN
            ), "Bot attempted to PATCH a comment owned by another user"
            matching[0]["body"] = comment
            self.writes.append((method, endpoint, comment))
            return json.dumps({"id": comment_id})
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

    def __init__(
        self,
        github: FakeGitHub,
        reply: str | None,
        *,
        on_call: ModelCallHook | None = None,
    ) -> None:
        self.github = github
        self.reply = reply
        self.on_call = on_call
        self.ledger_sizes_at_call: list[int] = []
        self.request_bodies: list[str] = []

    def urlopen(self, request: Request, timeout: int) -> BytesIO:
        self.ledger_sizes_at_call.append(len(self.github.ledger.reservations))
        if isinstance(request.data, bytes):
            self.request_bodies.append(request.data.decode("utf-8"))
        if self.on_call is not None:
            self.on_call()
        if self.reply is None:
            raise urllib.error.URLError("model unavailable")
        else:
            choice = {"message": {"content": self.reply}, "finish_reason": "stop"}
            return BytesIO(json.dumps({"choices": [choice]}).encode())


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    github: FakeGitHub,
    reply: str | None = "[]",
    on_call: ModelCallHook | None = None,
) -> FakeModel:
    model = FakeModel(github, reply, on_call=on_call)
    monkeypatch.setattr(subprocess, "run", github.run)
    monkeypatch.setattr(
        "scripts.docs_review.transport.urllib.request.urlopen", model.urlopen
    )
    return model


def git(root: Path, *arguments: str) -> str:
    return REAL_SUBPROCESS_RUN(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def configure_git_identity(root: Path) -> None:
    for name, value in (
        ("user.name", "Review Test"),
        ("user.email", "review@example.com"),
        ("core.hooksPath", "/dev/null"),
        ("commit.gpgsign", "false"),
        ("uploadpack.allowAnySHA1InWant", "true"),
    ):
        git(root, "config", name, value)


def initialize_repository(root: Path) -> None:
    root.mkdir()
    REAL_SUBPROCESS_RUN(["git", "init", "-q", "-b", "main", str(root)], check=True)
    configure_git_identity(root)


def commit_changes(repository: Path, message: str) -> str:
    git(repository, "add", ".")
    git(repository, "commit", "-qm", message)
    return git(repository, "rev-parse", "HEAD").strip()


def read_report(artifact: Path) -> dict[str, object]:
    raw: object = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def create_repository(root: Path, documentation_change: bool) -> tuple[str, str]:
    initialize_repository(root)
    (root / "README.md").write_text("Base\n", encoding="utf-8")
    base = commit_changes(root, "Base")
    if documentation_change:
        (root / "docs" / "api_reference").mkdir(parents=True)
        (root / "docs" / "api_reference" / "demo.md").write_text(
            "# API\nThe response is WAV.\n", encoding="utf-8"
        )
    (root / "demo.py").write_text('"""Return raw PCM."""\n', encoding="utf-8")
    head = commit_changes(root, "Head")
    git(root, "remote", "add", "origin", str(root))
    git(root, "update-ref", "refs/pull/1/head", head)
    return base, head


def create_remote_and_workspace(tmp_path: Path) -> tuple[Path, Path, str, str]:
    origin, workspace = tmp_path / "origin", tmp_path / "workspace"
    base, head = create_repository(origin, documentation_change=True)
    git(origin, "branch", "pr-head", head)
    git(origin, "switch", "--detach", base)
    git(origin, "branch", "-f", "main", base)
    git(origin, "switch", "main")
    REAL_SUBPROCESS_RUN(
        [
            "git",
            "clone",
            "-q",
            "--no-local",
            "--single-branch",
            str(origin),
            str(workspace),
        ],
        check=True,
    )
    configure_git_identity(workspace)
    return origin, workspace, base, head


def start_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_id: str) -> Path:
    runner_directory = tmp_path / f"runner-{run_id}"
    runner_directory.mkdir()
    monkeypatch.setenv("GITHUB_RUN_ID", run_id)
    monkeypatch.setenv("RUNNER_TEMP", str(runner_directory))
    monkeypatch.setenv("GITHUB_OUTPUT", str(runner_directory / "outputs"))
    return runner_directory / "docs-review" / "artifact"


def set_review_environment(
    monkeypatch: pytest.MonkeyPatch,
    repository: Path,
    *,
    offline: bool,
    automatic: bool,
) -> None:
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


def open_pull_request(
    base: str, head: str, body: str | None = CHECKLIST
) -> PullRequest:
    return PullRequest(
        state="open",
        body=body,
        base=Revision(sha=base, ref="main"),
        head=Revision(sha=head),
    )


def configure_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline: bool = False,
    automatic: bool = False,
    documentation_change: bool = True,
    body: str | None = CHECKLIST,
) -> PullRequest:
    repository = tmp_path / "repository"
    base, head = create_repository(repository, documentation_change)
    set_review_environment(
        monkeypatch, repository, offline=offline, automatic=automatic
    )
    return open_pull_request(base, head, body)


def documentation_finding(
    claim: str,
    *,
    evidence_file_path: str = "demo.py",
    evidence_line_start: int = 1,
    evidence_line_end: int = 1,
    evidence_quote: str = "Return raw PCM.",
) -> Finding:
    return Finding(
        rule_id="API_CONTRACT",
        file_path="docs/api_reference/demo.md",
        line_start=2,
        line_end=2,
        quote="The response is WAV.",
        claim=claim,
        evidence_file_path=evidence_file_path,
        evidence_line_start=evidence_line_start,
        evidence_line_end=evidence_line_end,
        evidence_quote=evidence_quote,
        impact="Clients decode the wrong format.",
        suggested_fix="Document PCM.",
        change_relation="introduced",
        certainty="supported",
    )


def test_offline_report_contains_results_and_excludes_full_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = configure_run(tmp_path, monkeypatch, offline=True, body="Expected answer")
    artifact = start_run(monkeypatch, tmp_path, "123")
    updated = initial.model_copy(update={"head": Revision(sha="f" * 40)})
    github = FakeGitHub([initial, updated])
    model = install_fakes(monkeypatch, github)
    github_pilot.run_github_review()
    assert {file.name for file in artifact.iterdir()} == {"report.json", "report.md"}
    report = read_report(artifact)
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


@pytest.mark.parametrize("body", [None, ""])
def test_blank_github_pull_request_body_triggers_checklist_finding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str | None,
) -> None:
    """GitHub returns null for an empty PR description; checklist audit must still run."""
    pull_request = configure_run(tmp_path, monkeypatch, body=body)
    github = FakeGitHub([pull_request])
    install_fakes(monkeypatch, github)
    artifact = start_run(monkeypatch, tmp_path, "123")

    github_pilot.run_github_review()

    report = read_report(artifact)["report"]
    assert report["author_confirmation"] == "not provided"
    assert [finding["rule_id"] for finding in report["findings"]] == [
        "CONTRIBUTOR_CHECKLIST"
    ]


def test_github_artifact_uses_sanitized_model_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Published artifacts must not expose model endpoint credentials."""
    pull_request = configure_run(tmp_path, monkeypatch)
    monkeypatch.setenv(
        "DOCS_REVIEW_MODEL",
        json.dumps(
            {
                "model_name": "reviewer",
                "endpoint_url": "https://user:redaction-probe@example.invalid/chat?token=redaction-probe#redaction-probe",
                "max_output_tokens": 512,
                "api_key_environment_variable": "DOCS_REVIEW_API_KEY",
            }
        ),
    )
    github = FakeGitHub([pull_request])
    install_fakes(monkeypatch, github)
    artifact = start_run(monkeypatch, tmp_path, "123")

    github_pilot.run_github_review()

    report = read_report(artifact)["report"]
    assert report["model_configuration"] == {
        "model_name": "reviewer",
        "endpoint_url": "https://example.invalid/chat",
        "max_output_tokens": 512,
        "reasoning_effort": None,
        "temperature": 0,
    }
    for path in artifact.iterdir():
        content = path.read_text(encoding="utf-8")
        assert "redaction-probe" not in content
        assert "DOCS_REVIEW_API_KEY" not in content
        assert "test-only-key" not in content


def test_reservations_precede_the_model_call_and_bound_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pull_request = configure_run(tmp_path, monkeypatch)
    github = FakeGitHub([pull_request])
    model = install_fakes(monkeypatch, github, None)

    artifact = start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    report = read_report(artifact)
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
    report = read_report(artifact)
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
    pull_request = configure_run(tmp_path, monkeypatch)
    start_run(monkeypatch, tmp_path, "123")
    github = FakeGitHub([pull_request])
    model = install_fakes(monkeypatch, github)
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
    pull_request = configure_run(tmp_path, monkeypatch)
    start_run(monkeypatch, tmp_path, "123")
    github = FakeGitHub([pull_request])
    github.ledger_conflicts = conflicts
    model = install_fakes(monkeypatch, github)
    if expected_error is None:
        github_pilot.run_github_review()
        assert model.ledger_sizes_at_call == [1]
        assert [entry.execution for entry in github.ledger.reservations] == ["complete"]
    else:
        with pytest.raises(PilotError, match=expected_error):
            github_pilot.run_github_review()
        assert model.ledger_sizes_at_call == []
        assert github.ledger.reservations == []


def test_automatic_review_updates_and_clears_sticky_comment_when_findings_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_push = configure_run(tmp_path, monkeypatch, automatic=True)
    github = FakeGitHub([first_push])
    human_comment = {
        "id": 99,
        "user": {"login": "contributor"},
        "body": f"> {github_comment.ADVISORY_REVIEW_MARKER}",
    }
    github.issue_comments.append(human_comment.copy())
    github.issue_comments.append(
        {
            "id": 98,
            "user": {"login": github_comment.ADVISORY_BOT_LOGIN},
            "body": None,
        }
    )
    finding = documentation_finding(
        "@someone [see](https://example.invalid) documentation says WAV, "
        "but code returns PCM."
    )
    model = install_fakes(monkeypatch, github, json.dumps([finding.model_dump()]))

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
        "# API\nThe response is raw PCM.\n", encoding="utf-8"
    )
    git(repository, "commit", "-qam", "Fix documented response format")
    second_head = git(repository, "rev-parse", "HEAD").strip()
    git(repository, "update-ref", "refs/pull/1/head", second_head)
    github.pull_requests = [
        first_push.model_copy(update={"head": Revision(sha=second_head)})
    ]
    model.reply = "[]"
    start_run(monkeypatch, tmp_path, "124")
    github_pilot.run_github_review()
    comments = github.comments()
    assert [method for method, body in comments] == ["POST", "PATCH"]
    cleared = comments[1][1]
    assert github_comment.ADVISORY_REVIEW_MARKER in cleared
    assert "No advisory findings" in cleared
    assert second_head in cleared
    assert "Document PCM." not in cleared
    assert len(model.ledger_sizes_at_call) == 2
    assert github.issue_comments[0] == human_comment


def test_fetch_uses_pinned_head_sha_even_after_pull_ref_force_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin, workspace, base, head_a = create_remote_and_workspace(tmp_path)
    git(origin, "checkout", "-q", "pr-head")
    git(origin, "update-ref", "refs/keep/head_a", head_a)
    (origin / "docs" / "api_reference" / "demo.md").write_text(
        "# API\nForce-pushed replacement.\n", encoding="utf-8"
    )
    git(origin, "commit", "-qam", "Force-pushed head")
    head_b = git(origin, "rev-parse", "HEAD").strip()
    git(origin, "update-ref", "refs/pull/1/head", head_b)
    git(origin, "checkout", "-q", "main")

    set_review_environment(monkeypatch, workspace, offline=False, automatic=True)
    initial = open_pull_request(base, head_a)
    github = FakeGitHub([initial])
    finding = documentation_finding("documentation says WAV, but code returns PCM.")
    install_fakes(monkeypatch, github, json.dumps([finding.model_dump()]))

    artifact = start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    report = read_report(artifact)
    assert report["report"]["head_commit_sha"] == head_a
    assert report["report"]["execution"] == "complete"
    assert report["publishable"] is True
    assert len(github.comments()) == 1


def test_main_advancing_during_review_fetches_new_base_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin, workspace, base_before, head = create_remote_and_workspace(tmp_path)
    set_review_environment(monkeypatch, workspace, offline=False, automatic=True)
    initial = open_pull_request(base_before, head)
    github = FakeGitHub([initial])

    def advance_main_on_origin() -> None:
        (origin / "CHANGELOG.txt").write_text(
            "Unrelated main commit\n", encoding="utf-8"
        )
        git(origin, "add", "CHANGELOG.txt")
        git(origin, "commit", "-qm", "Advance main")
        base_after = git(origin, "rev-parse", "HEAD").strip()
        github.pull_requests = [
            initial.model_copy(update={"base": Revision(sha=base_after, ref="main")})
        ]

    finding = documentation_finding("documentation says WAV, but code returns PCM.")
    install_fakes(
        monkeypatch,
        github,
        json.dumps([finding.model_dump()]),
        on_call=advance_main_on_origin,
    )

    artifact = start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    report = read_report(artifact)
    assert report["stale"] is False
    assert report["publishable"] is True
    assert report["report"]["execution"] == "complete"
    assert [entry.execution for entry in github.ledger.reservations] == ["complete"]
    assert [method for method, body in github.comments()] == ["POST"]


def test_automatic_review_skips_when_base_advances_or_unrelated_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin, workspace, base_before, head_before = create_remote_and_workspace(tmp_path)
    set_review_environment(monkeypatch, workspace, offline=False, automatic=True)
    first_pr = open_pull_request(base_before, head_before)
    github = FakeGitHub([first_pr])
    model = install_fakes(monkeypatch, github)

    start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    writes_after_first = list(github.writes)
    assert model.ledger_sizes_at_call == [1]

    (origin / "main-note.txt").write_text("Main moved forward\n", encoding="utf-8")
    base_after = commit_changes(origin, "Advance main")

    git(origin, "checkout", "-q", "pr-head")
    (origin / "scratch.txt").write_text("Unrelated PR artifact\n", encoding="utf-8")
    head_after = commit_changes(origin, "Touch unrelated txt file")
    git(origin, "checkout", "-q", "main")

    github.pull_requests = [open_pull_request(base_after, head_after)]
    start_run(monkeypatch, tmp_path, "124")
    github_pilot.run_github_review()
    assert model.ledger_sizes_at_call == [1]
    assert github.writes == writes_after_first


def test_large_python_file_with_small_diff_crops_under_input_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    initialize_repository(repository)
    filler_lines = [
        f"CONSTANT_{index:04d} = 'padding value for large python module'\n"
        for index in range(1, 3201)
    ]
    (repository / "big_engine.py").write_text("".join(filler_lines), encoding="utf-8")
    base = commit_changes(repository, "Base large python file")

    target_line = 2500
    filler_lines[target_line - 1] = "AUDIO_FORMAT = 'raw_pcm'  # Return raw PCM.\n"
    (repository / "big_engine.py").write_text("".join(filler_lines), encoding="utf-8")
    (repository / "docs" / "api_reference").mkdir(parents=True)
    (repository / "docs" / "api_reference" / "demo.md").write_text(
        "# API\nThe response is WAV.\n", encoding="utf-8"
    )
    head = commit_changes(repository, "Small diff on large file plus doc")
    git(repository, "remote", "add", "origin", str(repository))

    set_review_environment(monkeypatch, repository, offline=False, automatic=True)
    pull_request = open_pull_request(base, head)
    github = FakeGitHub([pull_request])
    finding = documentation_finding(
        "documentation says WAV, but big_engine.py emits raw PCM.",
        evidence_file_path="big_engine.py",
        evidence_line_start=target_line,
        evidence_line_end=target_line,
        evidence_quote="AUDIO_FORMAT = 'raw_pcm'",
    )
    model = install_fakes(monkeypatch, github, json.dumps([finding.model_dump()]))

    artifact = start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    report = read_report(artifact)
    assert report["report"]["execution"] == "complete"
    assert report["publishable"] is True
    assert len(model.request_bodies) == 1
    messages = json.loads(model.request_bodies[0])["messages"]
    (user_message,) = [
        message["content"] for message in messages if message["role"] == "user"
    ]
    assert len(user_message.encode()) <= github_pilot.MAX_INPUT_BYTES
    posted = github.comments()[0][1]
    assert f"/blob/{head}/big_engine.py#L{target_line}" in posted


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
    model = install_fakes(monkeypatch, github)
    start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    assert model.ledger_sizes_at_call == []
    assert github.writes == []


def test_automatic_review_skips_input_that_was_already_reviewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pull_request = configure_run(tmp_path, monkeypatch, automatic=True)
    github = FakeGitHub([pull_request])
    model = install_fakes(monkeypatch, github)
    start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    assert github.comments() == []
    writes_after_first_review = list(github.writes)

    start_run(monkeypatch, tmp_path, "124")
    github_pilot.run_github_review()
    assert model.ledger_sizes_at_call == [1]
    assert github.writes == writes_after_first_review


@pytest.mark.parametrize("change", ["head_pushed", "closed"])
def test_automatic_review_does_not_comment_when_the_pull_request_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    original = configure_run(tmp_path, monkeypatch, automatic=True)
    if change == "head_pushed":
        changed = original.model_copy(update={"head": Revision(sha="f" * 40)})
    else:
        changed = original.model_copy(update={"state": "closed"})
    github = FakeGitHub([original, changed])
    finding = documentation_finding("documentation says WAV, but code returns PCM.")
    install_fakes(monkeypatch, github, json.dumps([finding.model_dump()]))
    artifact = start_run(monkeypatch, tmp_path, "123")
    github_pilot.run_github_review()
    report = read_report(artifact)
    assert report["stale"] is True
    assert report["publishable"] is False
    assert github.comments() == []
    outputs = (artifact.parents[1] / "outputs").read_text(encoding="utf-8")
    assert "stale=true" in outputs


def test_unexpected_review_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    with pytest.raises(KeyError):
        github_pilot.main()


@pytest.mark.parametrize(
    "name", ["PULL_REQUEST_NUMBER", "DOCS_REVIEW_MODEL", "DOCS_REVIEW_BUDGET"]
)
def test_invalid_inputs_fail_before_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    pull_request = configure_run(tmp_path, monkeypatch)
    github = FakeGitHub([pull_request])
    model = install_fakes(monkeypatch, github)
    start_run(monkeypatch, tmp_path, "123")
    monkeypatch.setenv(name, "invalid private input")
    with pytest.raises(PilotError) as error:
        github_pilot.run_github_review()
    assert "invalid private input" not in str(error.value)
    assert github.writes == []
    assert model.request_bodies == []
