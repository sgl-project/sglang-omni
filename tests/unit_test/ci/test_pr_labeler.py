import json
import shutil
import subprocess
from pathlib import Path
from typing import TypedDict

import pytest
import yaml

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="Node.js is required"
)

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[3]
    / ".github/workflows/pr-documentation-labeler.yml"
)
NODE_RUNNER = r"""
const {readFileSync} = require('node:fs');
const input = JSON.parse(readFileSync(0, 'utf8'));
let added = [];
let removed = [];
const pullRequest = input.pullRequest;
const context = {
  payload: {action: input.action ?? 'opened', pull_request: pullRequest},
  repo: {owner: 'example', repo: 'example'},
};
const pages = input.filePages ?? [input.files ?? []];
async function* fileIterator() {
  for (const data of pages) {
    yield {data};
  }
}
const contentRequests = [];
const github = {
  paginate: {iterator: () => fileIterator()},
  rest: {
    pulls: {get: async () => ({data: pullRequest})},
    repos: {getContent: async request => {
      contentRequests.push(request);
      if (input.codeowners === null) throw new Error('404 Not Found');
      return {data: {
        type: 'file', encoding: 'base64',
        content: input.encodedContent ?? Buffer.from(input.codeowners).toString('base64'),
      }};
    }},
    issues: {
      addLabels: async request => {added = request.labels},
      removeLabel: async request => {removed.push(request.name)},
    },
  },
};
const messages = [];
const warnings = [];
const core = {info: text => messages.push(text), warning: text => warnings.push(text)};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
(async () => {
  await new AsyncFunction('github', 'context', 'core', input.script)(
    github,
    context,
    core
  );
  process.stdout.write(JSON.stringify({
    contentRequests, messages, warnings,
    added: added.sort(),
    removed: removed.sort(),
  }));
})().catch(error => {console.error(error); process.exit(1)});
"""


def labeler_script() -> str:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    return workflow["jobs"]["label"]["steps"][0]["with"]["script"]


class LabelerResult(TypedDict):
    added: list[str]
    removed: list[str]
    messages: list[str]
    warnings: list[str]
    contentRequests: list[dict[str, str]]


def run_labeler(
    *,
    title: str,
    branch: str,
    files: list[dict[str, str]],
    existing_labels: list[str] | None = None,
    action: str = "opened",
    association: str | None = "MEMBER",
    login: str = "author",
    default_branch: str = "main",
    codeowners: str | None = "",
    encoded_content: str | None = None,
    draft: bool = False,
) -> LabelerResult:
    pull_request = {
        "number": 42,
        "author_association": association,
        "user": {"login": login},
        "base": {"repo": {"default_branch": default_branch}},
        "draft": draft,
        "title": title,
        "head": {"ref": branch},
        "labels": [{"name": name} for name in (existing_labels or [])],
    }
    payload = {
        "script": labeler_script(),
        "pullRequest": pull_request,
        "files": files,
        "action": action,
        "codeowners": codeowners,
        "encodedContent": encoded_content,
    }
    result = subprocess.run(
        ["node", "-e", NODE_RUNNER],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    ("title", "branch", "files", "existing_labels", "expected_labels"),
    [
        (
            "[Fix][Ming-ASR] Restore MoE expert probability normalization",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            ["bug"],
        ),
        (
            "Neutral title",
            "codex/fix/timeout",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "[Docs] Add a Voxt Quickstart",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "Neutral",
            "codex/docs/readme",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "docs-only",
            "neutral",
            [{"filename": "docs/foo.md", "status": "modified"}],
            None,
            ["documentation", "run-lint-only"],
        ),
        (
            "docs plus code",
            "neutral",
            [
                {"filename": "docs/foo.md", "status": "modified"},
                {"filename": "sglang_omni/x.py", "status": "modified"},
            ],
            None,
            [],
        ),
        (
            "Move a code example into the documentation",
            "neutral",
            [
                {
                    "filename": "docs/example.py",
                    "previous_filename": "examples/example.py",
                    "status": "renamed",
                }
            ],
            None,
            [],
        ),
        (
            "docs-only with a CI label",
            "neutral",
            [{"filename": "docs/foo.md", "status": "modified"}],
            ["run-unit-test-ci-only"],
            ["documentation"],
        ),
        (
            "[Feat] [NPU] MiniMax-Music3: NPU support",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            ["enhancement"],
        ),
        (
            "[Refactor] tighten imports",
            "codex/refactor/imports",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            ["refactor"],
        ),
        (
            "[Good first issue] improve messages",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            ["good first issue"],
        ),
        (
            "Good first issue: improve messages",
            "codex/good-first-issue/messages",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "[Help wanted] reproduce a failure",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            ["help wanted"],
        ),
        (
            "Help wanted: reproduce a failure",
            "codex/help-wanted/repro",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "New model directory",
            "neutral",
            [
                {
                    "filename": "sglang_omni/models/labeler_probe/__init__.py",
                    "status": "added",
                }
            ],
            None,
            ["model support"],
        ),
        (
            "Unrelated change",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "$(touch /tmp/labeler-injection)",
            "neutral",
            [{"filename": "sglang_omni/x.py", "status": "modified"}],
            None,
            [],
        ),
        (
            "Already labeled",
            "neutral",
            [{"filename": "docs/foo.md", "status": "modified"}],
            ["documentation"],
            ["run-lint-only"],
        ),
    ],
)
def test_suggested_labels(
    title: str,
    branch: str,
    files: list[dict[str, str]],
    existing_labels: list[str] | None,
    expected_labels: list[str],
) -> None:
    result = run_labeler(
        title=title,
        branch=branch,
        files=files,
        existing_labels=existing_labels,
    )
    assert result["added"] == sorted(expected_labels)
    assert result["removed"] == []


@pytest.mark.parametrize("association", ["MEMBER", "CONTRIBUTOR"])
def test_code_push_removes_stale_run_lint_only(association: str) -> None:
    result = run_labeler(
        association=association,
        title="docs plus code",
        branch="neutral",
        files=[
            {"filename": "docs/foo.md", "status": "modified"},
            {"filename": "sglang_omni/models/minicpm_o/model.py", "status": "modified"},
        ],
        existing_labels=["documentation", "run-lint-only"],
        action="synchronize",
    )
    assert result["added"] == []
    assert result["removed"] == ["run-lint-only"]


def test_synchronize_does_not_restore_removed_title_label() -> None:
    result = run_labeler(
        title="[Fix] Restore MoE expert probability normalization",
        branch="codex/fix/timeout",
        files=[{"filename": "sglang_omni/x.py", "status": "modified"}],
        action="synchronize",
    )
    assert result["added"] == []
    assert result["removed"] == []


def test_edited_title_still_adds_category_label() -> None:
    result = run_labeler(
        title="[Fix] Restore MoE expert probability normalization",
        branch="neutral",
        files=[{"filename": "sglang_omni/x.py", "status": "modified"}],
        action="edited",
    )
    assert result["added"] == ["bug"]
    assert result["removed"] == []


@pytest.mark.parametrize("association", ["OWNER", "MEMBER", "COLLABORATOR"])
def test_repository_role_skips_codeowners_lookup(association: str) -> None:
    result = run_labeler(
        title="[Fix] Typo",
        branch="feature",
        files=[{"filename": "README.md"}],
        association=association,
    )
    assert result["added"] == ["bug", "documentation", "run-lint-only"]
    assert result["contentRequests"] == []


@pytest.mark.parametrize(
    ("association", "default_branch"),
    [("FIRST_TIME_CONTRIBUTOR", "main"), ("CONTRIBUTOR", "main"), ("NONE", "stable")],
)
def test_individual_codeowner_on_default_branch(
    association: str, default_branch: str
) -> None:
    result = run_labeler(
        title="Typo",
        branch="feature",
        files=[{"filename": "README.md"}],
        association=association,
        login="frankleeeee",
        default_branch=default_branch,
        codeowners="docs/* @other @FrankLeeeee # reviewers",
    )
    assert result["added"] == ["documentation", "run-lint-only"]
    assert result["contentRequests"] == [
        {
            "owner": "example",
            "repo": "example",
            "path": ".github/CODEOWNERS",
            "ref": default_branch,
        }
    ]


@pytest.mark.parametrize(
    "association",
    [
        "FIRST_TIME_CONTRIBUTOR",
        "FIRST_TIMER",
        "CONTRIBUTOR",
        "NONE",
        "MANNEQUIN",
        "UNKNOWN",
        None,
    ],
)
def test_external_author_receives_triage_only(association: str | None) -> None:
    result = run_labeler(
        title="[Fix] Typo",
        branch="feature",
        files=[{"filename": "README.md"}],
        association=association,
        codeowners="# @author\n@author @other\n* @author/team author@example.com # @author",
    )
    assert result["added"] == ["bug", "documentation"]
    assert any("untrusted" in message for message in result["messages"])
    assert any(
        "Suppressed" in message and "run-lint-only" in message
        for message in result["messages"]
    )


@pytest.mark.parametrize(
    ("codeowners", "encoded_content"),
    [(None, None), ("", "KiBAYXV0aG9yCg==!"), ("", "KiBAYXV0aG9yCv8=")],
)
def test_unreadable_codeowners_preserves_triage(
    codeowners: str | None, encoded_content: str | None
) -> None:
    result = run_labeler(
        title="[Fix] Typo",
        branch="feature",
        files=[{"filename": "README.md"}],
        association="NONE",
        codeowners=codeowners,
        encoded_content=encoded_content,
    )
    assert result["added"] == ["bug", "documentation"]
    assert result["warnings"]


def test_trusted_draft_receives_triage_only() -> None:
    result = run_labeler(
        title="[Fix] Typo",
        branch="feature",
        files=[{"filename": "README.md"}],
        draft=True,
    )
    assert result["added"] == ["bug", "documentation"]


def test_rename_from_code_removes_stale_lint_label() -> None:
    result = run_labeler(
        title="Move example",
        branch="feature",
        files=[
            {
                "filename": "docs/example.py",
                "previous_filename": "examples/example.py",
                "status": "renamed",
            }
        ],
        existing_labels=["run-lint-only"],
        action="synchronize",
        association="NONE",
    )
    assert result["removed"] == ["run-lint-only"]
    assert result["added"] == []
