import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

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
const github = {
  paginate: {iterator: () => fileIterator()},
  rest: {
    pulls: {get: async () => ({data: pullRequest})},
    issues: {
      addLabels: async request => {added = request.labels},
      removeLabel: async request => {removed.push(request.name)},
    },
  },
};
const core = {info: () => {}};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
(async () => {
  await new AsyncFunction('github', 'context', 'core', input.script)(
    github,
    context,
    core
  );
  process.stdout.write(JSON.stringify({
    added: added.sort(),
    removed: removed.sort(),
  }));
})().catch(error => {console.error(error); process.exit(1)});
"""


def labeler_script() -> str:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    return workflow["jobs"]["label"]["steps"][0]["with"]["script"]


def run_labeler(
    *,
    title: str,
    branch: str,
    files: list[dict[str, str]],
    existing_labels: list[str] | None = None,
    action: str = "opened",
) -> tuple[list[str], list[str]]:
    pull_request = {
        "number": 42,
        "title": title,
        "head": {"ref": branch},
        "labels": [{"name": name} for name in (existing_labels or [])],
    }
    payload = {
        "script": labeler_script(),
        "pullRequest": pull_request,
        "files": files,
        "action": action,
    }
    result = subprocess.run(
        ["node", "-e", NODE_RUNNER],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    changes = json.loads(result.stdout)
    return changes["added"], changes["removed"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
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
            ["documentation"],
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
            ["documentation"],
        ),
        (
            "docs plus code with run label",
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
    added, removed = run_labeler(
        title=title,
        branch=branch,
        files=files,
        existing_labels=existing_labels,
    )
    assert added == sorted(expected_labels)
    assert removed == []


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_code_push_removes_stale_run_lint_only() -> None:
    added, removed = run_labeler(
        title="docs plus code",
        branch="neutral",
        files=[
            {"filename": "docs/foo.md", "status": "modified"},
            {"filename": "sglang_omni/models/minicpm_o/model.py", "status": "modified"},
        ],
        existing_labels=["documentation", "run-lint-only"],
        action="synchronize",
    )
    assert added == []
    assert removed == ["run-lint-only"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_synchronize_does_not_restore_removed_title_label() -> None:
    added, removed = run_labeler(
        title="[Fix] Restore MoE expert probability normalization",
        branch="codex/fix/timeout",
        files=[{"filename": "sglang_omni/x.py", "status": "modified"}],
        action="synchronize",
    )
    assert added == []
    assert removed == []


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_edited_title_still_adds_category_label() -> None:
    added, removed = run_labeler(
        title="[Fix] Restore MoE expert probability normalization",
        branch="neutral",
        files=[{"filename": "sglang_omni/x.py", "status": "modified"}],
        action="edited",
    )
    assert added == ["bug"]
    assert removed == []
