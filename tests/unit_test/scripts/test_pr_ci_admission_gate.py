# SPDX-License-Identifier: Apache-2.0
"""Tests for the pull_request_target CI admission gate workflow and script."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Callable

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / ".github/scripts/pr_ci_admission_gate.py"
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/pr-ci-admission-gate.yaml"

SPEC = importlib.util.spec_from_file_location("pr_ci_admission_gate", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
# note (0xtoward): Register module before exec_module so annotations and future imports resolve cleanly.
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)

BASE_SHA = "b" * 40
NEW_MAIN_SHA = "e" * 40
HEAD_SHA = "a" * 40
BASE_TREE_SHA = "c" * 40
HEAD_TREE_SHA = "d" * 40
SHA_REF_PATTERN = re.compile(r"^[^@]+@[0-9a-f]{40}$")
EXPECTED_ACTIONS = {
    "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd",
    "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
    "actions/github-script@ed597411d8f924073f98dfc5c65a23a2325f34cd",
}


def make_snapshot_fetcher(
    pull_data: dict[str, object] | list[dict[str, object]],
    compare_data: dict[str, object],
    trees: dict[str, dict[str, object]] | None = None,
) -> tuple[Callable[[str], object], list[str]]:
    requested_paths: list[str] = []
    tree_map = trees if trees is not None else {}
    pull_state_index = 0

    def fetch_json(api_path: str) -> object:
        nonlocal pull_state_index
        requested_paths.append(api_path)
        if api_path.startswith("/pulls/"):
            if isinstance(pull_data, list):
                current = pull_data[min(pull_state_index, len(pull_data) - 1)]
                pull_state_index += 1
                return current
            else:
                return pull_data
        elif api_path.startswith("/compare/"):
            return compare_data
        elif api_path == f"/git/commits/{HEAD_SHA}":
            return {"sha": HEAD_SHA, "tree": {"sha": HEAD_TREE_SHA}}
        elif api_path in (
            f"/git/commits/{BASE_SHA}",
            f"/git/commits/{NEW_MAIN_SHA}",
        ):
            return {"sha": BASE_SHA, "tree": {"sha": BASE_TREE_SHA}}
        elif api_path.startswith("/git/trees/"):
            tree_key = api_path.removeprefix("/git/trees/")
            if tree_key in tree_map:
                return tree_map[tree_key]
            else:
                tree_id = tree_key.split("?", 1)[0]
                return tree_map[tree_id]
        else:
            raise AssertionError(f"Unexpected JSON API request: {api_path}")

    return fetch_json, requested_paths


def test_workflow_enforces_minimal_permissions_idempotent_notify_and_pinned_actions() -> (
    None
):
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    on_block = workflow.get("on", workflow.get(True))

    assert (
        "pull_request_target" in on_block
    ), "workflow must trigger on pull_request_target"
    assert workflow.get("permissions") == {}, "top-level permissions must be empty"
    assert (
        workflow["concurrency"]["cancel-in-progress"] is True
    ), "supersede stale runs on new pushes"

    evaluate_job = workflow["jobs"]["evaluate"]
    notify_job = workflow["jobs"]["notify-maintainers"]
    verdict_job = workflow["jobs"]["admission-gate"]
    assert "head.repo.full_name != github.repository" in evaluate_job["if"]
    assert evaluate_job["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
    }, "evaluate job must stay read-only"
    assert (
        evaluate_job["outputs"]["outcome"] == "${{ steps.gate.outputs.outcome }}"
    ), "gate outcome must flow to notify-maintainers and admission-gate"
    assert notify_job["needs"] == ["evaluate"]
    assert notify_job["permissions"] == {
        "pull-requests": "write",
    }, "notify-maintainers must only write PR comments"
    assert (
        notify_job["if"] == "needs.evaluate.outputs.outcome == 'blocked'"
    ), "notify-maintainers must run only on genuine policy blocks"
    assert verdict_job["name"] == "CI Admission Gate"
    assert verdict_job["needs"] == ["evaluate"]
    assert (
        verdict_job["if"]
        == "(success() || failure()) && needs.evaluate.result != 'skipped'"
    )
    assert verdict_job["permissions"] == {}
    assert len(verdict_job["steps"]) == 1
    assert verdict_job["steps"][0]["env"] == {
        "EVALUATE_RESULT": "${{ needs.evaluate.result }}",
        "GATE_OUTCOME": "${{ needs.evaluate.outputs.outcome }}",
    }

    for step in notify_job["steps"]:
        uses = step.get("uses", "")
        assert not uses.startswith(
            "actions/checkout"
        ), "notify job must never check out code"
        if uses.startswith("actions/github-script@"):
            assert (
                f"{uses} # v8.0.0" in workflow_text
            ), "github-script version comment must match SHA"
            script = step["with"]["script"]
            assert (
                "${{" not in script
            ), "notification JS must never interpolate context expressions"
            assert "@fengxiaohu @0xtoward" in script
            assert "startsWith(marker)" in script
            assert "existing.body === body" in script
        else:
            pass

    for step in evaluate_job["steps"]:
        uses = step.get("uses")
        if uses is not None:
            assert SHA_REF_PATTERN.match(uses), f"unpinned action reference: {uses}"
            assert uses in EXPECTED_ACTIONS, f"unexpected action dependency: {uses}"
            if uses.startswith("actions/checkout@"):
                checkout_with = step.get("with", {})
                assert checkout_with.get("repository") == "${{ github.repository }}"
                assert checkout_with.get("persist-credentials") is False
                assert checkout_with.get("ref") is None
            else:
                pass
        else:
            assert step.get("id") == "gate"
            assert step.get("continue-on-error") is True
            assert step.get("run") == "python3 .github/scripts/pr_ci_admission_gate.py"


@pytest.mark.parametrize(
    ("raw_path", "expected_reason"),
    [
        (".github/workflows/hack.yaml", "outside_allowlist"),
        (".github/scripts/resolve_ci_labels.py", "outside_allowlist"),
        (".github/actions/omni-setup/action.yml", "outside_allowlist"),
        ("scripts/ci/foo.py", "outside_allowlist"),
        ("docker/Dockerfile", "outside_allowlist"),
        ("sglang_omni_router/src/main.rs", "outside_allowlist"),
        ("pyproject.toml", "outside_allowlist"),
        ("pyproject_xpu.toml", "outside_allowlist"),
        ("install.sh", "outside_allowlist"),
        (".pre-commit-config.yaml", "outside_allowlist"),
        ("../escape.py", "invalid_path"),
        ("/etc/passwd", "invalid_path"),
        ("sglang_omni\\evil.py", "invalid_path"),
        ("sglang_omni/../.github/x.yaml", "invalid_path"),
        ("tests/conftest.py", "protected_path"),
        ("tests/unit_test/Conftest.py", "protected_path"),
        ("sglang_omni_mlx/native/CMakeLists.txt", "protected_path"),
        ("sglang_omni_mlx/native/scripts/build_runtime.sh", "protected_path"),
        ("Voxt/scripts/test-shard.sh", "protected_path"),
        ("Voxt/go.mod", "protected_path"),
        ("Voxt/go.sum", "protected_path"),
        ("Voxt/helper/Package.swift", "protected_path"),
        ("Voxt/helper/Package.resolved", "protected_path"),
        ("playground/package.json", "protected_path"),
        ("tests/requirements-dev.txt", "protected_path"),
        ("sglang_omni/.github/helper.py", "protected_path"),
        ("docs/.gitattributes", "protected_path"),
        ("docs/serve.sh", "disallowed_extension"),
        ("sglang_omni/foo.pth", "disallowed_extension"),
        ("tests/unit_test/helper.sh", "disallowed_extension"),
        ("sglang_omni/libbad.so", "disallowed_extension"),
        ("examples/demo.ipynb", "disallowed_extension"),
        ("benchmarks/run_tool", "disallowed_extension"),
        ("sglang_omni/README_TOOL", "disallowed_extension"),
    ],
)
def test_classify_touched_path_blocks_sensitive_and_untrusted_paths(
    raw_path: str,
    expected_reason: str,
) -> None:
    reason = GATE.classify_touched_path(raw_path)

    assert (
        reason == expected_reason
    ), f"expected {expected_reason} for {raw_path}, got {reason}"


@pytest.mark.parametrize(
    "raw_path",
    [
        "README.md",
        "CONTRIBUTING.md",
        "sglang_omni/models/qwen.py",
        "sglang_omni/models/qwen.pyi",
        "tests/unit_test/test_foo.py",
        "docs/guide.md",
        "Voxt/main.go",
        "sglang_omni_mlx/native/runtime.cc",
        "playground/cascade/frontend/orb/app.test.mjs",
        "Voxt/third_party/mygo-patches/0023-updater-status.patch",
        "Voxt/third_party/mygo/VOXTNEW-VENDORED",
        "Voxt/NOTICE",
        "benchmarks/c3/LICENSE",
        "benchmarks/c3/COPYING",
        "sglang_omni/models/audio8_tts/codec/NOTICE",
        "tests/data/sample.golden",
        "docs/page.tmpl",
        "Voxt/shader.hlsl",
    ],
)
def test_classify_touched_path_allows_standard_implementation_paths(
    raw_path: str,
) -> None:
    reason = GATE.classify_touched_path(raw_path)

    assert reason is None, f"expected None for {raw_path}, got {reason}"


@pytest.mark.parametrize(
    ("pull_request", "expected_same_repo"),
    [
        (
            {
                "author_association": "COLLABORATOR",
                "user": {"login": "maintainer"},
                "head": {
                    "repo": {"fork": False, "full_name": "sgl-project/sglang-omni"}
                },
            },
            True,
        ),
        (
            {
                "author_association": "OWNER",
                "user": {"login": "owner-dev"},
                "head": {"repo": {"fork": True, "full_name": "owner-dev/sglang-omni"}},
            },
            False,
        ),
        (
            {
                "author_association": "MEMBER",
                "user": {"login": "read-triage-newcomer"},
                "head": {
                    "repo": {
                        "fork": True,
                        "full_name": "read-triage-newcomer/sglang-omni",
                    }
                },
            },
            False,
        ),
        (
            {
                "author_association": "COLLABORATOR",
                "user": {"login": "charliechenye"},
                "head": {
                    "repo": {"fork": True, "full_name": "charliechenye/sglang-omni"}
                },
            },
            False,
        ),
        (
            {
                "author_association": "CONTRIBUTOR",
                "user": {"login": "ext-dev"},
                "head": {"repo": {"fork": True, "full_name": "ext-dev/sglang-omni"}},
            },
            False,
        ),
        (
            {
                "author_association": "FIRST_TIME_CONTRIBUTOR",
                "user": {"login": "new-user"},
                "head": {"repo": {"fork": True, "full_name": "new-user/sglang-omni"}},
            },
            False,
        ),
        (
            {
                "author_association": "NONE",
                "user": {"login": "anon"},
                "head": {"repo": {"fork": True, "full_name": "anon/sglang-omni"}},
            },
            False,
        ),
        (
            {
                "author_association": "NONE",
                "user": {"login": "spoof"},
                "head": {
                    "repo": {"fork": True, "full_name": "sgl-project/sglang-omni"}
                },
            },
            False,
        ),
        (
            {
                "author_association": "OWNER",
                "head": "not-a-dict",
            },
            False,
        ),
        (
            {
                "author_association": "OWNER",
                "head": {"repo": None},
            },
            False,
        ),
    ],
)
def test_is_same_repository_pr_only_exempts_non_fork_base_repo_branches(
    pull_request: dict[str, object],
    expected_same_repo: bool,
) -> None:
    same_repo = GATE.is_same_repository_pr(
        pull_request=pull_request,
        repo="sgl-project/sglang-omni",
    )

    assert same_repo is expected_same_repo


def test_evaluate_pull_request_exempts_same_repository_without_snapshot_calls() -> None:
    fetch_json, requested = make_snapshot_fetcher({}, {})

    outcome, categories = GATE.evaluate_pull_request(
        pull_number=1,
        base_ref="main",
        head_sha=HEAD_SHA,
        same_repository=True,
        fetch_json=fetch_json,
    )

    assert outcome == "passed"
    assert categories == []
    assert requested == [], "same-repository PR should skip snapshot API calls"


def test_evaluate_pull_request_audits_pinned_fork_snapshot_and_passes_clean_pr() -> (
    None
):
    blob_sha = "1" * 40
    fetch_json, requested = make_snapshot_fetcher(
        pull_data={
            "base": {"ref": "main", "sha": BASE_SHA},
            "head": {"sha": HEAD_SHA},
            "changed_files": 1,
        },
        compare_data={
            "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
            "files": [
                {
                    "filename": "sglang_omni/models/clean.py",
                    "status": "added",
                    "sha": blob_sha,
                }
            ],
        },
        trees={
            f"{HEAD_TREE_SHA}?recursive=1": {
                "truncated": False,
                "tree": [
                    {
                        "path": "sglang_omni/models/clean.py",
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob_sha,
                    }
                ],
            },
        },
    )

    outcome, categories = GATE.evaluate_pull_request(
        pull_number=10,
        base_ref="main",
        head_sha=HEAD_SHA,
        same_repository=False,
        fetch_json=fetch_json,
    )

    assert outcome == "passed"
    assert categories == []
    assert f"/compare/{BASE_SHA}...{HEAD_SHA}?per_page=1" in requested


def test_evaluate_pull_request_passes_when_main_advances_mid_run() -> None:
    blob_sha = "1" * 40
    fetch_json, requested = make_snapshot_fetcher(
        pull_data=[
            {
                "base": {"ref": "main", "sha": NEW_MAIN_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
            {
                "base": {"ref": "main", "sha": "9" * 40},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
        ],
        compare_data={
            "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
            "files": [
                {
                    "filename": "sglang_omni/models/clean.py",
                    "status": "added",
                    "sha": blob_sha,
                }
            ],
        },
        trees={
            f"{HEAD_TREE_SHA}?recursive=1": {
                "truncated": False,
                "tree": [
                    {
                        "path": "sglang_omni/models/clean.py",
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob_sha,
                    }
                ],
            },
        },
    )

    outcome, categories = GATE.evaluate_pull_request(
        pull_number=15,
        base_ref="main",
        head_sha=HEAD_SHA,
        same_repository=False,
        fetch_json=fetch_json,
    )

    assert outcome == "passed"
    assert categories == []
    assert f"/compare/{NEW_MAIN_SHA}...{HEAD_SHA}?per_page=1" in requested


@pytest.mark.parametrize(
    ("filename", "expected_outcome", "expected_categories"),
    [
        ("docs/guide.md", "passed", []),
        (".github/workflows/omni-ci.yaml", "blocked", ["outside_allowlist"]),
    ],
)
def test_evaluate_pull_request_audits_non_default_base_branch_against_live_base(
    filename: str,
    expected_outcome: str,
    expected_categories: list[str],
) -> None:
    blob_sha = "1" * 40
    fetch_json, requested = make_snapshot_fetcher(
        pull_data={
            "base": {"ref": "release/v1", "sha": BASE_SHA},
            "head": {"sha": HEAD_SHA},
            "changed_files": 1,
        },
        compare_data={
            "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
            "files": [
                {
                    "filename": filename,
                    "status": "added",
                    "sha": blob_sha,
                }
            ],
        },
        trees={
            f"{HEAD_TREE_SHA}?recursive=1": {
                "truncated": False,
                "tree": [
                    {
                        "path": filename,
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob_sha,
                    }
                ],
            },
        },
    )

    outcome, categories = GATE.evaluate_pull_request(
        pull_number=16,
        base_ref="release/v1",
        head_sha=HEAD_SHA,
        same_repository=False,
        fetch_json=fetch_json,
    )

    assert outcome == expected_outcome
    assert categories == expected_categories
    assert f"/compare/{BASE_SHA}...{HEAD_SHA}?per_page=1" in requested


def test_evaluate_pull_request_marks_superseded_when_non_default_base_retargeted_to_main() -> (
    None
):
    fetch_json, requested = make_snapshot_fetcher(
        pull_data={
            "base": {"ref": "main", "sha": BASE_SHA},
            "head": {"sha": HEAD_SHA},
            "changed_files": 1,
        },
        compare_data={},
    )

    outcome, categories = GATE.evaluate_pull_request(
        pull_number=17,
        base_ref="ASR_eval",
        head_sha=HEAD_SHA,
        same_repository=False,
        fetch_json=fetch_json,
    )

    assert outcome == "superseded"
    assert categories == ["base_ref_changed"]
    assert requested == ["/pulls/17"]


def test_evaluate_pull_request_allows_large_clean_pr_above_fifty_files() -> None:
    file_count = 60
    compare_files = [
        {"filename": f"docs/file_{index}.md", "status": "added", "sha": "4" * 40}
        for index in range(file_count)
    ]
    tree_entries = [
        {
            "path": f"docs/file_{index}.md",
            "mode": "100644",
            "type": "blob",
            "sha": "4" * 40,
        }
        for index in range(file_count)
    ]
    fetch_json, _ = make_snapshot_fetcher(
        pull_data={
            "base": {"ref": "main", "sha": BASE_SHA},
            "head": {"sha": HEAD_SHA},
            "changed_files": file_count,
        },
        compare_data={
            "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
            "files": compare_files,
        },
        trees={
            f"{HEAD_TREE_SHA}?recursive=1": {
                "truncated": False,
                "tree": tree_entries,
            },
        },
    )

    outcome, categories = GATE.evaluate_pull_request(
        pull_number=14,
        base_ref="main",
        head_sha=HEAD_SHA,
        same_repository=False,
        fetch_json=fetch_json,
    )

    assert outcome == "passed"
    assert categories == []


def test_evaluate_pull_request_resolves_truncated_recursive_tree_via_subtrees() -> None:
    blob_sha = "3" * 40
    dir_tree_sha = "e" * 40
    fetch_json, _ = make_snapshot_fetcher(
        pull_data={
            "base": {"ref": "main", "sha": BASE_SHA},
            "head": {"sha": HEAD_SHA},
            "changed_files": 1,
        },
        compare_data={
            "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
            "files": [
                {
                    "filename": "docs/link.md",
                    "status": "added",
                    "sha": blob_sha,
                }
            ],
        },
        trees={
            f"{HEAD_TREE_SHA}?recursive=1": {"truncated": True, "tree": []},
            HEAD_TREE_SHA: {
                "truncated": False,
                "tree": [
                    {
                        "path": "docs",
                        "mode": "040000",
                        "type": "tree",
                        "sha": dir_tree_sha,
                    }
                ],
            },
            dir_tree_sha: {
                "truncated": False,
                "tree": [
                    {
                        "path": "link.md",
                        "mode": "120000",
                        "type": "blob",
                        "sha": blob_sha,
                    }
                ],
            },
        },
    )

    outcome, categories = GATE.evaluate_pull_request(
        pull_number=13,
        base_ref="main",
        head_sha=HEAD_SHA,
        same_repository=False,
        fetch_json=fetch_json,
    )

    assert outcome == "blocked"
    assert "symlink_change" in categories


@pytest.mark.parametrize(
    ("pull_data", "compare_data", "trees", "expected_outcome", "expected_category"),
    [
        (
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": "f" * 40},
                "changed_files": 1,
            },
            {},
            {},
            "superseded",
            "head_sha_mismatch",
        ),
        (
            [
                {
                    "base": {"ref": "main", "sha": BASE_SHA},
                    "head": {"sha": HEAD_SHA},
                    "changed_files": 1,
                },
                {
                    "base": {"ref": "main", "sha": BASE_SHA},
                    "head": {"sha": "f" * 40},
                    "changed_files": 1,
                },
            ],
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {"filename": "docs/ok.md", "status": "added", "sha": "4" * 40}
                ],
            },
            {
                f"{HEAD_TREE_SHA}?recursive=1": {
                    "truncated": False,
                    "tree": [
                        {
                            "path": "docs/ok.md",
                            "mode": "100644",
                            "type": "blob",
                            "sha": "4" * 40,
                        }
                    ],
                },
            },
            "superseded",
            "head_sha_mismatch",
        ),
        (
            [
                {
                    "base": {"ref": "main", "sha": BASE_SHA},
                    "head": {"sha": HEAD_SHA},
                    "changed_files": 1,
                },
                {
                    "base": {"ref": "feature-x", "sha": BASE_SHA},
                    "head": {"sha": HEAD_SHA},
                    "changed_files": 1,
                },
            ],
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {"filename": "docs/ok.md", "status": "added", "sha": "4" * 40}
                ],
            },
            {
                f"{HEAD_TREE_SHA}?recursive=1": {
                    "truncated": False,
                    "tree": [
                        {
                            "path": "docs/ok.md",
                            "mode": "100644",
                            "type": "blob",
                            "sha": "4" * 40,
                        }
                    ],
                },
            },
            "superseded",
            "base_ref_changed",
        ),
        (
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 350,
            },
            {
                "files": [
                    {
                        "filename": f"docs/f_{index}.md",
                        "status": "added",
                        "sha": "4" * 40,
                    }
                    for index in range(300)
                ]
            },
            {},
            "blocked",
            "snapshot_mismatch",
        ),
        (
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 2,
            },
            {
                "files": [
                    {"filename": "sglang_omni/a.py", "status": "added", "sha": "4" * 40}
                ]
            },
            {},
            "blocked",
            "snapshot_mismatch",
        ),
        (
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {"filename": "docs/secret.md", "status": "added", "sha": "4" * 40}
                ],
            },
            {
                f"{HEAD_TREE_SHA}?recursive=1": {
                    "truncated": False,
                    "tree": [
                        {
                            "path": "docs/secret.md",
                            "mode": "120000",
                            "type": "blob",
                            "sha": "4" * 40,
                        }
                    ],
                },
            },
            "blocked",
            "symlink_change",
        ),
        (
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {"filename": "examples/sub", "status": "added", "sha": "4" * 40}
                ],
            },
            {},
            "blocked",
            "disallowed_extension",
        ),
        (
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {"filename": "examples/sub.md", "status": "added", "sha": "4" * 40}
                ],
            },
            {
                f"{HEAD_TREE_SHA}?recursive=1": {
                    "truncated": False,
                    "tree": [
                        {
                            "path": "examples/sub.md",
                            "mode": "160000",
                            "type": "commit",
                            "sha": "4" * 40,
                        }
                    ],
                },
            },
            "blocked",
            "submodule_change",
        ),
        (
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {"filename": "examples/tool.py", "status": "added", "sha": "4" * 40}
                ],
            },
            {
                f"{HEAD_TREE_SHA}?recursive=1": {
                    "truncated": False,
                    "tree": [
                        {
                            "path": "examples/tool.py",
                            "mode": "100755",
                            "type": "blob",
                            "sha": "4" * 40,
                        }
                    ],
                },
            },
            "blocked",
            "executable_mode_change",
        ),
        (
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 2,
            },
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {"filename": "docs/Guide.md", "status": "added", "sha": "4" * 40},
                    {"filename": "docs/guide.md", "status": "added", "sha": "5" * 40},
                ],
            },
            {},
            "blocked",
            "case_collision",
        ),
        (
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {"filename": "docs/empty.md", "status": "renamed", "sha": "4" * 40}
                ],
            },
            {},
            "blocked",
            "snapshot_mismatch",
        ),
        (
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {
                        "filename": "docs/moved.md",
                        "previous_filename": ".github/workflows/omni-ci.yaml",
                        "status": "renamed",
                        "sha": "4" * 40,
                    }
                ],
            },
            {},
            "blocked",
            "outside_allowlist",
        ),
        (
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {"filename": "Voxt/NOTICE", "status": "added", "sha": "4" * 40}
                ],
            },
            {
                f"{HEAD_TREE_SHA}?recursive=1": {
                    "truncated": False,
                    "tree": [
                        {
                            "path": "Voxt/NOTICE",
                            "mode": "100755",
                            "type": "blob",
                            "sha": "4" * 40,
                        }
                    ],
                },
            },
            "blocked",
            "executable_mode_change",
        ),
        (
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {
                        "filename": "benchmarks/c3/LICENSE",
                        "status": "added",
                        "sha": "4" * 40,
                    }
                ],
            },
            {
                f"{HEAD_TREE_SHA}?recursive=1": {
                    "truncated": False,
                    "tree": [
                        {
                            "path": "benchmarks/c3/LICENSE",
                            "mode": "120000",
                            "type": "blob",
                            "sha": "4" * 40,
                        }
                    ],
                },
            },
            "blocked",
            "symlink_change",
        ),
    ],
)
def test_evaluate_pull_request_fails_closed_on_snapshot_and_tree_anomalies(
    pull_data: dict[str, object] | list[dict[str, object]],
    compare_data: dict[str, object],
    trees: dict[str, dict[str, object]],
    expected_outcome: str,
    expected_category: str,
) -> None:
    fetch_json, _ = make_snapshot_fetcher(pull_data, compare_data, trees=trees)

    outcome, categories = GATE.evaluate_pull_request(
        pull_number=11,
        base_ref="main",
        head_sha=HEAD_SHA,
        same_repository=False,
        fetch_json=fetch_json,
    )

    assert outcome == expected_outcome
    assert (
        expected_category in categories
    ), f"expected {expected_category}, got {categories}"


@pytest.mark.parametrize(
    (
        "author_association",
        "pull_data",
        "compare_data",
        "trees",
        "expected_exit",
        "expected_outcome",
    ),
    [
        (
            "NONE",
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {
                        "filename": ".github/workflows/omni-ci.yaml",
                        "status": "modified",
                        "sha": "4" * 40,
                    }
                ],
            },
            {},
            1,
            "blocked",
        ),
        (
            "MEMBER",
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {
                        "filename": ".github/workflows/omni-ci.yaml",
                        "status": "modified",
                        "sha": "4" * 40,
                    }
                ],
            },
            {},
            1,
            "blocked",
        ),
        (
            "NONE",
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": "f" * 40},
                "changed_files": 1,
            },
            {},
            {},
            1,
            "superseded",
        ),
        (
            "NONE",
            {
                "base": {"ref": "main", "sha": BASE_SHA},
                "head": {"sha": HEAD_SHA},
                "changed_files": 1,
            },
            {
                "merge_base_commit": {"commit": {"tree": {"sha": BASE_TREE_SHA}}},
                "files": [
                    {"filename": "docs/guide.md", "status": "added", "sha": "4" * 40}
                ],
            },
            {
                f"{HEAD_TREE_SHA}?recursive=1": {
                    "truncated": False,
                    "tree": [
                        {
                            "path": "docs/guide.md",
                            "mode": "100644",
                            "type": "blob",
                            "sha": "4" * 40,
                        }
                    ],
                },
            },
            0,
            "passed",
        ),
    ],
)
def test_main_writes_github_output_and_sanitized_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    author_association: str,
    pull_data: dict[str, object],
    compare_data: dict[str, object],
    trees: dict[str, dict[str, object]],
    expected_exit: int,
    expected_outcome: str,
) -> None:
    event_file = tmp_path / "event.json"
    output_file = tmp_path / "github_output"
    event_file.write_text(
        json.dumps(
            {
                "repository": {"default_branch": "main"},
                "pull_request": {
                    "number": 99,
                    "author_association": author_association,
                    "user": {"login": "ext"},
                    "base": {"ref": "main", "sha": BASE_SHA},
                    "head": {
                        "sha": HEAD_SHA,
                        "repo": {"fork": True, "full_name": "ext/sglang-omni"},
                    },
                },
                "sender": {"login": "ext"},
            }
        ),
        encoding="utf-8",
    )
    fetch_json, _ = make_snapshot_fetcher(pull_data, compare_data, trees=trees)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request_target")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "sgl-project/sglang-omni")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setattr(
        GATE, "github_get_json", lambda token, repo, api_path: fetch_json(api_path)
    )

    exit_code = GATE.main()
    captured = capsys.readouterr()

    assert exit_code == expected_exit
    assert f"outcome={expected_outcome}" in output_file.read_text(encoding="utf-8")
    assert "omni-ci.yaml" not in captured.err
    assert "outside_allowlist" not in captured.err
