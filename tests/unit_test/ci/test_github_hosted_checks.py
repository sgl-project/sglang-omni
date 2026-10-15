# SPDX-License-Identifier: Apache-2.0
"""Contract for the checks that run on GitHub-hosted runners for every PR."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIRECTORY = REPO_ROOT / ".github/workflows"
ALWAYS_RUN_JOB_WORKFLOWS = (
    "lint.yaml",
    "test-layout.yaml",
    "docs-check.yaml",
    "rust-router.yml",
)
ALWAYS_RUN_WORKFLOWS = (
    *ALWAYS_RUN_JOB_WORKFLOWS,
    "release-docker-npu-nightly.yml",
)
SELF_HOSTED_GATE_JOBS = (
    ("omni-ci.yaml", "preflight"),
    ("omni-cpu-ci.yaml", "preflight"),
    ("omni-xpu-ci.yaml", "preflight"),
    ("voxt-mac-ci.yaml", "preflight"),
)
PULL_REQUEST_FILTER_KEYS = ("paths", "paths-ignore", "branches", "branches-ignore")

WorkflowNode: TypeAlias = str | list["WorkflowNode"] | dict[str, "WorkflowNode"]


def load_workflow(file_name: str) -> dict[str, WorkflowNode]:
    return yaml.load(
        (WORKFLOW_DIRECTORY / file_name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )


@pytest.mark.parametrize("workflow_name", ALWAYS_RUN_WORKFLOWS)
def test_github_hosted_workflow_starts_on_every_pull_request(
    workflow_name: str,
) -> None:
    pull_request_trigger = load_workflow(workflow_name)["on"]["pull_request"]
    if isinstance(pull_request_trigger, dict):
        assert not set(PULL_REQUEST_FILTER_KEYS) & pull_request_trigger.keys()
    else:
        assert pull_request_trigger == ""


@pytest.mark.parametrize("workflow_name", ALWAYS_RUN_JOB_WORKFLOWS)
def test_github_hosted_jobs_have_no_label_or_path_condition(
    workflow_name: str,
) -> None:
    for job_id, job in load_workflow(workflow_name)["jobs"].items():
        assert "self-hosted" not in job["runs-on"], job_id
        assert "if" not in job, job_id


@pytest.mark.parametrize(("workflow_name", "gate_job_id"), SELF_HOSTED_GATE_JOBS)
def test_self_hosted_gate_waits_for_lint(workflow_name: str, gate_job_id: str) -> None:
    gate_steps = load_workflow(workflow_name)["jobs"][gate_job_id]["steps"]
    assert any(
        "pr_ci_gate.py" in step.get("run", "") for step in gate_steps
    ), workflow_name


def test_gate_check_name_matches_the_lint_job() -> None:
    lint_job_ids = load_workflow("lint.yaml")["jobs"].keys()
    gate_source = (REPO_ROOT / ".github/scripts/pr_ci_gate.py").read_text(
        encoding="utf-8"
    )
    assert '"--check-name", default="lint"' in gate_source
    assert "lint" in lint_job_ids
