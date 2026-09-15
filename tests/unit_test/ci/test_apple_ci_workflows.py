# SPDX-License-Identifier: Apache-2.0
"""Exercise Apple CI gates without GitHub credentials or accelerator hardware."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location(
        "pr_ci_gate", ROOT / ".github/scripts/pr_ci_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("draft", "labels", "error"),
    [
        (True, [{"name": "run-ci"}], "draft"),
        (False, [], "run-ci"),
        (False, [{"name": "run-ci"}], None),
    ],
)
def test_pr_gate_checks_live_opt_in(gate, monkeypatch, draft, labels, error):
    calls = []

    def fetch(token, repo, path):
        calls.append(path)
        return {"draft": draft, "labels": labels}

    monkeypatch.setattr(gate, "_github_json", fetch)
    if error:
        with pytest.raises(RuntimeError, match=error):
            gate._require_pr_opt_in("token", "owner/repo", 7)
    else:
        gate._require_pr_opt_in("token", "owner/repo", 7)
    assert calls == ["/pulls/7"]


def test_manual_dispatch_bypasses_pr_gate(gate, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pr_ci_gate.py", "--require-run-ci"])
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert gate.main() == 0


@pytest.mark.parametrize("backend", ["mlx", "mps"])
@pytest.mark.parametrize(
    ("changes", "change_result", "gate_result", "stage_result", "success"),
    [
        (True, "success", "success", "success", True),
        (False, "success", "skipped", "skipped", True),
        (True, "success", "failure", "skipped", False),
        (True, "success", "success", "failure", False),
        (True, "success", "success", "cancelled", False),
        (True, "success", "success", "skipped", False),
        (False, "failure", "skipped", "skipped", False),
    ],
)
def test_finish_requires_selected_stages(
    backend, changes, change_result, gate_result, stage_result, success
):
    workflow = yaml.safe_load(
        (ROOT / f".github/workflows/omni-{backend}-ci.yaml").read_text()
    )
    finish = workflow["jobs"][f"omni-{backend}-ci-finish"]
    # Execute the workflow's actual Python summary against simulated job results.
    script = "\n".join(finish["steps"][0]["run"].splitlines()[1:-1])
    jobs = {
        "check-changes": {
            "result": change_result,
            "outputs": {"changes_exist": str(changes).lower()},
        },
        "pr-gate": {"result": gate_result},
        f"stage-a-unit-test-{backend}": {"result": stage_result},
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "JOB_RESULTS": json.dumps(jobs)},
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == success, result.stdout + result.stderr
