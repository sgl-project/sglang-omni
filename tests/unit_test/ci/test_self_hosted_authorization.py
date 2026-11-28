# SPDX-License-Identifier: Apache-2.0
"""Every PR-accessible hardware job must depend on the protected authorization check."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github/workflows"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_current_sha_authorization() -> None:
    subprocess.run(
        ["node", "--test", Path(__file__).with_suffix(".cjs")], cwd=ROOT, check=True
    )


def test_hardware_workflows_require_authorization() -> None:
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        workflow = yaml.safe_load(path.read_text())
        jobs = workflow.get("jobs", {})
        for name, job in jobs.items():
            runner = job.get("runs-on", "")
            if path.name == "omni-xpu-docker-release.yaml" and name == "publish":
                assert "pull_request" not in workflow[True]
                assert "workflow_call" not in workflow[True]
                assert "github.ref == 'refs/heads/main'" in job["if"]
                continue
            if isinstance(runner, dict):
                assert runner["group"] == "sglang-omni-ci", (path, name)
                assert (
                    jobs["authorize"]["uses"]
                    == "./.github/workflows/authorize-self-hosted.yml"
                )
                checkouts = [
                    step["with"]["ref"]
                    for step in job["steps"]
                    if step.get("uses", "").startswith("actions/checkout@")
                ]
                if name == "cleanup":
                    assert checkouts == ["refs/heads/main"]
                    assert (
                        "setup" in job["needs"]
                        and "authorize" in jobs["setup"]["needs"]
                    )
                    assert "needs.setup.result == 'success'" in job["if"]
                    assert "needs.setup.result == 'failure'" in job["if"]
                else:
                    assert checkouts == [
                        "${{ github.event.pull_request.head.sha || github.sha }}"
                    ]
                    needs = job["needs"]
                    assert "authorize" in (
                        needs if isinstance(needs, list) else [needs]
                    ), (path, name)
                    assert "always()" not in job.get("if", "")
            else:
                assert "self-hosted" not in runner, (path, name)
