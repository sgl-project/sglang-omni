# SPDX-License-Identifier: Apache-2.0
"""Mac CI's label gate lets a Voxt documentation-only pull request pass without a CI label."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((REPO_ROOT / ".github/workflows/mac-ci.yaml").read_text())
STEPS = {step["name"]: step for step in WORKFLOW["jobs"]["preflight"]["steps"]}


def resolve(tmp_path, labels, voxt_docs_only):
    output = tmp_path / "github_output"
    output.write_text("")
    env = dict(
        os.environ,
        LABELS_JSON=json.dumps(labels),
        VOXT_DOCS_ONLY=voxt_docs_only,
        GITHUB_RUN_ID="1",
        GITHUB_OUTPUT=str(output),
    )
    result = subprocess.run(
        ["bash", "-e", "-c", STEPS["Resolve CI stages"]["run"]],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return result, output.read_text()


def test_voxt_documentation_pull_request_without_a_label_passes_and_runs_no_mac_job(
    tmp_path,
):
    result, outputs = resolve(tmp_path, [], "true")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_mac=false" in outputs.splitlines()
    assert "::error::" not in result.stdout


def test_other_pull_request_without_a_label_still_fails(tmp_path):
    result, _ = resolve(tmp_path, [], "false")
    assert result.returncode == 1
    assert "::error::No CI label found" in result.stdout


def test_a_label_still_resolves_as_before(tmp_path):
    result, outputs = resolve(tmp_path, ["run-mac-ci"], "true")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_mac=true" in outputs.splitlines()


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
@pytest.mark.parametrize(
    ("paths", "voxt_docs_only", "helper_selected"),
    [
        (["Voxt/main.go", "Voxt/CHANGELOG.md"], "false", "true"),
        (["Voxt/helper/README.md"], "true", "false"),
        (["Voxt/helper/Sources/Helper/Mic.swift"], "false", "true"),
        (["Voxt/main.go", "sglang_omni_mlx/native/runtime.cc"], "false", "true"),
        (["Voxt/main.go", ".github/workflows/mac-ci.yaml"], "false", "true"),
        (["Voxt/third_party/mygo-patches/0001.patch"], "false", "true"),
        (["Voxt/scripts/vendor-mygo.sh"], "false", "true"),
        ([".github/workflows/mac-ci.yaml"], "false", "true"),
    ],
)
def test_voxt_selection_outputs(paths, voxt_docs_only, helper_selected):
    script = STEPS["Fetch latest PR gate state"]["with"]["script"]
    harness = f"""
const files = {json.dumps([{"filename": p} for p in paths])};
const outputs = {{}};
const github = {{
  rest: {{ pulls: {{ get: async () => ({{ data: {{ labels: [], draft: false }} }}), listFiles: "listFiles" }} }},
  paginate: async () => files,
}};
const context = {{ repo: {{ owner: "o", repo: "r" }}, issue: {{ number: 1 }} }};
const core = {{ setOutput: (name, value) => {{ outputs[name] = String(value); }} }};
(async () => {{ {script} }})().then(() => console.log(JSON.stringify(outputs)));
"""
    result = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, check=True
    )
    outputs = json.loads(result.stdout)
    assert outputs["voxt_docs_only"] == voxt_docs_only
    assert outputs["has_voxt_helper_changes"] == helper_selected
