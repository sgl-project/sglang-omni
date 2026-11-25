# SPDX-License-Identifier: Apache-2.0
"""Mac Metal CI's label gate requires run-mac-metal-ci and skips Markdown-only PRs."""

import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load(
    (REPO_ROOT / ".github/workflows/mac-metal-ci.yaml").read_text()
)
STEPS = {step["name"]: step for step in WORKFLOW["jobs"]["preflight"]["steps"]}


@pytest.fixture(scope="module")
def resolver() -> Iterator[ModuleType]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(REPO_ROOT / ".github/scripts"))
        yield importlib.import_module("resolve_ci_labels")
        sys.modules.pop("resolve_ci_labels")


def resolve_stage_step(
    tmp_path: Path, labels: list[str], has_mlx_changes: str
) -> tuple[subprocess.CompletedProcess[str], str]:
    output_path = tmp_path / "github_output"
    output_path.write_text("")
    environment = dict(
        os.environ,
        LABELS_JSON=json.dumps(labels),
        HAS_MLX_CHANGES=has_mlx_changes,
        GITHUB_RUN_ID="1",
        GITHUB_OUTPUT=str(output_path),
    )
    result = subprocess.run(
        ["bash", "-e", "-c", STEPS["Resolve CI stages"]["run"]],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    return result, output_path.read_text()


def test_mac_metal_label_selects_only_mac_metal_workflow(
    resolver: ModuleType,
) -> None:
    labels = frozenset({"run-mac-metal-ci"})
    assert resolver.select_workflows(labels) == frozenset({"Mac Metal CI"})
    assert resolver.build_labels_by_target()["mac-metal"] == labels


def test_markdown_only_pr_without_label_passes_and_skips_metal_job(
    tmp_path: Path,
) -> None:
    result, outputs = resolve_stage_step(tmp_path, [], "false")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_mac_metal=false" in outputs.splitlines()
    assert "::error::" not in result.stdout


def test_code_pr_without_label_fails_gate(tmp_path: Path) -> None:
    result, _ = resolve_stage_step(tmp_path, [], "true")
    assert result.returncode == 1
    assert "::error::No CI label found" in result.stdout


@pytest.mark.parametrize(
    ("labels", "expected_output"),
    [
        (["run-mac-metal-ci"], "run_mac_metal=true"),
        (["run-mac-ci"], "run_mac_metal=false"),
    ],
)
def test_label_resolution_decouples_mac_and_mac_metal(
    tmp_path: Path, labels: list[str], expected_output: str
) -> None:
    result, outputs = resolve_stage_step(tmp_path, labels, "true")
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected_output in outputs.splitlines()


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
@pytest.mark.parametrize(
    ("changed_paths", "expected_has_mlx_changes"),
    [
        (["sglang_omni_mlx/native/README.md"], "false"),
        (["sglang_omni_mlx/native/src/qwen3_asr/qwen3_asr_server.cpp"], "true"),
        ([".github/workflows/mac-metal-ci.yaml"], "true"),
    ],
)
def test_has_mlx_changes_output(
    changed_paths: list[str], expected_has_mlx_changes: str
) -> None:
    script = STEPS["Fetch latest PR gate state"]["with"]["script"]
    harness = f"""
const files = {json.dumps([{"filename": path} for path in changed_paths])};
const outputs = {{}};
const github = {{
  rest: {{ pulls: {{ get: async () => ({{ data: {{ labels: [], draft: false, head: {{ sha: "abc123" }} }} }}), listFiles: "listFiles" }} }},
  paginate: async () => files,
}};
const context = {{ repo: {{ owner: "o", repo: "r" }}, issue: {{ number: 1 }} }};
const core = {{ setOutput: (name, value) => {{ outputs[name] = String(value); }} }};
(async () => {{ {script} }})().then(() => console.log(outputs.has_mlx_changes));
"""
    result = subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == expected_has_mlx_changes
