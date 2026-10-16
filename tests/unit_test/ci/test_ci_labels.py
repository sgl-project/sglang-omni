# SPDX-License-Identifier: Apache-2.0
"""Contracts of the CI label rules and of the workflows that apply them."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import TypeAlias

import pytest
import yaml

from tests.test_ci.tts_mps_ci_config import MPS_PERFORMANCE_REFERENCES

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIRECTORY = REPO_ROOT / ".github/workflows"
RESOLVER_PATH = REPO_ROOT / ".github/scripts/resolve_ci_labels.py"
GITHUB_LABEL_MAX_LENGTH = 50
RANDOM_DRAW_SEED_COUNT = 64
UNRELATED_STAGE_LABEL = "run-mac-ci"

WorkflowNode: TypeAlias = str | list["WorkflowNode"] | dict[str, "WorkflowNode"]


def load_module(name: str, path: Path) -> ModuleType:
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RULES = load_module("resolve_ci_labels", RESOLVER_PATH)
ALL_STAGE_NAMES = frozenset(stage.name for stage in RULES.CI_STAGES)
MODEL_STAGES = [stage for stage in RULES.CI_STAGES if stage.preset_by_model_label]
MODEL_LABEL_CASES = [
    pytest.param(stage, model_label, preset, id=model_label)
    for stage in MODEL_STAGES
    for model_label, preset in stage.preset_by_model_label.items()
]
STAGE_BY_NAME = {stage.name: stage for stage in RULES.CI_STAGES}


def resolve(*labels: str, seed: str = "1") -> RULES.CiPlan:
    return RULES.resolve_ci_plan(frozenset(labels), seed)


def run_resolver_cli(*cli_arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RESOLVER_PATH), *cli_arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def load_workflow(file_name: str) -> dict[str, WorkflowNode]:
    return yaml.load(
        (WORKFLOW_DIRECTORY / file_name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )


@pytest.mark.parametrize(
    "labels",
    [(), ("documentation",), ("run-ci",), ("run-tts-higgs-tts-3-4b",)],
    ids=["no-labels", "unrelated-label", "legacy-run-ci", "model-label-only"],
)
def test_no_stage_label_fails_the_gate(labels: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="No CI label found"):
        resolve(*labels)


def test_run_all_selects_every_stage() -> None:
    assert resolve(RULES.RUN_ALL_LABEL).enabled_stage_names == ALL_STAGE_NAMES


@pytest.mark.parametrize("stage", RULES.CI_STAGES, ids=lambda stage: stage.label)
def test_a_stage_label_selects_only_its_stage(stage: RULES.CiStage) -> None:
    assert resolve(stage.label).enabled_stage_names == {stage.name}


def test_stage_labels_combine() -> None:
    plan = resolve("run-asr-ci", "run-intel-cpu-ci", "documentation")
    assert plan.enabled_stage_names == {"asr", "intel_cpu"}


@pytest.mark.parametrize("use_run_all", [False, True], ids=["stage-label", "run-all"])
@pytest.mark.parametrize("stage,model_label,preset", MODEL_LABEL_CASES)
def test_a_model_label_picks_the_model_of_its_selected_stage(
    stage: RULES.CiStage, model_label: str, preset: str, use_run_all: bool
) -> None:
    stage_selector = RULES.RUN_ALL_LABEL if use_run_all else stage.label
    plan = resolve(stage_selector, model_label)
    assert plan.preset_by_stage_name[stage.name] == preset
    assert plan.ignored_model_labels == ()


@pytest.mark.parametrize("stage,model_label,preset", MODEL_LABEL_CASES)
def test_a_model_label_without_its_stage_is_ignored(
    stage: RULES.CiStage, model_label: str, preset: str
) -> None:
    plan = resolve(UNRELATED_STAGE_LABEL, model_label)
    assert plan.enabled_stage_names == {"mac"}
    assert stage.name not in plan.preset_by_stage_name
    assert plan.ignored_model_labels == (model_label,)


@pytest.mark.parametrize("stage", MODEL_STAGES, ids=lambda stage: stage.name)
def test_a_selected_stage_without_a_model_label_draws_every_random_preset(
    stage: RULES.CiStage,
) -> None:
    drawn_presets = {
        resolve(stage.label, seed=str(seed)).preset_by_stage_name[stage.name]
        for seed in range(RANDOM_DRAW_SEED_COUNT)
    }
    assert drawn_presets == set(stage.random_presets)


def test_the_random_draw_is_stable_for_one_seed() -> None:
    assert resolve(RULES.RUN_ALL_LABEL, seed="123") == resolve(
        RULES.RUN_ALL_LABEL, seed="123"
    )


def test_tts_custom_voice_runs_only_by_its_model_label() -> None:
    tts = STAGE_BY_NAME["tts"]
    assert "qwen3-tts-custom-voice" not in tts.random_presets
    plan = resolve("run-tts-ci", "run-tts-qwen3-tts-12hz-1.7b-customvoice")
    assert plan.preset_by_stage_name["tts"] == "qwen3-tts-custom-voice"


@pytest.mark.parametrize(
    "is_stage_selected", [True, False], ids=["selected", "unselected"]
)
@pytest.mark.parametrize("stage", MODEL_STAGES, ids=lambda stage: stage.name)
def test_two_model_labels_in_one_stage_fail(
    stage: RULES.CiStage, is_stage_selected: bool
) -> None:
    stage_selector = stage.label if is_stage_selected else UNRELATED_STAGE_LABEL
    first_label, second_label = list(stage.preset_by_model_label)[:2]
    with pytest.raises(ValueError, match="Keep only one"):
        resolve(stage_selector, first_label, second_label)


def test_label_names_are_unique_and_name_their_stage() -> None:
    all_labels = [RULES.RUN_ALL_LABEL]
    for stage in RULES.CI_STAGES:
        assert stage.label == f"run-{stage.name.replace('_', '-')}-ci"
        assert all(
            model_label.startswith(f"run-{stage.name}-")
            for model_label in stage.preset_by_model_label
        )
        assert set(stage.random_presets) <= set(stage.preset_by_model_label.values())
        all_labels += [stage.label, *stage.preset_by_model_label]
    assert len(all_labels) == len(set(all_labels))
    assert max(len(label) for label in all_labels) <= GITHUB_LABEL_MAX_LENGTH


def test_every_mps_preset_has_a_pool_config_and_references() -> None:
    mps_presets = set(STAGE_BY_NAME["mps"].preset_by_model_label.values())
    assert set(RULES.MPS_CONFIG_BY_PRESET) == mps_presets
    assert mps_presets <= set(MPS_PERFORMANCE_REFERENCES)
    for config_path in RULES.MPS_CONFIG_BY_PRESET.values():
        assert (REPO_ROOT / config_path).is_file()


def test_resolve_cli_writes_the_plan_to_github_output(tmp_path: Path) -> None:
    github_output = tmp_path / "github_output"
    labels = ["run-mps-ci", "run-mps-moss-tts-local-transformer-v1.5"]
    result = run_resolver_cli(
        "resolve",
        "--labels-json",
        json.dumps(labels),
        "--seed",
        "1",
        "--github-output",
        str(github_output),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = dict(
        line.split("=", 1) for line in github_output.read_text().splitlines()
    )
    assert outputs == {
        "run_asr": "false",
        "asr_ci_model": "",
        "run_tts": "false",
        "tts_ci_model": "",
        "run_mps": "true",
        "mps_ci_model": "moss",
        "run_omni": "false",
        "omni_ci_model": "",
        "run_intel_cpu": "false",
        "run_intel_xpu": "false",
        "run_mac": "false",
        "mps_ci_config": RULES.MPS_CONFIG_BY_PRESET["moss"],
    }


def test_resolve_cli_warns_about_an_ignored_model_label(tmp_path: Path) -> None:
    result = run_resolver_cli(
        "resolve",
        "--labels-json",
        json.dumps(["run-asr-ci", "run-omni-minicpm-o-4.5"]),
        "--seed",
        "1",
        "--github-output",
        str(tmp_path / "github_output"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "::warning::Ignoring run-omni-minicpm-o-4.5" in result.stdout


def test_resolve_cli_fails_without_a_stage_label(tmp_path: Path) -> None:
    github_output = tmp_path / "github_output"
    result = run_resolver_cli(
        "resolve",
        "--labels-json",
        "[]",
        "--seed",
        "1",
        "--github-output",
        str(github_output),
    )
    assert result.returncode == 1
    assert "::error::No CI label found" in result.stdout
    assert not github_output.exists()


@pytest.mark.parametrize(
    "labels,expected_workflows",
    [
        ([], ["CPU CI", "Omni CI", "Voxt Mac CI", "XPU CI"]),
        (["run-tts-higgs-tts-3-4b"], ["CPU CI", "Omni CI", "Voxt Mac CI", "XPU CI"]),
        (["run-tts-ci"], ["CPU CI", "Voxt Mac CI", "XPU CI"]),
        (["run-mac-ci"], ["CPU CI", "XPU CI"]),
        (["run-all-ci"], []),
    ],
)
def test_workflows_to_cancel_are_those_no_label_opts_into(
    labels: list[str], expected_workflows: list[str]
) -> None:
    result = run_resolver_cli(
        "workflows-to-cancel", "--labels-json", json.dumps(labels)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == expected_workflows


def test_every_gated_workflow_name_exists() -> None:
    workflow_names = {
        load_workflow(path.name)["name"] for path in WORKFLOW_DIRECTORY.glob("*.y*ml")
    }
    assert RULES.GATED_WORKFLOWS <= workflow_names


def test_no_workflow_reads_the_legacy_run_ci_label() -> None:
    for path in WORKFLOW_DIRECTORY.glob("*.y*ml"):
        assert "'run-ci'" not in path.read_text(encoding="utf-8"), path.name


@pytest.mark.parametrize(
    "job_id,stage_name,model_input",
    [
        ("asr-ci", "asr", "asr_ci_model"),
        ("tts-ci", "tts", "tts_ci_model"),
        ("mps-ci", "mps", "mps_ci_model"),
        ("qwen3-omni-ci", "omni", "omni_ci_model"),
    ],
)
def test_omni_ci_runs_a_model_suite_only_when_its_stage_is_selected(
    job_id: str, stage_name: str, model_input: str
) -> None:
    jobs = load_workflow("omni-ci.yaml")["jobs"]
    preflight_outputs = jobs["preflight"]["outputs"]
    job = jobs[job_id]
    assert f"run_{stage_name}" in preflight_outputs
    assert f"needs.preflight.outputs.run_{stage_name} == 'true'" in job["if"]
    assert job["with"][model_input] == (
        f"${{{{ needs.preflight.outputs.{model_input} }}}}"
    )


def test_omni_ci_runs_unit_tests_for_any_stage_label() -> None:
    jobs = load_workflow("omni-ci.yaml")["jobs"]
    assert jobs["pr-test"]["if"] == "needs.setup.result == 'success'"
    assert jobs["setup"]["if"] == "needs.preflight.result == 'success'"
    assert "resolve_ci_labels.py resolve" in yaml.safe_dump(jobs["preflight"])


@pytest.mark.parametrize(
    "file_name,stage_name,job_ids",
    [
        ("omni-cpu-ci.yaml", "intel_cpu", ["cpu-ci"]),
        ("omni-xpu-ci.yaml", "intel_xpu", ["xpu-ci"]),
        ("voxt-mac-ci.yaml", "mac", ["native-runtime", "voxt-unit"]),
    ],
)
def test_hardware_ci_runs_only_when_its_stage_is_selected(
    file_name: str, stage_name: str, job_ids: list[str]
) -> None:
    jobs = load_workflow(file_name)["jobs"]
    preflight = jobs["preflight"]
    assert "resolve_ci_labels.py resolve" in yaml.safe_dump(preflight)
    assert f"steps.plan.outputs.run_{stage_name}" in (
        preflight["outputs"][f"run_{stage_name}"]
    )
    for job_id in job_ids:
        assert f"needs.preflight.outputs.run_{stage_name} == 'true'" in (
            jobs[job_id]["if"]
        )


def test_omni_ci_keeps_process_replicas_qwen_only() -> None:
    child = load_workflow("test-qwen3-omni-ci.yaml")
    assert child["env"]["OMNI_CI_MODEL"] == "${{ inputs.omni_ci_model }}"
    assert "inputs.omni_ci_model == 'qwen3-omni'" in (
        child["jobs"]["stage-11-process-replicas"]["if"]
    )
