# SPDX-License-Identifier: Apache-2.0
"""This file aim to test the model pick logic in the Omni CI workflow.

Author: chenyang zhang https://github.com/zhaochenyang20

In short, if having labels like run-higgs, our CI workflow will
pick the Higgs model for TTS. This file aim to test the logic.
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OMNI_WORKFLOW = REPO_ROOT / ".github/workflows/omni-ci.yaml"
OMNI_MODEL_WORKFLOW = REPO_ROOT / ".github/workflows/test-qwen3-omni-ci.yaml"

TTS_LABELS = {
    "higgs": "RUN_HIGGS_LABEL",
    "moss": "RUN_MOSS_LABEL",
    "qwen3-tts": "RUN_QWEN3_TTS_LABEL",
}
ASR_LABELS = {
    "fun": "RUN_FUN_ASR_LABEL",
    "qwen3": "RUN_QWEN3_ASR_LABEL",
    "whisper": "RUN_WHISPER_ASR_LABEL",
}


def _pick_scripts() -> tuple[str, str, str]:
    jobs = yaml.load(OMNI_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)[
        "jobs"
    ]
    tts = next(
        step["run"]
        for step in jobs["pick-tts-model"]["steps"]
        if step.get("id") == "tts"
    )
    return (
        jobs["pick-asr-model"]["steps"][0]["run"],
        tts,
        jobs["pick-omni-model"]["steps"][0]["run"],
    )


def _run_one(
    script: str, tmp_path: Path, labels: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    github_output = tmp_path / "github_output"
    github_output.touch()
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(github_output),
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "sgl-project/sglang-omni",
        "EXACT_SHA": "deadbeef",
        "TTS_CI_MODEL_OVERRIDE": "",
        "ASR_CI_MODEL_OVERRIDE": "",
        "OMNI_CI_MODEL_OVERRIDE": "",
        "PR_LABELS": "[]",
        "RUN_HIGGS_LABEL": "false",
        "RUN_MOSS_LABEL": "false",
        "RUN_QWEN3_TTS_LABEL": "false",
        "RUN_FUN_ASR_LABEL": "false",
        "RUN_QWEN3_ASR_LABEL": "false",
        "RUN_WHISPER_ASR_LABEL": "false",
        "RUN_QWEN3_OMNI_LABEL": "false",
        "RUN_MINICPMO_LABEL": "false",
        **labels,
    }
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_both(
    tmp_path: Path, labels: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], subprocess.CompletedProcess[str]]:
    asr_script, tts_script, _ = _pick_scripts()
    asr_home = tmp_path / "asr"
    tts_home = tmp_path / "tts"
    asr_home.mkdir()
    tts_home.mkdir()
    return _run_one(asr_script, asr_home, labels), _run_one(
        tts_script, tts_home, labels
    )


def _assert_asr_random(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Random seed for ASR CI model:" in result.stdout
    assert "Selected ASR CI model:" in result.stdout


def _assert_tts_random(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Selection digest for TTS CI model:" in result.stdout
    assert "Selected TTS CI model:" in result.stdout


def _assert_asr_specified(result: subprocess.CompletedProcess[str], model: str) -> None:
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"Selected ASR CI model: {model}" in result.stdout
    assert "Random seed for ASR CI model:" not in result.stdout


def _assert_tts_specified(result: subprocess.CompletedProcess[str], model: str) -> None:
    assert result.returncode == 0, result.stderr + result.stdout
    assert f"Selected TTS CI model: {model}" in result.stdout
    assert "Selection digest for TTS CI model:" not in result.stdout


def test_both_picks_are_random_without_labels(tmp_path: Path) -> None:
    asr, tts = _run_both(tmp_path, {})
    _assert_asr_random(asr)
    _assert_tts_random(tts)


@pytest.mark.parametrize("asr_model,asr_label", list(ASR_LABELS.items()))
@pytest.mark.parametrize("tts_model,tts_label", list(TTS_LABELS.items()))
def test_both_picks_honor_labels(
    tmp_path: Path,
    asr_model: str,
    asr_label: str,
    tts_model: str,
    tts_label: str,
) -> None:
    asr, tts = _run_both(tmp_path, {asr_label: "true", tts_label: "true"})
    _assert_asr_specified(asr, asr_model)
    _assert_tts_specified(tts, tts_model)


@pytest.mark.parametrize("tts_model,tts_label", list(TTS_LABELS.items()))
def test_specified_tts_leaves_asr_random(
    tmp_path: Path, tts_model: str, tts_label: str
) -> None:
    asr, tts = _run_both(tmp_path, {tts_label: "true"})
    _assert_asr_random(asr)
    _assert_tts_specified(tts, tts_model)


@pytest.mark.parametrize("asr_model,asr_label", list(ASR_LABELS.items()))
def test_specified_asr_leaves_tts_random(
    tmp_path: Path, asr_model: str, asr_label: str
) -> None:
    asr, tts = _run_both(tmp_path, {asr_label: "true"})
    _assert_asr_specified(asr, asr_model)
    _assert_tts_random(tts)


@pytest.mark.parametrize(
    "labels,expected",
    [
        ({}, "qwen3-omni"),
        ({"RUN_QWEN3_OMNI_LABEL": "true"}, "qwen3-omni"),
        ({"RUN_MINICPMO_LABEL": "true"}, "minicpmo"),
        ({"OMNI_CI_MODEL_OVERRIDE": "minicpmo"}, "minicpmo"),
        ({"OMNI_CI_MODEL_OVERRIDE": "QWEN3-OMNI"}, "qwen3-omni"),
        (
            {"RUN_MINICPMO_LABEL": "true", "OMNI_CI_MODEL_OVERRIDE": "qwen3-omni"},
            "qwen3-omni",
        ),
    ],
)
def test_omni_pick_honors_default_labels_and_override(
    tmp_path: Path, labels: dict[str, str], expected: str
) -> None:
    result = _run_one(_pick_scripts()[2], tmp_path, labels)
    assert result.returncode == 0, result.stderr + result.stdout
    assert (tmp_path / "github_output").read_text() == f"omni_ci_model={expected}\n"


@pytest.mark.parametrize(
    "labels",
    [
        {"OMNI_CI_MODEL_OVERRIDE": "unknown"},
        {"RUN_QWEN3_OMNI_LABEL": "true", "RUN_MINICPMO_LABEL": "true"},
        {
            "RUN_QWEN3_OMNI_LABEL": "true",
            "RUN_MINICPMO_LABEL": "true",
            "OMNI_CI_MODEL_OVERRIDE": "minicpmo",
        },
    ],
)
def test_omni_pick_rejects_invalid_or_conflicting_selection(
    tmp_path: Path, labels: dict[str, str]
) -> None:
    result = _run_one(_pick_scripts()[2], tmp_path, labels)
    assert result.returncode != 0
    assert (tmp_path / "github_output").read_text() == ""


def test_omni_label_does_not_change_asr_or_tts_selection(tmp_path: Path) -> None:
    asr, tts = _run_both(tmp_path, {"RUN_MINICPMO_LABEL": "true"})
    _assert_asr_random(asr)
    _assert_tts_random(tts)


def test_workflow_passes_selected_model_and_keeps_pcm_qwen_only() -> None:
    workflow = yaml.load(OMNI_WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["qwen3-omni-ci"]
    assert "pick-omni-model" in job["needs"]
    assert "needs.pick-omni-model.result == 'success'" in job["if"]
    assert job["with"]["omni_ci_model"] == (
        "${{ needs.pick-omni-model.outputs.omni_ci_model }}"
    )
    child = yaml.load(OMNI_MODEL_WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    assert child["env"]["OMNI_CI_MODEL"] == "${{ inputs.omni_ci_model }}"
    assert child["on"]["workflow_call"]["inputs"]["omni_ci_model"]["default"] == (
        "qwen3-omni"
    )
    assert "inputs.omni_ci_model == 'qwen3-omni'" in (
        child["jobs"]["stage-11-process-replicas"]["if"]
    )


@pytest.fixture
def slash_handler(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setitem(
        sys.modules, "github", SimpleNamespace(Auth=Mock(), Github=Mock())
    )
    monkeypatch.setitem(
        sys.modules,
        "github.GithubException",
        SimpleNamespace(GithubException=Exception),
    )
    spec = spec_from_file_location(
        "slash_command_handler", REPO_ROOT / "scripts/ci/utils/slash_command_handler.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slash_targets_support_all_three_families(slash_handler: ModuleType) -> None:
    assert slash_handler.parse_model_targets(
        "/tag-and-rerun-ci moss qwen3-asr minicpmo".split()
    ) == ("moss", "qwen3-asr", "minicpmo", None)


@pytest.mark.parametrize(
    "targets", ["higgs moss", "fun-asr qwen3-asr", "minicpmo qwen3-omni"]
)
def test_slash_targets_reject_conflicting_models(
    slash_handler: ModuleType, targets: str
) -> None:
    *models, error = slash_handler.parse_model_targets(
        f"/tag-run-ci-label {targets}".split()
    )
    assert models == [None, None, None]
    assert error is not None


@pytest.mark.parametrize(
    "selected,previous", [("minicpmo", "qwen3-omni"), ("qwen3-omni", "minicpmo")]
)
def test_slash_tag_replaces_only_omni_label(
    slash_handler: ModuleType, selected: str, previous: str
) -> None:
    pr, comment = Mock(), Mock()
    pr.get_labels.return_value = [
        SimpleNamespace(name=label)
        for label in (f"run-{previous}", "run-moss", "run-qwen3-asr")
    ]
    assert slash_handler.handle_tag_run_ci(
        pr, comment, {"can_tag_run_ci_label": True}, omni_model_target=selected
    )
    pr.remove_from_labels.assert_called_once_with(f"run-{previous}")
    assert pr.add_to_labels.call_args_list == [call("run-ci"), call(f"run-{selected}")]
    comment.create_reaction.assert_called_once_with("+1")


@pytest.mark.parametrize("command", ["/tag-run-ci-label", "/tag-and-rerun-ci"])
def test_slash_commands_forward_omni_target(
    slash_handler: ModuleType, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    environment = {
        "GITHUB_TOKEN": "test-token",
        "REPO_FULL_NAME": "owner/repo",
        "PR_NUMBER": "1",
        "COMMENT_ID": "2",
        "COMMENT_BODY": f"{command} minicpmo",
        "USER_LOGIN": "contributor",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        slash_handler,
        "load_permissions",
        Mock(return_value={"can_tag_run_ci_label": True, "can_rerun_failed_ci": True}),
    )
    tagged = Mock(return_value=True)
    rerun = Mock(return_value=True)
    monkeypatch.setattr(slash_handler, "handle_tag_run_ci", tagged)
    monkeypatch.setattr(slash_handler, "handle_rerun_failed_ci", rerun)
    monkeypatch.setattr(slash_handler.time, "sleep", Mock())
    slash_handler.main()
    assert tagged.call_args.kwargs["omni_model_target"] == "minicpmo"
    if command == "/tag-and-rerun-ci":
        assert rerun.call_args.kwargs["force_full_omni_ci_rerun"] is True
    else:
        rerun.assert_not_called()
