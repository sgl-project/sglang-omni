# SPDX-License-Identifier: Apache-2.0
"""This file aim to test the model pick logic in the Omni CI workflow.

Author: chenyang zhang https://github.com/zhaochenyang20

In short, if having labels like run-higgs, our CI workflow will
pick the Higgs model for TTS. This file aim to test the logic.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OMNI_WORKFLOW = REPO_ROOT / ".github/workflows/omni-ci.yaml"

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


def _pick_scripts() -> tuple[str, str]:
    jobs = yaml.load(OMNI_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)[
        "jobs"
    ]
    tts = next(
        step["run"]
        for step in jobs["pick-tts-model"]["steps"]
        if step.get("id") == "tts"
    )
    return jobs["pick-asr-model"]["steps"][0]["run"], tts


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
        "PR_LABELS": "[]",
        "RUN_HIGGS_LABEL": "false",
        "RUN_MOSS_LABEL": "false",
        "RUN_QWEN3_TTS_LABEL": "false",
        "RUN_FUN_ASR_LABEL": "false",
        "RUN_QWEN3_ASR_LABEL": "false",
        "RUN_WHISPER_ASR_LABEL": "false",
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
    asr_script, tts_script = _pick_scripts()
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
