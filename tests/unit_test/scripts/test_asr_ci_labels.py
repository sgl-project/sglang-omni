# SPDX-License-Identifier: Apache-2.0
"""The ARK-ASR model label must reach the ASR workflow without selecting a stage."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LABEL_RESOLVER = REPO_ROOT / ".github/scripts/resolve_ci_labels.py"


@pytest.mark.parametrize(
    "labels, expected_status, expected_output",
    [
        (
            ["run-asr-ci", "run-ark-asr-3b"],
            0,
            ["run_asr=true", "asr_ci_model=ark", "run_self_hosted=true"],
        ),
        (
            ["run-ark-asr-3b"],
            1,
            ["No CI label found"],
        ),
        (
            ["run-asr-ci", "run-ark-asr-3b", "run-fun-asr-nano-2512"],
            1,
            ["run-ark-asr-3b", "run-fun-asr-nano-2512"],
        ),
    ],
)
def test_arkasr_label_selects_only_the_requested_asr_preset(
    labels: list[str],
    expected_status: int,
    expected_output: list[str],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "github-output"
    result = subprocess.run(
        [
            sys.executable,
            str(LABEL_RESOLVER),
            "resolve",
            "--labels-json",
            json.dumps(labels),
            "--seed",
            "arkasr-ci",
            "--github-output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected_status, result.stdout + result.stderr
    for expected in expected_output:
        assert expected in result.stdout
    if expected_status == 0:
        assert "asr_ci_model=ark" in output_path.read_text().splitlines()
    else:
        assert not output_path.exists()
