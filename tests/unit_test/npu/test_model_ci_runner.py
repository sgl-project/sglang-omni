# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete integration settings before accessing Docker or NPUs."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/npu/run_model_ci.sh"


@pytest.mark.parametrize(
    "key,value",
    [
        ("NPU_CI_IMAGE", "image:latest"),
        ("NPU_CI_DEVICE", "0,1"),
        ("NPU_CI_DEVICE", "$(touch unsafe)"),
        ("NPU_CI_DATA_DIR", "/"),
        ("NPU_CI_MODEL", "unknown"),
    ],
)
def test_invalid_configuration_is_rejected(key, value, tmp_path):
    env = {
        **os.environ,
        "NPU_CI_IMAGE": "test@sha256:" + "a" * 64,
        "NPU_CI_DEVICE": "0",
        "NPU_CI_DATA_DIR": str(tmp_path),
        "NPU_CI_MODEL": "qwen3-tts",
        key: value,
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert not list(tmp_path.iterdir())
