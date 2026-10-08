# SPDX-License-Identifier: Apache-2.0
"""CPU-only consistency checks for NPU image and dependency configuration."""

import shutil
from pathlib import Path

import pytest

from scripts.npu.config import read_config

ROOT = Path(__file__).resolve().parents[3]
CONFIG, MATRIX = read_config(ROOT)


@pytest.fixture
def config_root(tmp_path):
    shutil.copy(ROOT / "pyproject_npu.toml", tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "old,new",
    [
        (
            f'sglang-version = "{CONFIG["sglang-version"]}"',
            'sglang-version = "99.0.0"',
        ),
        (f'cann{MATRIX[1]["cann"]}-910b', "cann99.0.0-910b"),
        ("-a3@sha256:", "-910b@sha256:"),
        (CONFIG["base-image-a3"], CONFIG["base-image-a3"].split("@", 1)[0]),
    ],
)
def test_reject_inconsistent_base_images(config_root, old, new):
    path = config_root / "pyproject_npu.toml"
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError):
        read_config(config_root)
