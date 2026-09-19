# SPDX-License-Identifier: Apache-2.0
"""CPU-only consistency checks for NPU image and dependency configuration."""

import re
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.npu.config import check_installed, check_requirements, read_config

ROOT = Path(__file__).resolve().parents[3]
CONFIG, MATRIX = read_config(ROOT)


@pytest.fixture
def config_root(tmp_path):
    shutil.copy(ROOT / "pyproject_npu.toml", tmp_path)
    (tmp_path / "scripts/npu").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/npu/requirements.txt", tmp_path / "scripts/npu")
    return tmp_path


def test_repository_configuration_and_dependency_pins():
    config, matrix = read_config(ROOT)
    assert [item["device"] for item in matrix] == ["a3", "910b"]
    assert matrix[0]["base_image"] == config["base-image-a3"]
    check_requirements(ROOT)


@pytest.mark.parametrize(
    "old,new",
    [
        (
            f'sglang-version = "{CONFIG["sglang-version"]}"',
            'sglang-version = "99.0.0"',
        ),
        (f'cann{MATRIX[1]["cann"]}-910b', "cann99.0.0-910b"),
        ("-a3@sha256:", "-910b@sha256:"),
        ("@sha256:", "@invalid:"),
    ],
)
def test_reject_inconsistent_base_images(config_root, old, new):
    path = config_root / "pyproject_npu.toml"
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError):
        read_config(config_root)


@pytest.mark.parametrize("replacement", ["transformers==0.0.1", "transformers>=5", ""])
def test_reject_incompatible_unpinned_or_missing_dependency(config_root, replacement):
    path = config_root / "scripts/npu/requirements.txt"
    path.write_text(
        re.sub(r"^transformers==[^\n]+", replacement, path.read_text(), flags=re.M)
    )
    with pytest.raises(ValueError):
        check_requirements(config_root)


def test_actual_base_version_must_match_config():
    config, _ = read_config(ROOT)
    with patch("importlib.metadata.version", return_value="0.0.1"):
        with pytest.raises(ValueError, match="Installed SGLang"):
            check_installed(config)
    with patch("importlib.metadata.version", return_value=config["sglang-version"]):
        check_installed(config)
