# SPDX-License-Identifier: Apache-2.0
"""CPU-only consistency checks for NPU image and dependency configuration."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.npu.config import read_config

ROOT = Path(__file__).resolve().parents[3]
CONFIG, MATRIX = read_config(ROOT)


def test_release_matrix_selects_matching_base_images() -> None:
    assert {row["device"]: row["base_image"] for row in MATRIX} == {
        "a3": CONFIG["base-image-a3"],
        "950": CONFIG["base-image-950"],
        "910b": "sglang-a2-base",
    }
    (cann_version,) = {row["cann"] for row in MATRIX}
    assert f"-cann{cann_version}-a3@" in CONFIG["base-image-a3"]


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
        (f'cann{MATRIX[1]["cann"]}-950', "cann99.0.0-950"),
        ("-a3@sha256:", "-950@sha256:"),
        (CONFIG["base-image-a3"], CONFIG["base-image-a3"].split("@", 1)[0]),
    ],
)
def test_reject_inconsistent_base_images(config_root, old, new):
    path = config_root / "pyproject_npu.toml"
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError):
        read_config(config_root)


@pytest.mark.parametrize(
    "config_field", ["sglang-dockerfile-sha256", "cann-image-910b"]
)
@pytest.mark.parametrize("replacement", ['{config_field} = "invalid"', ""])
def test_a2_pins_do_not_block_shared_config(
    config_root: Path, config_field: str, replacement: str
) -> None:
    """An invalid A2 pin used to block the installer and every device's CI job."""
    manifest_path = config_root / "pyproject_npu.toml"
    manifest_path.write_text(
        re.sub(
            rf"^{config_field} = .*?$",
            replacement.format(config_field=config_field),
            manifest_path.read_text(),
            flags=re.MULTILINE,
        )
    )
    script_directory = config_root / "scripts" / "npu"
    script_directory.mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "npu" / "config.py", script_directory)
    command = [sys.executable, str(script_directory / "config.py")]

    version = subprocess.run(
        [*command, "--get", "sglang-version"],
        capture_output=True,
        text=True,
        check=True,
    )
    matrix = subprocess.run(
        [*command, "--matrix"], capture_output=True, text=True, check=True
    )
    a2_pin = subprocess.run(
        [*command, "--get", config_field], capture_output=True, text=True, check=False
    )

    assert version.stdout.strip() == CONFIG["sglang-version"]
    assert {row["device"] for row in json.loads(matrix.stdout)["include"]} == {
        "a3",
        "950",
        "910b",
    }
    assert a2_pin.returncode != 0
    assert config_field in a2_pin.stderr
