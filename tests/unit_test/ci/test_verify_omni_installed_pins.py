# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFIER = REPO_ROOT / ".github/scripts/verify_omni_installed_pins.py"


@pytest.fixture(scope="module")
def verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_omni_installed_pins", VERIFIER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_project(tmp_path: Path) -> Path:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        """
[project]
dependencies = [
  "torch==2.13.0; sys_platform != 'darwin' or platform_machine != 'arm64'",
  "torch==2.11.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
  "torchcodec==0.15.0; sys_platform != 'darwin' or platform_machine != 'arm64'",
  "torchcodec==0.11.1; sys_platform == 'darwin' and platform_machine == 'arm64'",
  "flashinfer_python[cu13]==0.6.17; sys_platform != 'darwin' or platform_machine != 'arm64'",
  "transformers==5.12.1",
]

[tool.uv]
override-dependencies = [
  "protobuf==6.33.6; sys_platform != 'darwin' or platform_machine != 'arm64'",
  "protobuf==7.36.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return project


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (
            {"sys_platform": "linux", "platform_machine": "x86_64"},
            {
                "torch": "2.13.0",
                "torchcodec": "0.15.0",
                "flashinfer_python": "0.6.17",
                "transformers": "5.12.1",
                "protobuf": "6.33.6",
            },
        ),
        (
            {"sys_platform": "darwin", "platform_machine": "arm64"},
            {
                "torch": "2.11.0",
                "torchcodec": "0.11.1",
                "transformers": "5.12.1",
                "protobuf": "7.36.0",
            },
        ),
    ],
)
def test_exact_pins_respect_platform_markers(
    verifier, tmp_path: Path, environment: dict[str, str], expected: dict[str, str]
) -> None:
    assert (
        verifier._exact_pins(_write_project(tmp_path), environment=environment)
        == expected
    )


def test_exact_pins_preserve_override_precedence(verifier, tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        """
[project]
dependencies = [
  "torch==2.13.0; sys_platform == 'linux'",
]

[tool.uv]
override-dependencies = ["torch==2.12.0; sys_platform == 'linux'"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert verifier._exact_pins(
        project,
        environment={"sys_platform": "linux", "platform_machine": "x86_64"},
    ) == {"torch": "2.12.0"}


def test_exact_pins_select_non_arm_darwin_variant(verifier, tmp_path: Path) -> None:
    pins = verifier._exact_pins(
        _write_project(tmp_path),
        environment={"sys_platform": "darwin", "platform_machine": "x86_64"},
    )

    assert pins["torch"] == "2.13.0"
    assert pins["torchcodec"] == "0.15.0"
    assert pins["flashinfer_python"] == "0.6.17"
    assert pins["protobuf"] == "6.33.6"
