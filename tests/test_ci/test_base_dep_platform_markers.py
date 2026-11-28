"""Pynini-backed base dependencies must skip Apple arm64 and Linux aarch64."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"
EXPECTED_SPECIFIERS = {"dots.tts": "==0.2.1", "nemo_text_processing": "==1.2.0"}


@pytest.mark.parametrize(
    "sys_platform, platform_machine, is_installed",
    [
        ("linux", "aarch64", False),
        ("linux", "x86_64", True),
        ("darwin", "arm64", False),
        ("darwin", "x86_64", True),
    ],
)
def test_pynini_dependencies_skip_arm_platforms(
    sys_platform: str, platform_machine: str, is_installed: bool
) -> None:
    dependencies = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    requirements = [Requirement(dependency) for dependency in dependencies]
    environment = {"sys_platform": sys_platform, "platform_machine": platform_machine}
    for name, specifier in EXPECTED_SPECIFIERS.items():
        matches = [
            requirement for requirement in requirements if requirement.name == name
        ]
        assert len(matches) == 1, name
        requirement = matches[0]
        assert str(requirement.specifier) == specifier
        assert requirement.marker is not None
        assert requirement.marker.evaluate(environment) is is_installed
