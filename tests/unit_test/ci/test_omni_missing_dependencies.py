# SPDX-License-Identifier: Apache-2.0
"""Dependency checks must cover the optional models selected by CI."""

from importlib import metadata
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3] / ".github/scripts/omni_missing_dependencies.py"
)
SPEC = spec_from_file_location("omni_missing_dependencies", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dependencies = module_from_spec(SPEC)
SPEC.loader.exec_module(dependencies)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\ndependencies = ["torch==2.13.0"]\n'
        "[project.optional-dependencies]\n"
        'minicpm-o = ["einops>=0.8.1", "onnx>=1.18.0"]\n'
    )
    return path


@pytest.mark.parametrize("onnx_version", [None, "1.17.0", "1.18.0"])
def test_minicpm_extra_checks_missing_and_outdated_dependencies(
    project: Path, monkeypatch: pytest.MonkeyPatch, onnx_version: str | None
) -> None:
    versions = {"torch": "2.13.0", "einops": "0.8.1", "onnx": onnx_version}

    def version(name: str) -> str:
        installed = versions[name]
        if installed is None:
            raise metadata.PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(dependencies.importlib.metadata, "version", version)
    assert dependencies.missing_requirements(project) == []
    assert dependencies.missing_requirements(project, ("minicpm-o",)) == (
        [] if onnx_version == "1.18.0" else ["onnx>=1.18.0"]
    )


def test_unknown_extra_fails_instead_of_silently_omitting_dependencies(
    project: Path,
) -> None:
    with pytest.raises(KeyError, match="minicpm-typo"):
        dependencies.missing_requirements(project, ("minicpm-typo",))
