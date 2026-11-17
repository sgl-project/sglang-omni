import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / ".github/scripts/omni_missing_dependencies.py"
SPEC = importlib.util.spec_from_file_location("omni_missing_dependencies", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_missing_requirements(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """[project]
dependencies = [
  "packaging>=1",
  "packaging>999",
  "definitely-not-installed",
  "also-not-installed; python_version < '1'",
]

[tool.uv]
override-dependencies = ["protobuf>=6,<7"]
"""
    )

    assert MODULE.missing_requirements(pyproject) == [
        "packaging>999",
        "definitely-not-installed",
    ]

    assert MODULE.override_requirements(pyproject) == ["protobuf>=6,<7"]


def test_git_pin_rejects_a_release_or_different_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-version release must not satisfy the pinned Git dependency."""
    metadata = tmp_path / "packaging.dist-info"
    metadata.mkdir()
    installed_version = MODULE.importlib.metadata.version("packaging")
    (metadata / "METADATA").write_text(
        f"Name: packaging\nVersion: {installed_version}\n"
    )
    installed = MODULE.importlib.metadata.Distribution.at(metadata)
    monkeypatch.setattr(
        MODULE.importlib.metadata, "distribution", lambda name: installed
    )
    requirement = (
        "packaging @ git+https://example.com/packaging.git@abc123#subdirectory=python"
    )

    assert MODULE.unsatisfied_requirements([requirement]) == [requirement]
    direct_url = {
        "url": "https://example.com/packaging.git",
        "vcs_info": {"vcs": "git", "commit_id": "old"},
        "subdirectory": "python",
    }
    (metadata / "direct_url.json").write_text(json.dumps(direct_url))
    assert MODULE.unsatisfied_requirements([requirement]) == [requirement]
    direct_url["vcs_info"]["commit_id"] = "abc123"
    (metadata / "direct_url.json").write_text(json.dumps(direct_url))
    assert MODULE.unsatisfied_requirements([requirement]) == []
