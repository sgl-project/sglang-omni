import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_source_pin_requires_the_actual_installed_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching package version cannot make an old wheel satisfy a VCS pin."""
    requirement = (
        "sglang @ git+https://github.com/sgl-project/sglang.git@abc123"
        "#subdirectory=python"
    )
    origin = None
    installed = SimpleNamespace(version="0.5.22", read_text=lambda name: origin)
    monkeypatch.setattr(
        MODULE.importlib.metadata, "distribution", lambda name: installed
    )
    assert MODULE.unsatisfied_requirements([requirement]) == [requirement]
    origin = json.dumps(
        {
            "url": "https://github.com/sgl-project/sglang.git",
            "subdirectory": "python",
            "vcs_info": {"commit_id": "old"},
        }
    )
    assert MODULE.unsatisfied_requirements([requirement]) == [requirement]
    origin = origin.replace('"old"', '"abc123"')
    assert MODULE.unsatisfied_requirements([requirement]) == []
