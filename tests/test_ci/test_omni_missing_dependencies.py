import importlib.util
from pathlib import Path

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
