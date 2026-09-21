# SPDX-License-Identifier: Apache-2.0
"""XPU model imports must be represented in the XPU package manifest."""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_auk_runtime_dependencies_are_in_xpu_manifest() -> None:
    manifest = Path(__file__).parents[3] / "pyproject_xpu.toml"
    project = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]
    dependencies = {Requirement(value).name for value in project["dependencies"]}

    assert {"omegaconf", "x-transformers"} <= dependencies
