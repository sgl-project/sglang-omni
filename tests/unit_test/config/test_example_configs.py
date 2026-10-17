# SPDX-License-Identifier: Apache-2.0
"""The shipped example configs must load, compile and resolve as documented."""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni_router.python.launcher.config import load_launcher_config
from tests.unit_test.pipeline.helpers import build_compiled_process_topology

REPO_ROOT = Path(__file__).resolve().parents[3]

EXAMPLE_CONFIG_PATHS = sorted((REPO_ROOT / "examples" / "configs").glob("*.yaml"))


def test_example_configs_are_found() -> None:
    assert len(EXAMPLE_CONFIG_PATHS) > 30


@pytest.mark.parametrize(
    "config_path", EXAMPLE_CONFIG_PATHS, ids=lambda config_path: config_path.name
)
def test_example_config_loads_compiles_and_resolves_every_factory(
    config_path: Path,
) -> None:
    if config_path.name.endswith("_router.yaml"):
        load_launcher_config(config_path)
    else:
        config = ConfigManager.from_file(str(config_path)).config
        build_compiled_process_topology(config)
        for stage in config.stages:
            resolve_stage_factory_args(stage, config)
