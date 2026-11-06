# SPDX-License-Identifier: Apache-2.0
"""The shipped example configs must load, compile and resolve as documented."""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
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
    config = ConfigManager.from_file(str(config_path)).config
    build_compiled_process_topology(config)
    for stage in config.stages:
        resolve_stage_factory_args(stage, config)


def test_disable_speculative_decoding_drops_only_the_draft_keys() -> None:
    manager = ConfigManager.from_file(
        str(REPO_ROOT / "examples" / "configs" / "qwen3_asr_1_7b_dflash.yaml")
    )
    drafted = resolve_stage_factory_args(
        manager.config.stage_named("asr"), manager.config
    )["server_args_overrides"]
    config = manager.merge_config(
        manager.parse_extra_args(["--asr.engine.disable_speculative_decoding", "true"])
    )
    overrides = resolve_stage_factory_args(config.stage_named("asr"), config)[
        "server_args_overrides"
    ]
    assert "speculative_algorithm" in drafted
    assert overrides == {
        key: value
        for key, value in drafted.items()
        if not key.startswith("speculative_")
    }
