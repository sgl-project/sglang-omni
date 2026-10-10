# SPDX-License-Identifier: Apache-2.0
"""Config surface for the pipeline-level mps switch."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.resolver import ConfigResolver

FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"
MPS_FLAG = ConfigSource(SourceKind.CLI_FLAG, "--mps")


def config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="thinker",
                process="pipeline",
                factory_path=FACTORY,
                gpu=0,
                terminal=True,
            )
        ],
        **kwargs,
    )


def test_mps_defaults_off():
    assert config().mps == "off"


def resolve_mps(mode: str) -> PipelineConfig:
    patch = ConfigPatch.create("mps", mode, MPS_FLAG)
    return ConfigResolver(config()).resolve(ConfigPatchSet([patch])).config


@pytest.mark.parametrize("mode", ["off", "on", "auto"])
def test_mps_accepts_valid_modes(mode):
    assert resolve_mps(mode).mps == mode


def test_mps_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        resolve_mps("always")
