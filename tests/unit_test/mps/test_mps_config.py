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

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"
_MPS_FLAG = ConfigSource(SourceKind.CLI_FLAG, "--mps")


def _config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="thinker",
                process="pipeline",
                factory_path=_FACTORY,
                gpu=0,
                terminal=True,
            )
        ],
        **kwargs,
    )


def test_mps_defaults_off():
    assert _config().mps == "off"


def _resolve_mps(mode: str) -> PipelineConfig:
    patch = ConfigPatch.create("mps", mode, _MPS_FLAG)
    return ConfigResolver(_config()).resolve(ConfigPatchSet([patch])).config


@pytest.mark.parametrize("mode", ["off", "on", "auto"])
def test_mps_accepts_valid_modes(mode):
    assert _resolve_mps(mode).mps == mode


def test_mps_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        _resolve_mps("always")
