# SPDX-License-Identifier: Apache-2.0
"""Config surface for the pipeline-level mps switch."""

from __future__ import annotations

from sglang_omni.config import PipelineConfig, StageConfig

FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


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
