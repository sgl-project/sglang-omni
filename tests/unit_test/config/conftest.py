# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the configuration-core unit tests."""

from __future__ import annotations

from typing import ClassVar

import pytest

from sglang_omni.config.schema import EngineStageConfig, PipelineConfig, StageConfig


class FakePipelineConfig(PipelineConfig):
    """A minimal two-stage pipeline that needs no model weights.

    ``thinker`` is declared an engine stage so the ``engine.*`` group exists
    on it and path compilation can be exercised against a per-stage type.
    The factory-kwargs hook seeds one author-owned constructor kwarg so the
    code-channel/config-channel overlay is part of the fixture too.
    """

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "thinker": EngineStageConfig,
    }

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, object]:
        if stage_name == "thinker":
            return {"lookahead": 4}
        return {}


def build_pipeline_config() -> PipelineConfig:
    return FakePipelineConfig(
        model_path="/models/fake-omni",
        stages=[
            StageConfig(
                name="preprocessing",
                factory_path="tests.fake:create_preprocessing",
                process="front",
                next="thinker",
                env={"OMP_NUM_THREADS": "4"},
            ),
            EngineStageConfig(
                name="thinker",
                factory_path="tests.fake:create_thinker",
                process="gen",
                terminal=True,
                factory={"max_concurrency": 4, "max_seq_len": 8192},
            ),
        ],
    )


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    return build_pipeline_config()
