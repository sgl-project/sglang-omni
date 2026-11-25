# SPDX-License-Identifier: Apache-2.0
"""One GPU owns the D2 encoder, decoder caches and delayed playback feedback."""

import sys
from typing import ClassVar

from pydantic import Field, PositiveInt

from sglang_omni.config.schema import FactoryArgs, PipelineConfig, StageConfig


class D2FactoryArgs(FactoryArgs):
    runtime_python: str = Field(default_factory=lambda: sys.executable, min_length=1)
    kv_budget: PositiveInt = 2048
    seed: int = 1337


class D2StageConfig(StageConfig):
    factory: D2FactoryArgs = Field(default_factory=D2FactoryArgs)


def duplex_stages() -> list[StageConfig]:
    return [
        D2StageConfig(
            name="duplex",
            process="duplex",
            gpu=0,
            factory_path="sglang_omni.models.qwen3_omni_d2.stages.create_scheduler",
            terminal=True,
        )
    ]


class Qwen3OmniD2PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Qwen3OmniD2"
    is_realtime_only: ClassVar[bool] = True
    realtime_deployment_factory: ClassVar[str] = (
        "sglang_omni.models.qwen3_omni_d2.realtime.create_realtime_deployment"
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "duplex": D2StageConfig
    }
    stages: list[StageConfig] = Field(default_factory=duplex_stages)


EntryClass = Qwen3OmniD2PipelineConfig
