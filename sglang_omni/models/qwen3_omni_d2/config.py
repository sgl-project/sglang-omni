# SPDX-License-Identifier: Apache-2.0
"""One GPU owns the D2 encoder, decoder caches and delayed playback feedback."""

import sys
from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, PipelineConfig, StageConfig

INPUT_SAMPLE_RATE_HZ = 16_000
OUTPUT_SAMPLE_RATE_HZ = 24_000
NATIVE_FRAME_MS = 80
MAX_SESSIONS = 1

D2Language = Literal["zh", "auto"]


class D2FactoryArgs(FactoryArgs):
    runtime_python: str = Field(default_factory=lambda: sys.executable, min_length=1)
    kv_budget: int = Field(default=2048, ge=128)
    seed: int = 1337


class D2StageConfig(StageConfig):
    factory: D2FactoryArgs = Field(default_factory=D2FactoryArgs)


def duplex_stages() -> list[StageConfig]:
    return [
        D2StageConfig(
            name="duplex",
            process="duplex",
            gpu=0,
            allow_child_processes=True,
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
    language: D2Language = Field(
        default="zh",
        description="Spoken response language. Use auto for checkpoint defaults.",
    )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, str]:
        if stage_name == "duplex":
            return {"language": self.language}
        else:
            return {}


EntryClass = Qwen3OmniD2PipelineConfig
