# SPDX-License-Identifier: Apache-2.0
"""One GPU owns the D2 encoder, decoder caches and delayed playback feedback."""

from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, PipelineConfig, StageConfig

INPUT_SAMPLE_RATE_HZ = 16_000
OUTPUT_SAMPLE_RATE_HZ = 24_000
NATIVE_FRAME_MS = 80
MAX_SESSIONS = 1

Qwen3OmniD2Language = Literal["zh", "auto"]


class Qwen3OmniD2FactoryArgs(FactoryArgs):
    kv_budget: int = Field(default=2048, ge=128)
    seed: int = 1337


class Qwen3OmniD2StageConfig(StageConfig):
    factory: Qwen3OmniD2FactoryArgs = Field(default_factory=Qwen3OmniD2FactoryArgs)


def duplex_stages() -> list[StageConfig]:
    return [
        Qwen3OmniD2StageConfig(
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
        "duplex": Qwen3OmniD2StageConfig
    }
    stages: list[StageConfig] = Field(default_factory=duplex_stages)
    language: Qwen3OmniD2Language = Field(
        default="zh",
        description="Spoken response language. Use auto for checkpoint defaults.",
    )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, str]:
        if stage_name == "duplex":
            return {"language": self.language}
        else:
            return {}


EntryClass = Qwen3OmniD2PipelineConfig
