# SPDX-License-Identifier: Apache-2.0
"""Haru checkpoint discovery and CUDA full-duplex pipeline settings."""

from collections.abc import Mapping
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.config.schema import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

HARU_STAGE = "haru"
ARCHITECTURE = "HaruForConditionalGeneration"
SAMPLE_RATE_HZ = 24_000
INPUT_FRAME_SAMPLES = 3_200
NATIVE_UNIT_MILLISECONDS = 400


class HaruSampling(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    temperature: float = Field(default=0.7, ge=0, le=2)
    top_k: int = Field(default=25, ge=1)
    top_p: float = Field(default=1.0, gt=0, le=1)
    guidance_scale: float = Field(default=1.5, ge=1, le=2)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    speak_first: bool = False
    eop_bias: float = 0.0
    dry: bool = True


def stages() -> list[StageConfig]:
    return [
        EngineStageConfig(
            name=HARU_STAGE,
            process=HARU_STAGE,
            gpu=0,
            factory_path="sglang_omni.models.haru.engine.create_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            terminal=True,
        )
    ]


class HaruPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = ARCHITECTURE
    realtime_deployment_factory: ClassVar[str] = (
        "sglang_omni.models.haru.realtime.create_realtime_deployment"
    )
    is_realtime_only: ClassVar[bool] = True
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        HARU_STAGE: EngineStageConfig
    }

    stages: list[StageConfig] = Field(default_factory=stages)
    max_sessions: int = Field(default=1, ge=1)
    codec_backend: Literal["torch", "triton", "auto"] = "auto"
    instructions: str = ""
    sampling: HaruSampling = Field(default_factory=HaruSampling)

    def stage_factory_kwargs(
        self, stage_name: str
    ) -> Mapping[str, str | int | float | bool | None]:
        assert stage_name == HARU_STAGE
        return {
            "max_sessions": self.max_sessions,
            "codec_backend": self.codec_backend,
        }


EntryClass = HaruPipelineConfig
Variants = {"session": HaruPipelineConfig}
