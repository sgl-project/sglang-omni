# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import EngineStageConfig, PipelineConfig, StageConfig

_PKG = "sglang_omni.models.fun_asr"


class FunASRPipelineConfig(PipelineConfig):

    architecture: ClassVar[str] = "FunAsrNanoForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "FunASRNano",
        "FunASRForConditionalGeneration",
    )

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": EngineStageConfig,
    }

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        EngineStageConfig(
            name="asr",
            process="asr",
            factory_path=f"{_PKG}.stages.create_sglang_fun_asr_executor",
            gpu=0,
            terminal=True,
        )
    ]

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "asr":
            return {"enable_encoder_cuda_graph": True}
        return {}


EntryClass = FunASRPipelineConfig
