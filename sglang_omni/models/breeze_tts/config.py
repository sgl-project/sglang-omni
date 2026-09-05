# SPDX-License-Identifier: Apache-2.0
"""Single-GPU, eager Breeze-TTS-2 pipeline."""

from typing import ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.breeze_tts"


class BreezeTTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "BreezeForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    speech_reference_text_required: ClassVar[bool] = True
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            gpu=0,
            gpu_memory_fraction=0.25,
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_tts_engine_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            gpu_memory_fraction=0.60,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            gpu=0,
            gpu_memory_fraction=0.15,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    @classmethod
    def generation_admission_defaults(cls) -> dict[str, int]:
        # Two SGLang rows form one logical CFG request. HTTP requests can queue.
        return {"max_running_requests": 2, "max_queued_requests": 64}


EntryClass = BreezeTTSPipelineConfig
