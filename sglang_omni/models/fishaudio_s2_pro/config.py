# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for FishAudio S2-Pro TTS."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)
from sglang_omni.config.schema import stage_process_name

_PKG = "sglang_omni.models.fishaudio_s2_pro"
PREPROCESSING_TOTAL_RESERVE_BYTES = "6GiB"


class S2ProPipelineConfig(PipelineConfig):
    """3-stage TTS pipeline: preprocessing → tts_engine → vocoder."""

    architecture: ClassVar[str] = "FishQwen3OmniForCausalLM"
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    # Note (Jiaxin Deng): the engine sizes its KV pool from mem_fraction_static and
    # free memory, so the reference codec's reserve comes out of that headroom.
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="preprocessing",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            gpu=0,
            total_reserve_bytes=PREPROCESSING_TOTAL_RESERVE_BYTES,
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(max_new_tokens=2048),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def supports_uploaded_voice_references(self) -> bool:
        return True

    def validate_processes(self) -> None:
        super().validate_processes()
        preprocessing = self.stage_named("preprocessing")
        if preprocessing.gpu is None:
            return
        else:
            pass
        process_name = stage_process_name(preprocessing)
        if any(
            stage.name != preprocessing.name
            and stage_process_name(stage) == process_name
            for stage in self.stages
        ):
            # Note (Jiaxin Deng): the allocator reserve cap and TF32 off apply to
            # the whole process, so co-tenant stages would inherit both.
            raise ValueError(
                "GPU preprocessing needs its own process; give "
                "stages.preprocessing a dedicated process or set its gpu to null"
            )
        else:
            pass


EntryClass = S2ProPipelineConfig
