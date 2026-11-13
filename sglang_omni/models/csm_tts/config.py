# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for CSM TTS."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.utils.cpu import bounded_intraop_threads

_PKG = "sglang_omni.models.csm_tts"
_DEFAULT_MAX_CONCURRENCY = 4
_PREPROCESS_MAX_WORKERS = 2


class CsmTtsPipelineConfig(PipelineConfig):
    """preprocessing → audio_encoder → tts_engine → vocoder."""

    architecture: ClassVar[str] = "CsmForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("CsmTTS",)
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="tts_frontend",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=_PREPROCESS_MAX_WORKERS),
            next="audio_encoder",
        ),
        StageConfig(
            name="audio_encoder",
            process="tts_frontend",
            factory_path=f"{_PKG}.stages.create_audio_encoder_executor",
            factory=FactoryArgs(max_batch_size=_DEFAULT_MAX_CONCURRENCY),
            gpu=0,
            gpu_memory_fraction=0.03,
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(
                max_new_tokens=125,
                enable_async_decode=False,
            ),
            gpu=0,
            gpu_memory_fraction=0.50,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(
                dtype="float32",
                max_batch_size=_DEFAULT_MAX_CONCURRENCY,
            ),
            gpu=0,
            gpu_memory_fraction=0.10,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        stages = {stage.name: stage for stage in self.stages}
        preprocessing = stages["preprocessing"]
        if "OMP_NUM_THREADS" not in self.env_defaults:
            preprocessing.env.setdefault(
                "OMP_NUM_THREADS",
                str(
                    bounded_intraop_threads(
                        worker_count=_PREPROCESS_MAX_WORKERS,
                        max_threads=8,
                    )
                ),
            )

    def requires_uploaded_voice_for_named_voice(self) -> bool:
        return False

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = CsmTtsPipelineConfig
