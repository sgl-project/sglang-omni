# SPDX-License-Identifier: Apache-2.0
"""Single-stage native FireRedASR2-LLM transcription pipeline."""

from typing import ClassVar

from sglang_omni.config.schema import (
    AudioChunkingConfig,
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)


class FireRedASR2PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "FireRedASR2ForConditionalGeneration"
    allow_audio_chunking: ClassVar[bool] = True
    max_native_clip_s: ClassVar[float] = 40.0
    audio_chunking: AudioChunkingConfig = AudioChunkingConfig(max_audio_clip_s=30.0)
    model_path: str
    entry_stage: str = "asr"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": EngineStageConfig,
    }
    stages: list[StageConfig] = [
        EngineStageConfig(
            name="asr",
            process="asr",
            factory_path="sglang_omni.models.firered_asr2.stages.create_sglang_firered_asr2_executor",
            factory=FactoryArgs(
                dtype="bfloat16",
                max_new_tokens=256,
                mm_embedding_cache_size_bytes=256 * 1024 * 1024,
                enable_encoder_cuda_graph=True,
            ),
            engine=EngineArgs(max_running_requests=8),
            gpu=0,
            terminal=True,
        )
    ]

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, int]:
        stage = self.stage_named(stage_name)
        assert isinstance(stage, EngineStageConfig)
        assert stage.engine is not None
        assert stage.engine.max_running_requests is not None
        return {"max_running_requests": stage.engine.max_running_requests}


EntryClass = FireRedASR2PipelineConfig
