# SPDX-License-Identifier: Apache-2.0
"""Single-stage VoXtream2 pipeline with explicit local assets."""

from typing import ClassVar, Literal

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.config.schema import CustomVoiceConfig


class VoXtream2PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "VoXtream2"
    requires_model_capabilities: ClassVar[bool] = True
    model_path: str
    reference_audio: str
    mimi_checkpoint: str
    speaker_checkpoint: str
    generator_config_path: str
    speaking_rate_config_path: str
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    use_cuda_graph: bool = True
    max_pending: int = Field(default=32, ge=1)
    output_buffer_chunks: int = Field(default=64, ge=1)
    model_python: str | None = None
    model_source_paths: list[str] = Field(default_factory=list)
    worker_startup_timeout: float = Field(default=180.0, gt=0, allow_inf_nan=False)
    worker_response_timeout: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    worker_cancel_timeout: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    entry_stage: str = "tts_engine"
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            StageConfig(
                name="tts_engine",
                process="pipeline",
                gpu=0,
                terminal=True,
                factory_path="sglang_omni.models.voxtream2.stages.create_executor",
            )
        ]
    )

    def stage_factory_kwargs(
        self, stage_name: str
    ) -> dict[str, str | float | int | bool | list[str] | None]:
        if stage_name != "tts_engine":
            return {}
        else:
            pass
        return {
            key: getattr(self, key)
            for key in (
                "reference_audio",
                "mimi_checkpoint",
                "speaker_checkpoint",
                "generator_config_path",
                "speaking_rate_config_path",
                "dtype",
                "use_cuda_graph",
                "max_pending",
                "output_buffer_chunks",
                "model_python",
                "model_source_paths",
                "worker_startup_timeout",
                "worker_response_timeout",
                "worker_cancel_timeout",
            )
        }

    def resolve_custom_voice_config(self) -> CustomVoiceConfig:
        return CustomVoiceConfig(speakers=("reference",), task_type="CustomVoice")


EntryClass = VoXtream2PipelineConfig
