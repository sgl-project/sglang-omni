"""Eight-GPU Duo duplex serving configuration."""

from pathlib import Path
from typing import ClassVar

from pydantic import Field

from sglang_omni.config import EngineArgs, PipelineConfig, StageConfig
from sglang_omni.models.qwen3_asr.config import Qwen3ASRFactoryArgs, Qwen3ASRStageConfig


class DuoPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Duo"
    is_realtime_only: ClassVar[bool] = True
    realtime_deployment_factory: ClassVar[str] = (
        "sglang_omni.models.duo.pipeline.create_deployment"
    )
    stage_config_types: ClassVar = {"asr": Qwen3ASRStageConfig}
    name: str = "duo"
    entry_stage: str = "asr"
    realization_kv_reset_interval_blocks: int | None = Field(default=None, ge=1)
    realization_modelopt_checkpoint: str | None = None
    stages: list[StageConfig] = [
        Qwen3ASRStageConfig(
            name="asr",
            process="asr",
            factory_path="sglang_omni.models.qwen3_asr.stages.create_sglang_qwen3_asr_executor",
            factory=Qwen3ASRFactoryArgs(
                max_new_tokens=128,
                pre_lm_cache_max_entries=1,
                prefill_coalesce_wait_ms=0.01,
            ),
            engine=EngineArgs(
                max_running_requests=1,
                enable_torch_compile=False,
                mem_fraction_static=0.1,
            ),
            gpu=3,
            terminal=True,
        )
    ]

    def stage_factory_kwargs(self, stage_name: str):
        return {
            "model_path": str(
                Path(self.model_path) / "Duo-1.0-causal-base/speech/recognizer"
            )
        }


EntryClass = DuoPipelineConfig
