# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-Transcribe-Diarize."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.models.moss_transcribe_diarize import (  # noqa: F401
    hf_config as _hf_config,
)
from sglang_omni.utils.cpu import bounded_intraop_threads

_PKG = "sglang_omni.models.moss_transcribe_diarize"
_REQUEST_BUILD_MAX_WORKERS = 8
_ENCODER_MAX_BATCH_SIZE = 2
_ENCODER_CACHE_SIZE_BYTES = 4 * 1024**3
_MAX_PIPELINE_INTRAOP_THREADS = 8


class MossTDFactoryArgs(FactoryArgs):
    """MOSS-TD's own constructor knobs, typed like the shared ones."""

    encoder_cache_size_bytes: int | None = Field(default=None, ge=0)
    encoder_max_batch_size: int | None = Field(default=None, ge=1)


class MossTDStageConfig(EngineStageConfig):
    factory: MossTDFactoryArgs = Field(default_factory=MossTDFactoryArgs)


class MossTranscribeDiarizePipelineConfig(PipelineConfig):
    """Single-stage batched ASR/diarization pipeline for MOSS-TD checkpoints."""

    architecture: ClassVar[str] = "MossTranscribeDiarizeForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": MossTDStageConfig,
    }

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        MossTDStageConfig(
            name="asr",
            process="asr",
            factory_path=f"{_PKG}.stages.create_sglang_moss_transcribe_diarize_executor",
            factory=MossTDFactoryArgs(
                device="cuda:0",
                encoder_cache_size_bytes=4 * 1024**3,
                encoder_max_batch_size=_ENCODER_MAX_BATCH_SIZE,
                request_build_max_workers=_REQUEST_BUILD_MAX_WORKERS,
                request_build_max_pending=16,
                prefill_coalesce_requests=4,
                prefill_coalesce_wait_ms=12,
                prefill_coalesce_when_idle=True,
                prefill_coalesce_requires_pending_builds=True,
                prefill_coalesce_after_builds_during_decode=True,
            ),
            engine=EngineArgs(
                max_running_requests=16,
                enable_torch_compile=True,
                torch_compile_max_bs=4,
            ),
            gpu=0,
            terminal=True,
        )
    ]

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "asr":
            return {"encoder_cache_size_bytes": _ENCODER_CACHE_SIZE_BYTES}
        return {}

    def resolved_env_defaults(self) -> dict[str, str]:
        asr_stage = next(stage for stage in self.stages if stage.name == "asr")
        configured_workers = asr_stage.factory.request_build_max_workers
        request_build_workers = max(
            int(
                configured_workers
                if configured_workers is not None
                else _REQUEST_BUILD_MAX_WORKERS
            ),
            1,
        )
        # Request builders run torch/OpenMP-backed fbank extraction on CPU
        # threads inside the asr stage process. Without a bound, each of the
        # eight concurrent builders spawns a machine-sized OMP team; under a
        # cgroup CPU quota the oversubscription starves the scheduler thread
        # mid-prefill (50-300ms host stalls that gate streaming decode). The
        # env must be in place before the spawned stage process imports Torch.
        # Derived at launch so a rebuilt config re-derives from the resolved
        # worker count; a written env_defaults entry wins.
        derived = {
            "OMP_NUM_THREADS": str(
                bounded_intraop_threads(
                    worker_count=request_build_workers,
                    max_threads=_MAX_PIPELINE_INTRAOP_THREADS,
                )
            )
        }
        return {**derived, **self.env_defaults}


EntryClass = MossTranscribeDiarizePipelineConfig
