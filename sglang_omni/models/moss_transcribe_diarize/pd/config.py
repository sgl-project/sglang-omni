# SPDX-License-Identifier: Apache-2.0
"""Two-GPU Prefill/Decode profile for MOSS-Transcribe-Diarize."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.models.moss_transcribe_diarize.pd import (
    DECODE_STAGE,
    PREFILL_STAGE,
)
from sglang_omni.utils.cpu import bounded_intraop_threads

_PREFILL_FACTORY_PATH = (
    "sglang_omni.models.moss_transcribe_diarize.pd.stages."
    "create_sglang_moss_transcribe_diarize_prefill_executor"
)
_DECODE_FACTORY_PATH = (
    "sglang_omni.models.moss_transcribe_diarize.pd.stages."
    "create_sglang_moss_transcribe_diarize_decode_executor"
)
_REQUEST_BUILD_MAX_WORKERS = 8
_ENCODER_MAX_BATCH_SIZE = 2
_ENCODER_CACHE_SIZE_BYTES = 4 * 1024**3
_MAX_PIPELINE_INTRAOP_THREADS = 8


class MossTDPDFactoryArgs(FactoryArgs):
    encoder_cache_size_bytes: int | None = Field(default=None, ge=0)
    encoder_max_batch_size: int | None = Field(default=None, ge=1)


class MossTDPDStageConfig(EngineStageConfig):
    factory: MossTDPDFactoryArgs = Field(default_factory=MossTDPDFactoryArgs)


def _engine_args() -> EngineArgs:
    return EngineArgs(
        max_running_requests=16,
        enable_torch_compile=True,
        torch_compile_max_bs=4,
        disable_radix_cache=True,
        page_size=1,
    )


class MossTranscribeDiarizePDPipelineConfig(PipelineConfig):
    """MOSS-TD Prefill on GPU 0 and Decode on GPU 1."""

    architecture: ClassVar[str] = "MossTranscribeDiarizeForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        PREFILL_STAGE: MossTDPDStageConfig,
        DECODE_STAGE: MossTDPDStageConfig,
    }

    model_path: str
    entry_stage: str = PREFILL_STAGE
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            MossTDPDStageConfig(
                name=PREFILL_STAGE,
                process=PREFILL_STAGE,
                factory_path=_PREFILL_FACTORY_PATH,
                factory=MossTDPDFactoryArgs(
                    encoder_cache_size_bytes=_ENCODER_CACHE_SIZE_BYTES,
                    encoder_max_batch_size=_ENCODER_MAX_BATCH_SIZE,
                    enable_async_decode=False,
                    request_build_max_workers=_REQUEST_BUILD_MAX_WORKERS,
                    request_build_max_pending=16,
                    prefill_coalesce_requests=4,
                    prefill_coalesce_wait_ms=12,
                    prefill_coalesce_when_idle=True,
                    prefill_coalesce_requires_pending_builds=True,
                    prefill_coalesce_after_builds_during_decode=True,
                ),
                engine=_engine_args(),
                gpu=0,
                next=DECODE_STAGE,
            ),
            MossTDPDStageConfig(
                name=DECODE_STAGE,
                process=DECODE_STAGE,
                factory_path=_DECODE_FACTORY_PATH,
                factory=MossTDPDFactoryArgs(
                    encoder_cache_size_bytes=0,
                    encoder_max_batch_size=_ENCODER_MAX_BATCH_SIZE,
                    request_build_max_workers=1,
                ),
                engine=_engine_args(),
                gpu=1,
                terminal=True,
            ),
        ]
    )

    def resolved_env_defaults(self) -> dict[str, str]:
        configured_workers = self.stage_named(
            PREFILL_STAGE
        ).factory.request_build_max_workers
        request_build_workers = max(
            int(
                configured_workers
                if configured_workers is not None
                else _REQUEST_BUILD_MAX_WORKERS
            ),
            1,
        )
        derived = {
            "OMP_NUM_THREADS": str(
                bounded_intraop_threads(
                    worker_count=request_build_workers,
                    max_threads=_MAX_PIPELINE_INTRAOP_THREADS,
                )
            )
        }
        return {**derived, **self.env_defaults}


__all__ = [
    "DECODE_STAGE",
    "PREFILL_STAGE",
    "MossTranscribeDiarizePDPipelineConfig",
]
