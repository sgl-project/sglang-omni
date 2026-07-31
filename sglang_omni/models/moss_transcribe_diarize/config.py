# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for MOSS-Transcribe-Diarize."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.config.runtime import resolve_stage_static_factory_args
from sglang_omni.models.moss_transcribe_diarize import (  # noqa: F401
    hf_config as _hf_config,
)
from sglang_omni.utils.cpu import bounded_intraop_threads

_PKG = "sglang_omni.models.moss_transcribe_diarize"
_REQUEST_BUILD_MAX_WORKERS = 8
_ENCODER_MAX_BATCH_SIZE = 2
_MAX_PIPELINE_INTRAOP_THREADS = 8


class MossTranscribeDiarizePipelineConfig(PipelineConfig):
    """Single-stage batched ASR/diarization pipeline for MOSS-TD checkpoints."""

    architecture: ClassVar[str] = "MossTranscribeDiarizeForConditionalGeneration"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"asr": "asr"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "asr"}

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_sglang_moss_transcribe_diarize_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 16,
                "encoder_cache_size_bytes": 4 * 1024**3,
                "encoder_max_batch_size": _ENCODER_MAX_BATCH_SIZE,
                "request_build_max_workers": _REQUEST_BUILD_MAX_WORKERS,
                "request_build_max_pending": 16,
            },
            gpu=0,
            terminal=True,
        )
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if "OMP_NUM_THREADS" in self.env_defaults:
            return

        asr_stage = next(stage for stage in self.stages if stage.name == "asr")
        factory_args = resolve_stage_static_factory_args(asr_stage, self)
        request_build_workers = max(
            int(
                factory_args.get(
                    "request_build_max_workers", _REQUEST_BUILD_MAX_WORKERS
                )
            ),
            1,
        )
        # Request builders run torch/OpenMP-backed fbank extraction on CPU
        # threads inside the asr stage process. Without a bound, each of the
        # eight concurrent builders spawns a machine-sized OMP team; under a
        # cgroup CPU quota the oversubscription starves the scheduler thread
        # mid-prefill (50-300ms host stalls that gate streaming decode). The
        # env must be in place before the spawned stage process imports Torch.
        self.env_defaults["OMP_NUM_THREADS"] = str(
            bounded_intraop_threads(
                worker_count=request_build_workers,
                max_threads=_MAX_PIPELINE_INTRAOP_THREADS,
            ),
        )


EntryClass = MossTranscribeDiarizePipelineConfig
