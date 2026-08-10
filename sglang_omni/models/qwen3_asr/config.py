# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-ASR."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.qwen3_asr"


class Qwen3ASRPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for Qwen3-ASR checkpoints."""

    architecture: ClassVar[str] = "Qwen3ASRForConditionalGeneration"

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
            factory=f"{_PKG}.stages.create_sglang_qwen3_asr_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 64,
                "max_new_tokens": 128,
                "request_build_max_workers": 8,
                "request_build_max_pending": 32,
                "prefill_coalesce_requests": 16,
                "prefill_coalesce_wait_ms": 24,
                "prefill_coalesce_when_idle": True,
                "prefill_coalesce_requires_pending_builds": True,
                "prefill_coalesce_after_builds_during_decode": True,
                "enable_pre_lm_encoder": True,
                "pre_lm_cache_max_entries": 4096,
                "pre_lm_cache_size_bytes": 2 * 1024**3,
                "pre_lm_max_batch_size": 8,
                "pre_lm_max_batch_wait_ms": 0,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = Qwen3ASRPipelineConfig
