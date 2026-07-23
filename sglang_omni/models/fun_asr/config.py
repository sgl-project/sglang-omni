# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.fun_asr"


class FunASRPipelineConfig(PipelineConfig):

    architecture: ClassVar[str] = "FunAsrNanoForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "FunASRNano",
        "FunASRForConditionalGeneration",
    )

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory=f"{_PKG}.stages.create_sglang_fun_asr_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 32,
                "max_new_tokens": 200,
                "enable_encoder_torch_compile": False,
                "enable_pre_lm_encoder": True,
                "pre_lm_cache_max_entries": 4096,
                "pre_lm_cache_size_bytes": 2 * 1024**3,
                "request_build_max_workers": 8,
                "request_build_max_pending": 16,
            },
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = FunASRPipelineConfig
