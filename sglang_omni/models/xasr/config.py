# SPDX-License-Identifier: Apache-2.0
"""X-ASR Zipformer transcription pipeline."""

from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig


class XASRPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "XASRZipformerTransducer"
    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        StageConfig(
            name="asr",
            process="asr",
            factory_path="sglang_omni.models.xasr.stages.create_xasr_executor",
            factory=FactoryArgs(max_batch_size=4, max_batch_wait_ms=20),
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = XASRPipelineConfig
