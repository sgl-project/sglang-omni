# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Fun-CosyVoice3."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.fun_cosyvoice3"


class FunCosyVoice3PipelineConfig(PipelineConfig):
    """3-stage Fun-CosyVoice3 pipeline: preprocessing -> tts_engine -> vocoder."""

    architecture: ClassVar[str] = "FunCosyVoice3SGLangModel"
    # note (PoTaTo): This checkpoint has no built-in speaker presets, so public requests need
    # one reference clip for speaker conditioning.
    required_speech_reference_count: ClassVar[int | None] = 1
    speech_reference_text_excludes_instructions: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({("preprocessing", "tts_engine")})

    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=8),
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                onnx_intra_op_threads=16,
            ),
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(
                dtype="bfloat16",
                flow_batch_bucket_frames=50,
                flow_batch_admission_frames=8000,
                max_batch_size=16,
                max_batch_wait_ms=30,
                # note (guozhihao-224): mutually exclusive DiT accelerators; both default off.
                enable_dit_torch_compile=False,
                enable_flow_estimator_trt=False,
            ),
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = FunCosyVoice3PipelineConfig
