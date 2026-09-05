# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Fun-CosyVoice3."""

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
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.fun_cosyvoice3"

# Note (Jiaxin Deng): a stage that shares a GPU with another process group must declare
# its budget, and these sum to the 0.92 the single-process topology already uses.
_ISOLATED_TTS_ENGINE_GPU_MEMORY_FRACTION = 0.80
_ISOLATED_VOCODER_GPU_MEMORY_FRACTION = 0.12


def _stages(*, isolate_vocoder: bool) -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(device=current_platform.device_type, dtype="bfloat16"),
            engine=EngineArgs(),
            gpu_memory_fraction=(
                _ISOLATED_TTS_ENGINE_GPU_MEMORY_FRACTION if isolate_vocoder else None
            ),
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="vocoder" if isolate_vocoder else "pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(
                dtype="bfloat16",
                flow_batch_bucket_frames=50,
                flow_batch_admission_frames=2000,
                # Opt-in; off by default (one-time startup compile cost).
                enable_dit_torch_compile=False,
            ),
            gpu_memory_fraction=(
                _ISOLATED_VOCODER_GPU_MEMORY_FRACTION if isolate_vocoder else None
            ),
            gpu=0,
            terminal=True,
        ),
    ]


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

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(isolate_vocoder=False)
    )

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({("preprocessing", "tts_engine")})


class FunCosyVoice3IsolatedVocoderPipelineConfig(FunCosyVoice3PipelineConfig):
    """Flow vocoder in its own process on the same GPU.

    Note (Jiaxin Deng): the flow vocoder holds most in-flight requests while the ONNX
    reference encoders are the busiest thread in the same interpreter, so the two loops
    jitter each other. Measured +32% QPS at cap 16 on one H200.
    """

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(isolate_vocoder=True)
    )


Variants = {
    "default": FunCosyVoice3PipelineConfig,
    "isolated_vocoder": FunCosyVoice3IsolatedVocoderPipelineConfig,
}

EntryClass = FunCosyVoice3PipelineConfig
