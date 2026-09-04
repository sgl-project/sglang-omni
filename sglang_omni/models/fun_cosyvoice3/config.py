# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Fun-CosyVoice3."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.fun_cosyvoice3"


class FunCosyVoice3EngineFactoryArgs(FactoryArgs):
    """Engine-only knobs, including the optional native MLX checkpoint."""

    mlx_model_path: str | None = Field(default=None)
    mlx_model_revision: str | None = Field(default=None)


class FunCosyVoice3EngineStageConfig(EngineStageConfig):
    factory: FunCosyVoice3EngineFactoryArgs = Field(
        default_factory=FunCosyVoice3EngineFactoryArgs
    )


class FunCosyVoice3VocoderFactoryArgs(FactoryArgs):
    """Vocoder knobs, including the converted native MLX artifact."""

    mlx_model_path: str | None = Field(default=None)
    mlx_model_revision: str | None = Field(default=None)


class FunCosyVoice3VocoderStageConfig(StageConfig):
    factory: FunCosyVoice3VocoderFactoryArgs = Field(
        default_factory=FunCosyVoice3VocoderFactoryArgs
    )


class FunCosyVoice3PipelineConfig(PipelineConfig):
    """3-stage Fun-CosyVoice3 pipeline: preprocessing -> tts_engine -> vocoder."""

    architecture: ClassVar[str] = "FunCosyVoice3SGLangModel"
    # note (PoTaTo): This checkpoint has no built-in speaker presets, so public requests need
    # one reference clip for speaker conditioning.
    required_speech_reference_count: ClassVar[int | None] = 1
    speech_reference_text_excludes_instructions: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": FunCosyVoice3EngineStageConfig,
        "vocoder": FunCosyVoice3VocoderStageConfig,
    }

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({("preprocessing", "tts_engine")})

    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            next="tts_engine",
        ),
        FunCosyVoice3EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FunCosyVoice3EngineFactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
            ),
            gpu=0,
            next="vocoder",
        ),
        FunCosyVoice3VocoderStageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FunCosyVoice3VocoderFactoryArgs(
                flow_batch_bucket_frames=50,
                flow_batch_admission_frames=2000,
                # Opt-in; off by default (one-time startup compile cost).
                enable_dit_torch_compile=False,
            ),
            gpu=0,
            terminal=True,
        ),
    ]

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name != "vocoder":
            return {}
        vocoder_factory = self.stage_named("vocoder").factory
        if vocoder_factory.mlx_model_path is not None:
            # A distinct vocoder repository must not inherit the engine
            # repository's revision. Both explicit vocoder fields travel
            # through typed config instead.
            return {}
        # One converted artifact contains the speech-token LLM, Flow, and
        # HiFT weights. Reuse the engine's artifact by default so the common
        # MLX launch only needs one model override; an explicit vocoder factory
        # value still wins through the normal typed-config precedence.
        engine_factory = self.stage_named("tts_engine").factory
        kwargs: dict[str, Any] = {}
        if engine_factory.mlx_model_path is not None:
            kwargs["mlx_model_path"] = engine_factory.mlx_model_path
        if engine_factory.mlx_model_revision is not None:
            kwargs["mlx_model_revision"] = engine_factory.mlx_model_revision
        return kwargs


EntryClass = FunCosyVoice3PipelineConfig
