# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for dots.tts."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.dots_tts"


class DotsTTSPipelineConfig(PipelineConfig):
    """preprocess -> reference encode -> SGLang latent AR -> AudioVAE."""

    architecture: ClassVar[str] = "DotsTTSForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    required_speech_reference_count: ClassVar[int | None] = 1
    speech_reference_text_required: ClassVar[bool] = True
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "dots_tts",
        "DotsTtsModel",
    )
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset({"auto_detect"})

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next="reference_encode",
        ),
        StageConfig(
            name="reference_encode",
            process="pipeline",
            factory=f"{_PKG}.stages.create_reference_encode_executor",
            factory_args={"device": "cuda"},
            gpu=0,
            next="latent_engine",
        ),
        StageConfig(
            name="latent_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_latent_engine_executor",
            factory_args={"device": "cuda", "precision": "bfloat16"},
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"device": "cuda", "precision": "bfloat16"},
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if any(stage.tp_size != 1 for stage in self.stages):
            raise ValueError("dots.tts currently supports tp_size=1 only")
        stages = {stage.name: stage for stage in self.stages}
        preprocessing = stages["preprocessing"]
        latent_engine = stages["latent_engine"]
        preprocessing_overrides = self.runtime_overrides.get("preprocessing", {})
        latent_overrides = self.runtime_overrides.get("latent_engine", {})
        for engine_key, preprocessing_key, default in (
            ("num_steps", "default_num_steps", 4),
            ("max_generate_length", "max_generate_length", 500),
        ):
            engine_value = latent_overrides.get(
                engine_key, latent_engine.factory_args.get(engine_key, default)
            )
            if preprocessing_key in preprocessing_overrides:
                raise ValueError(
                    f"dots.tts {preprocessing_key!r} is derived from {engine_key!r}; "
                    "configure it on the latent_engine stage"
                )
            preprocessing.factory_args[preprocessing_key] = engine_value

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "latent_engine"}

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = DotsTTSPipelineConfig
