# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for dots.tts."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.dots_tts"


class DotsVocoderFactoryArgs(FactoryArgs):
    """dots.tts vocoder constructor knobs, typed like the shared ones."""

    stream_slots: int | None = Field(default=None, ge=1)


class DotsVocoderStageConfig(StageConfig):
    factory: DotsVocoderFactoryArgs = Field(default_factory=DotsVocoderFactoryArgs)


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

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "latent_engine": EngineStageConfig,
        "vocoder": DotsVocoderStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            next="reference_encode",
        ),
        StageConfig(
            name="reference_encode",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_reference_encode_executor",
            gpu=0,
            next="latent_engine",
        ),
        EngineStageConfig(
            name="latent_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_latent_engine_executor",
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        DotsVocoderStageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if any(stage.tp_size != 1 for stage in self.stages):
            raise ValueError("dots.tts currently supports tp_size=1 only")
        # note (guozhihao-224): preprocessing bakes the generation schedule the
        # latent engine executes, so num_steps and max_generate_length must
        # agree between the two stages when both are set. The latent_engine
        # group is the owner; a preprocessing value that conflicts with the
        # latent_engine value is refused. Unset preprocessing keys are derived
        # at launch from the latent_engine.
        preprocessing_extra = (
            self.stage_named("preprocessing").factory.model_extra or {}
        )
        latent_extra = self.stage_named("latent_engine").factory.model_extra or {}
        for derived_key, default in (("num_steps", 4), ("max_generate_length", 500)):
            pre_val = preprocessing_extra.get(derived_key)
            lat_val = latent_extra.get(derived_key, default)
            if pre_val is not None and int(pre_val) != int(lat_val):
                raise ValueError(
                    f"dots.tts preprocessing {derived_key!r} ({int(pre_val)}) "
                    f"disagrees with latent_engine {derived_key!r} ({int(lat_val)}); "
                    f"configure only latent_engine.factory.{derived_key} and let "
                    "preprocessing derive it, or set both to the same value"
                )
        # note (guozhihao-224): stream_slots must match backbone concurrency
        # so a max_running_requests override cannot outrun vocoder admission
        # after readiness. A pinned value that disagrees is refused here, on
        # every rebuild; an unset value stays unset and is derived at launch
        # (writing the derivation into the config would make it look pinned
        # on the next merge).
        stream_slots = self.stage_named("vocoder").factory.stream_slots
        if stream_slots is not None:
            derived = self._latent_max_running_requests(
                self.stage_named("latent_engine")
            )
            if int(stream_slots) != derived:
                raise ValueError(
                    "dots.tts vocoder stream_slots "
                    f"({int(stream_slots)}) must equal latent_engine "
                    f"max_running_requests ({derived}); "
                    "omit stream_slots to derive it from the latent engine"
                )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "preprocessing":
            latent_extra = self.stage_named("latent_engine").factory.model_extra or {}
            return {
                "num_steps": int(latent_extra.get("num_steps", 4)),
                "max_generate_length": int(
                    latent_extra.get("max_generate_length", 500)
                ),
            }
        if stage_name == "vocoder":
            return {
                "stream_slots": self._latent_max_running_requests(
                    self.stage_named("latent_engine")
                )
            }
        return {}

    @staticmethod
    def _latent_max_running_requests(latent_engine: StageConfig) -> int:
        engine = latent_engine.engine
        value = engine.max_running_requests if engine is not None else None
        if value is None:
            # Match DotsTTSEngineBuilder default when unset.
            return 16
        return int(value)

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = DotsTTSPipelineConfig
