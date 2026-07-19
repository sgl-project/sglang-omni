# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Higgs TTS (V1)."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.higgs_tts"


class HiggsTtsPipelineConfig(PipelineConfig):
    """4-stage TTS pipeline: preprocessing → audio_encoder → tts_engine → vocoder.

    Mirrors the V0 layout: preprocessing tokenises text + delay-pattern-encodes
    the reference audio codes; audio_encoder runs the fused multi-codebook
    embedding once on the delayed ref codes (CPU- or GPU-side); tts_engine
    drives the AR loop on the sglang backbone with the precomputed embed
    pasted at ``-100`` placeholder positions; vocoder reverses the delay
    pattern and decodes to waveform via the higgs-audio-v2-tokenizer codec.
    """

    architecture: ClassVar[str] = "HiggsMultimodalQwen3ForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "tts_engine"}

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next="audio_encoder",
        ),
        StageConfig(
            name="audio_encoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_audio_encoder_executor",
            factory_args={"device": "cuda"},
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={
                "device": "cuda",
                "max_new_tokens": 2048,
                "enable_async_decode": True,
            },
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"device": "cuda"},
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        stages = {stage.name: stage for stage in self.stages}
        vocoder = stages["vocoder"]
        tts_engine = stages["tts_engine"]
        vocoder_overrides = self.runtime_overrides.get("vocoder", {})
        tts_engine_overrides = self.runtime_overrides.get("tts_engine", {})
        missing = object()
        for key in ("stream_stride", "stream_followup_stride"):
            value = vocoder_overrides.get(key, vocoder.factory_args.get(key, missing))
            if value is missing:
                if key in tts_engine.factory_args or key in tts_engine_overrides:
                    raise ValueError(
                        f"Higgs TTS {key!r} must be configured on the vocoder stage"
                    )
                continue
            if key in tts_engine_overrides and tts_engine_overrides[key] != value:
                raise ValueError(
                    f"Higgs TTS {key!r} runtime overrides must match between "
                    "the tts_engine and vocoder stages"
                )
            tts_engine.factory_args[key] = value

    def requires_uploaded_voice_for_named_voice(self) -> bool:
        return True

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = HiggsTtsPipelineConfig
