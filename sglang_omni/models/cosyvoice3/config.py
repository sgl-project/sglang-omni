# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for CosyVoice3 TTS.

3-stage TTS pipeline: preprocessing -> tts_engine (AR speech-token LM, Qwen2-0.5B
backbone) -> vocoder (CausalMaskedDiffWithDiT flow -> CausalHiFTGenerator).
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.cosyvoice3"


class CosyVoice3PipelineConfig(PipelineConfig):
    """CosyVoice3 zero-shot TTS pipeline."""

    # Registry key. CosyVoice3 checkpoints are served via YAML (config_cls), but we
    # still register the arch so the AR engine can resolve our SGLang model class via
    # model_arch_override="CosyVoice3ForCausalLM".
    architecture: ClassVar[str] = "CosyVoice3ForCausalLM"
    requires_model_capabilities: ClassVar[bool] = True

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "tts_engine"}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            # The frontend runs mel/campplus tensors and the ORT speech tokenizer on
            # a GPU; declare the placement (colocated with the engine, whose
            # embeddings the prompt build reads) so relocation moves it too.
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            # flow + hift run in fp32 (flow asserts batch==1; hift f0 predictor needs
            # fp64) — see create_vocoder_executor; advertise the real runtime dtype.
            factory_args={"dtype": "float32"},
            gpu=0,
            terminal=True,
        ),
    ]

    def supports_uploaded_voice_references(self) -> bool:
        # Zero-shot: a reference (audio + text) clones the target voice.
        return True


EntryClass = CosyVoice3PipelineConfig
