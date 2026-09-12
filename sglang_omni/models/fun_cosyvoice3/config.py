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

_PKG = "sglang_omni.models.fun_cosyvoice3"

_DIT_ACCELERATOR_CONFLICT = (
    "enable_flow_estimator_trt and enable_dit_torch_compile both "
    "target flow.decoder.estimator; enable only one"
)


def reject_conflicting_dit_accelerators(
    *,
    enable_dit_torch_compile: bool,
    enable_flow_estimator_trt: bool,
) -> None:
    if enable_flow_estimator_trt and enable_dit_torch_compile:
        raise ValueError(_DIT_ACCELERATOR_CONFLICT)


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
            factory=FactoryArgs(max_concurrency=8),
            next="tts_engine",
        ),
        FunCosyVoice3EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FunCosyVoice3EngineFactoryArgs(
                dtype="bfloat16",
                onnx_intra_op_threads=16,
                # Keep in sync with vocoder token_hop_len (AR flush cadence).
                token_hop_len=25,
            ),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        FunCosyVoice3VocoderStageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FunCosyVoice3VocoderFactoryArgs(
                flow_batch_bucket_frames=50,
                flow_batch_admission_frames=8000,
                max_batch_wait_ms=30,
                # note (guozhihao-224, chenyang):
                # torch.compile is opt-in via enable_dit_torch_compile.
                enable_flow_estimator_trt=False,
                token_hop_len=25,
                token_max_hop_len=100,
                disable_hop_growth=False,
            ),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        # TODO (chenyang): Indeed, TRT and Torch compile conflicts are pretty
        # common in this repo, so we should make this into config level, not in each model.
        super().model_post_init(__context)
        vocoder = next(stage for stage in self.stages if stage.name == "vocoder")
        extras = vocoder.factory.model_extra
        reject_conflicting_dit_accelerators(
            enable_dit_torch_compile=bool(extras.get("enable_dit_torch_compile")),
            enable_flow_estimator_trt=bool(extras.get("enable_flow_estimator_trt")),
        )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name != "vocoder":
            return {}
        vocoder_factory = self.stage_named("vocoder").factory
        if vocoder_factory.mlx_model_path is not None:
            # Note (yexiaodong): A separate vocoder artifact must keep its own
            # revision; both explicit fields therefore stay in typed config.
            return {}
        # Note (yexiaodong): The converted artifact contains the speech-token
        # LLM, Flow, and HiFT weights, so reuse it unless the vocoder overrides it.
        engine_factory = self.stage_named("tts_engine").factory
        kwargs: dict[str, Any] = {}
        if engine_factory.mlx_model_path is not None:
            kwargs["mlx_model_path"] = engine_factory.mlx_model_path
        if engine_factory.mlx_model_revision is not None:
            kwargs["mlx_model_revision"] = engine_factory.mlx_model_revision
        return kwargs


EntryClass = FunCosyVoice3PipelineConfig
