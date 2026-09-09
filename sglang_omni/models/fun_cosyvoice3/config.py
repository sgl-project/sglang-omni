# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Fun-CosyVoice3."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.platforms import current_platform

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
                # Keep in sync with vocoder token_hop_len (AR flush cadence).
                token_hop_len=25,
            ),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
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


EntryClass = FunCosyVoice3PipelineConfig
