# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 pipeline configuration.

Four-stage pipeline: preprocessing -> speaker_encode -> tts_engine -> vocoder.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.zonos2"


def _stages(*, auxiliary_gpu: int, auxiliary_process: str) -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            gpu=0,
            next="speaker_encode",
        ),
        StageConfig(
            name="speaker_encode",
            process=auxiliary_process,
            factory_path=f"{_PKG}.stages.create_speaker_encode_executor",
            gpu=auxiliary_gpu,
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_omni_tts_engine_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process=auxiliary_process,
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            gpu=auxiliary_gpu,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class Zonos2PipelineConfig(PipelineConfig):
    """Single-GPU colocated default."""

    architecture: ClassVar[str] = "Zonos2ForCausalLM"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "Zonos2",
        "Zonos2Model",
        "ZONOS2",
    )
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(auxiliary_gpu=0, auxiliary_process="pipeline")
    )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "tts_engine":
            return {
                "fp8": True,
                "frame_graph": True,
                "compile_sampler": True,
                "async_decode": True,
                "stream_emit_chunk_frames": 32,
            }
        if stage_name == "vocoder":
            # note (Yue Yin): keep dac_batch OFF -- verified numerically unsafe.
            # Batched DAC right-pads shorter items and the padding contaminates
            # them GLOBALLY (only the longest, unpadded item matches single
            # decode; shorter items diverge ~0.2-0.3 abs / ~0.3 rel = audible).
            # The DAC has no variable-length batching, so it is not croppable.
            # Do not enable without fixing the DAC itself.
            return {"dac_batch": False}
        return {}


class Zonos2MultiGPUPipelineConfig(Zonos2PipelineConfig):
    """Offload codec + speaker encoder to cuda:1, leaving the AR engine alone on cuda:0."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(auxiliary_gpu=1, auxiliary_process="auxiliary")
    )


EntryClass = Zonos2PipelineConfig

Variants = {
    "default": Zonos2PipelineConfig,
    "multi_gpu": Zonos2MultiGPUPipelineConfig,
}
