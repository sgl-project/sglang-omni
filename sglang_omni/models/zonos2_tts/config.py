# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for ZONOS2 TTS."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.zonos2_tts"


def _codec_gpu(codec_device: str) -> int:
    if codec_device.startswith("cuda:"):
        return int(codec_device.split(":", 1)[1])
    if codec_device == "cuda":
        return 0
    return 0


def _stages(
    *,
    codec_device: str = "cuda:0",
    speaker_device: str = "cuda:0",
) -> list[StageConfig]:
    codec_gpu = _codec_gpu(codec_device)
    speaker_gpu = _codec_gpu(speaker_device)
    speaker_process = "pipeline" if speaker_gpu == 0 else "speaker_encode"
    vocoder_process = "pipeline" if codec_gpu == 0 else "vocoder"
    if speaker_gpu == codec_gpu and speaker_gpu != 0:
        speaker_process = vocoder_process
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={
                "device": "cpu",
                "load_speaker_model": False,
            },
            next="speaker_encode",
        ),
        StageConfig(
            name="speaker_encode",
            process=speaker_process,
            factory=f"{_PKG}.stages.create_speaker_encode_executor",
            factory_args={
                "device": speaker_device,
                "max_concurrency": 4,
            },
            gpu=speaker_gpu,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={
                "gpu_id": 0,
                "dtype": "bfloat16",
                "enable_async_decode": True,
            },
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process=vocoder_process,
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={
                "device": codec_device,
                "gpu_id": None,
                "max_batch_size": 16,
                "max_batch_frames": 1024,
            },
            gpu=codec_gpu,
            terminal=True,
        ),
    ]


class Zonos2TTSPipelineConfig(PipelineConfig):
    """ZONOS2 TTS pipeline: preprocessing -> speaker -> AR engine -> DAC."""

    architecture: ClassVar[str] = "Zonos2ForCausalLM"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "Zonos2SGLangModel",
        "Zonos2TTS",
        "Zonos2ForConditionalGeneration",
    )

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def talker_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "tts_engine"}

    @classmethod
    def code2wav_stage(cls) -> str | None:
        return "vocoder"

    model_path: str
    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0", speaker_device="cuda:0")
    )


class Zonos2TTSColocatedPipelineConfig(Zonos2TTSPipelineConfig):
    """Single-GPU variant that colocates the DAC vocoder with the AR engine."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:0", speaker_device="cuda:0")
    )


class Zonos2TTSMultiGPUPipelineConfig(Zonos2TTSPipelineConfig):
    """Two-GPU variant: AR engine on cuda:0, speaker encoder and DAC on cuda:1."""

    stages: list[StageConfig] = Field(
        default_factory=lambda: _stages(codec_device="cuda:1", speaker_device="cuda:1")
    )


EntryClass = Zonos2TTSPipelineConfig

Variants = {
    "default": Zonos2TTSPipelineConfig,
    "colocated": Zonos2TTSColocatedPipelineConfig,
    "multi_gpu": Zonos2TTSMultiGPUPipelineConfig,
}
