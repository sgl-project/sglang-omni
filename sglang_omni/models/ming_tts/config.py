# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Ming-Omni-TTS 16B."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.ming_tts"

PREPROCESSING_STAGE = "preprocessing"
REFERENCE_ENCODE_STAGE = "reference_encode"
TTS_ENGINE_STAGE = "tts_engine"
AUDIO_DECODE_STAGE = "audio_decode"


class MingTTSPipelineConfig(PipelineConfig):
    """Ming-Omni-TTS pipeline.

    preprocessing -> reference_encode -> tts_engine -> audio_decode.
    The reference stage is kept as a fixed cheap/no-op boundary for text-only
    requests so reference-conditioned requests use the same serving graph.
    """

    architecture: ClassVar[str] = "BailingMMNativeForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"talker": TTS_ENGINE_STAGE}

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": TTS_ENGINE_STAGE}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": TTS_ENGINE_STAGE}

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next=REFERENCE_ENCODE_STAGE,
        ),
        StageConfig(
            name=REFERENCE_ENCODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_reference_encode_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            next=TTS_ENGINE_STAGE,
        ),
        StageConfig(
            name=TTS_ENGINE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            next=AUDIO_DECODE_STAGE,
        ),
        StageConfig(
            name=AUDIO_DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_audio_decode_executor",
            factory_args={
                "dtype": "bfloat16",
                "decode_mode": "chunked",
            },
            gpu=0,
            terminal=True,
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        for stage in self.stages:
            if stage.name != TTS_ENGINE_STAGE:
                if stage.tp_size != 1:
                    raise ValueError(
                        "Ming-Omni-TTS supports tensor parallelism only on "
                        f"{TTS_ENGINE_STAGE!r}; stage {stage.name!r} has "
                        f"tp_size={stage.tp_size}."
                    )
                continue

            if stage.tp_size <= 0:
                raise ValueError(
                    "Ming-Omni-TTS tts_engine tp_size must be positive; "
                    f"got tp_size={stage.tp_size}."
                )
            if stage.tp_size == 1:
                continue
            if not isinstance(stage.gpu, list):
                raise ValueError(
                    "Ming-Omni-TTS tts_engine tensor parallelism requires "
                    "gpu=[rank0_gpu, rank1_gpu, ...]."
                )
            if len(stage.gpu) != stage.tp_size:
                raise ValueError(
                    "Ming-Omni-TTS tts_engine TP GPU list length must match "
                    f"tp_size; got gpu={stage.gpu!r}, tp_size={stage.tp_size}."
                )


EntryClass = MingTTSPipelineConfig
