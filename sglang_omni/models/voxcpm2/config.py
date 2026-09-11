# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2: preprocessing, reference encode, AR + local diffusion, VAE decode."""

from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.models.voxcpm2 import constants as C
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.voxcpm2"
PREPROCESSING_STAGE = "preprocessing"
REFERENCE_ENCODE_STAGE = "reference_encode"
ENGINE_STAGE = "tts_engine"
VOCODER_STAGE = "vocoder"


class VoxCPM2PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = C.ARCHITECTURE
    required_speech_reference_count: ClassVar[int | None] = None

    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=8),
            next=REFERENCE_ENCODE_STAGE,
        ),
        StageConfig(
            name=REFERENCE_ENCODE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_reference_encode_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                max_batch_size=8,
                max_batch_wait_ms=10,
            ),
            gpu=0,
            next=ENGINE_STAGE,
        ),
        StageConfig(
            name=ENGINE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_tts_engine_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                inference_timesteps=C.DEFAULT_INFERENCE_TIMESTEPS,
                cfg_value=C.DEFAULT_CFG_VALUE,
                min_len=C.DEFAULT_MIN_LEN,
                max_len=C.DEFAULT_MAX_LEN,
            ),
            gpu=0,
            next=VOCODER_STAGE,
            stream_to=[VOCODER_STAGE],
        ),
        StageConfig(
            name=VOCODER_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                max_batch_size=4,
            ),
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


EntryClass = VoxCPM2PipelineConfig
