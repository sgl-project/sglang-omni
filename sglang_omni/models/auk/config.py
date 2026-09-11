# SPDX-License-Identifier: Apache-2.0
"""AuK: preprocessing, conditioning, batched DiT sampling, and VAE decode."""

from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.models.auk import constants as C
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.auk"
PREPROCESSING_STAGE = "preprocessing"
CONDITIONING_STAGE = "conditioning"
ENGINE_STAGE = "auk_engine"
DECODE_STAGE = "decode"


class AuKPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "AuKForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("AuK", "AuK-Flash")
    required_speech_reference_count: ClassVar[int | None] = None

    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=8),
            next=CONDITIONING_STAGE,
        ),
        StageConfig(
            name=CONDITIONING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_conditioning_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                text_encoder_path=C.DEFAULT_TEXT_ENCODER,
                max_batch_size=8,
                max_batch_wait_ms=10,
            ),
            gpu=0,
            next=ENGINE_STAGE,
        ),
        StageConfig(
            name=ENGINE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_auk_engine_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                nfe=C.DEFAULT_NFE,
                enable_dit_fused_qk_norm_rope=False,
                cfg_strength=C.DEFAULT_CFG_STRENGTH,
                sway_sampling_coef=C.DEFAULT_SWAY_SAMPLING_COEF,
                max_seconds=C.MAX_SECONDS,
                max_batch_size=16,
                max_batch_wait_ms=10,
                weight_dtype="bfloat16",
            ),
            gpu=0,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_decode_executor",
            factory=FactoryArgs(device=current_platform.device_type, max_batch_size=4),
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = AuKPipelineConfig
