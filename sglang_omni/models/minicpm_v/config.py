# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration helpers for MiniCPM-V 2.6 and MiniCPM-o 2.6.

This module defines the pipeline configurations:

MiniCPM-V 2.6 (Vision-only):
- Image input only
- Text output
- SGLang-backed LLM for production performance

MiniCPM-o 2.6 (Vision + Audio):
- Image and audio input
- Text and audio output (via CosyVoice vocoder)
- SGLang-backed LLM for production performance

Pipeline stages:
1. preprocessing: Tokenization with LLaVA-style image slicing
2. image_encoder: SigLIP + Perceiver Resampler
3. audio_encoder: Whisper encoder (MiniCPM-o only)
4. mm_aggregate: Merge encoder outputs with preprocessing
5. llm: Text/audio token generation with embedded images/audio
6. vocoder: CosyVoice audio synthesis (MiniCPM-o only)
7. decode: Token to text conversion
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    ExecutorConfig,
    InputHandlerConfig,
    PipelineConfig,
    RelayConfig,
    StageConfig,
)
from sglang_omni.models.minicpm_v.pipeline.next_stage import (
    AGGREGATE_STAGE,
    AUDIO_STAGE,
    DECODE_STAGE,
    IMAGE_STAGE,
    LLM_STAGE,
    PREPROCESSING_STAGE,
    VOCODER_STAGE,
)

# Module path prefix for executor factories
_MINICPM_PKG = "sglang_omni.models.minicpm_v.pipeline"


class MiniCPMVPipelineConfig(PipelineConfig):
    """Pipeline configuration for MiniCPM-V 2.6.

    This config uses SGLang-backed LLM for production performance.
    For Phase 1 testing with HuggingFace backend, use MiniCPMVHFPipelineConfig.
    """

    # Architecture string must match HF config.json architectures[0]
    architecture: ClassVar[str] = "MiniCPMV"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        # Stage 1: Preprocessing (CPU)
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_preprocessing_executor",
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        # Stage 2: Image Encoder (CUDA)
        StageConfig(
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_image_encoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                },
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        # Stage 3: Aggregate (CPU)
        StageConfig(
            name=AGGREGATE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_aggregate_executor",
                args={},
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.aggregate_next",
            input_handler=InputHandlerConfig(
                type="aggregated",
                sources=[PREPROCESSING_STAGE, IMAGE_STAGE],
                merge_fn=f"{_MINICPM_PKG}.merge.merge_for_llm",
            ),
            relay=RelayConfig(device="cpu"),
        ),
        # Stage 4: LLM (CUDA) - SGLang backend
        StageConfig(
            name=LLM_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_sglang_llm_executor_from_config",
                args={
                    "llm_max_seq_len": 8192,
                },
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.llm_next",
            relay=RelayConfig(device="cuda"),
        ),
        # Stage 5: Decode (CPU)
        StageConfig(
            name=DECODE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_decode_executor",
                args={},
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.decode_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]


class MiniCPMVHFPipelineConfig(PipelineConfig):
    """Pipeline configuration for MiniCPM-V 2.6 with HuggingFace backend.

    This config uses HuggingFace generate() for the LLM stage.
    Useful for Phase 1 testing before SGLang integration.
    """

    # Same architecture string as the SGLang version
    architecture: ClassVar[str] = "MiniCPMV"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        # Stage 1: Preprocessing (CPU)
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_preprocessing_executor",
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        # Stage 2: Image Encoder (CUDA)
        StageConfig(
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_image_encoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                },
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        # Stage 3: Aggregate (CPU)
        StageConfig(
            name=AGGREGATE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_aggregate_executor",
                args={},
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.aggregate_next",
            input_handler=InputHandlerConfig(
                type="aggregated",
                sources=[PREPROCESSING_STAGE, IMAGE_STAGE],
                merge_fn=f"{_MINICPM_PKG}.merge.merge_for_llm",
            ),
            relay=RelayConfig(device="cpu"),
        ),
        # Stage 4: LLM (CUDA) - HuggingFace backend
        StageConfig(
            name=LLM_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_llm_executor",
                args={
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "max_seq_len": 8192,
                },
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.llm_next",
            relay=RelayConfig(device="cuda"),
        ),
        # Stage 5: Decode (CPU)
        StageConfig(
            name=DECODE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_decode_executor",
                args={},
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.decode_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]


# EntryClass is used by the registry to discover this pipeline config
# When loading a model with architecture="MiniCPMV", this config will be used
EntryClass = MiniCPMVPipelineConfig


class MiniCPMOPipelineConfig(PipelineConfig):
    """Pipeline configuration for MiniCPM-o 2.6 (Vision + Audio I/O).

    This config extends MiniCPM-V with full audio support:
    - Whisper-based audio encoder with apm.* weight prefix
    - Audio token injection via masked_scatter in LLM
    - CosyVoice vocoder for audio output synthesis
    """

    # Architecture string for MiniCPM-o
    architecture: ClassVar[str] = "MiniCPMO"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        # Stage 1: Preprocessing (CPU)
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_preprocessing_executor",
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        # Stage 2: Image Encoder (CUDA)
        StageConfig(
            name=IMAGE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_image_encoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                },
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        # Stage 3: Audio Encoder (CUDA) - MiniCPM-o specific
        StageConfig(
            name=AUDIO_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_audio_encoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                },
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.encoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        # Stage 4: Aggregate (CPU)
        StageConfig(
            name=AGGREGATE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_aggregate_executor",
                args={},
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.aggregate_next",
            input_handler=InputHandlerConfig(
                type="aggregated",
                sources=[PREPROCESSING_STAGE, IMAGE_STAGE, AUDIO_STAGE],
                merge_fn=f"{_MINICPM_PKG}.merge.merge_for_llm",
            ),
            relay=RelayConfig(device="cpu"),
        ),
        # Stage 5: LLM (CUDA) - SGLang backend
        StageConfig(
            name=LLM_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_sglang_llm_executor_from_config",
                args={
                    "llm_max_seq_len": 8192,
                },
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.llm_next",
            relay=RelayConfig(device="cuda"),
        ),
        # Stage 6: Vocoder (CUDA) - CosyVoice audio synthesis
        StageConfig(
            name=VOCODER_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_vocoder_executor",
                args={
                    "device": "cuda",
                    "dtype": None,
                },
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.vocoder_next",
            relay=RelayConfig(device="cuda"),
        ),
        # Stage 7: Decode (CPU)
        StageConfig(
            name=DECODE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_decode_executor",
                args={},
            ),
            get_next=f"{_MINICPM_PKG}.next_stage.decode_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]
