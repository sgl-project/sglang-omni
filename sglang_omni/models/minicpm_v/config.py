# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration helpers for MiniCPM-V and MiniCPM-o.

This module defines the pipeline configurations for all supported versions:

MiniCPM-V 2.6 (Vision-only):
- SigLIP-400M + MiniCPM-3.0 LLM + Perceiver Resampler
- Image input only, text output
- SGLang-backed LLM for production performance

MiniCPM-V 4.5 (Vision-only, Qwen3 backbone):
- SigLIP2-400M + Qwen3-8B LLM + 3D-Resampler
- Enhanced vision understanding with 3D resampling
- max_position_embeddings=40960 for longer context

MiniCPM-o 2.6 (Vision + Audio):
- Image and audio input
- Text and audio output (via CosyVoice vocoder)
- SGLang-backed LLM for production performance

Pipeline stages:
1. preprocessing: Tokenization with LLaVA-style image slicing
2. image_encoder: SigLIP/SigLIP2 + Perceiver/3D Resampler
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
    """Pipeline configuration for MiniCPM-V (vision-only models).

    Supports both 2.6 and 4.5 versions:
    - 2.6: SigLIP-400M + MiniCPM-3.0 LLM + Perceiver Resampler
    - 4.5: SigLIP2-400M + Qwen3-8B LLM + 3D-Resampler

    The pipeline automatically detects the version from model config.
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


class MiniCPMV45PipelineConfig(PipelineConfig):
    """Pipeline configuration for MiniCPM-V 4.5 (Qwen3-8B backbone).

    This config is optimized for MiniCPM-V 4.5 with:
    - SigLIP2-400M vision encoder (27 layers, hidden_size=1152)
    - Qwen3-8B LLM backbone (36 layers, hidden_size=4096)
    - 3D-Resampler for enhanced spatial understanding
    - Extended context length: max_position_embeddings=40960
    """

    # Architecture string matches HF config.json architectures[0]
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
        # Stage 2: Image Encoder (CUDA) - SigLIP2 + 3D-Resampler
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
        # Stage 4: LLM (CUDA) - SGLang backend with Qwen3-8B
        StageConfig(
            name=LLM_STAGE,
            executor=ExecutorConfig(
                factory=f"{_MINICPM_PKG}.stages.create_sglang_llm_executor_from_config",
                args={
                    "llm_max_seq_len": 32768,  # Extended context for 4.5
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
    """Pipeline configuration for MiniCPM-V with HuggingFace backend.

    Supports both 2.6 and 4.5 versions with HuggingFace generate() for LLM.
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
