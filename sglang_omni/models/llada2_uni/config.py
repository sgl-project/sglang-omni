# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for LLaDA2-Uni (Diffusion LLM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from sglang_omni.config import (
    PipelineConfig,
    SequenceParallelPolicy,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)

_PKG = "sglang_omni.models.llada2_uni"

PREPROCESSING_STAGE = "preprocessing"
IMAGE_STAGE = "image_encoder"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"
IMAGE_DECODE_STAGE = "image_decode"

DEFAULT_THINKER_MAX_NEW_TOKENS = 2048
LLADA2_IMAGE_DECODER_ATTENTION_HEADS = 30
LLADA2_IMAGE_DECODER_SP_POLICY = SequenceParallelPolicy(
    attention_heads=LLADA2_IMAGE_DECODER_ATTENTION_HEADS,
    requires_power_of_two=True,
)


@dataclass(frozen=True)
class ImageDecoderRuntimeSettings:
    backend: str
    attention_backend: str


def _normalize_image_decoder_backend(backend: str) -> str:
    normalized = backend.strip().lower().replace("-", "_")
    aliases = {
        "native": "diffusers",
        "local": "diffusers",
        "omni": "diffusers",
    }
    return aliases.get(normalized, normalized)


def resolve_image_decoder_runtime_settings(
    *,
    backend: str | None,
    attention_backend: str | None,
    sp_size: int,
    ulysses_degree: int | None,
    ring_degree: int,
) -> ImageDecoderRuntimeSettings:
    """Resolve model-owned Z-Image backend and SP constraints."""
    if sp_size < 1:
        raise ValueError("Image decoder SP size must be positive")
    if sp_size & (sp_size - 1):
        raise ValueError("Image decoder SP size must be a power of two")
    if ring_degree < 1:
        raise ValueError("Image decoder ring degree must be positive")
    resolved_ulysses_degree = sp_size if ulysses_degree is None else ulysses_degree
    if resolved_ulysses_degree < 1:
        raise ValueError("Image decoder Ulysses degree must be positive")
    if sp_size != resolved_ulysses_degree * ring_degree:
        raise ValueError(
            "Image decoder sp_size must equal ulysses_degree * ring_degree: "
            f"{sp_size} != {resolved_ulysses_degree} * {ring_degree}"
        )
    if LLADA2_IMAGE_DECODER_ATTENTION_HEADS % resolved_ulysses_degree:
        raise ValueError(
            f"Image decoder Ulysses degree {resolved_ulysses_degree} must divide "
            f"the model's {LLADA2_IMAGE_DECODER_ATTENTION_HEADS} attention heads"
        )

    resolved_backend = (
        ("sglang" if sp_size > 1 else "diffusers")
        if backend is None
        else _normalize_image_decoder_backend(backend)
    )
    if resolved_backend not in {"diffusers", "sglang"}:
        raise ValueError(f"Unsupported image decoder backend: {backend!r}")
    if sp_size > 1 and resolved_backend != "sglang":
        raise ValueError("Image decoder SP requires backend='sglang'")

    if attention_backend is None:
        resolved_attention_backend = (
            "fa" if resolved_backend == "sglang" else "torch_sdpa"
        )
    else:
        resolved_attention_backend = attention_backend.strip().lower()
        if not resolved_attention_backend:
            raise ValueError("Image decoder attention backend must not be empty")

    return ImageDecoderRuntimeSettings(
        backend=resolved_backend,
        attention_backend=resolved_attention_backend,
    )


class LLaDA2UniPipelineConfig(PipelineConfig):
    """4-stage DLLM pipeline: preprocessing → image_encoder → thinker → decode."""

    architecture: ClassVar[str] = "LLaDA2MoeModelLM"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {THINKER_STAGE: THINKER_STAGE}

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"thinker_max_seq_len": 8192},
            runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
            next=IMAGE_STAGE,
        ),
        StageConfig(
            name=IMAGE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next=THINKER_STAGE,
        ),
        StageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory_args={"thinker_max_seq_len": 8192},
            gpu=0,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
    ]


class LLaDA2UniOmniPipelineConfig(PipelineConfig):
    """Native text/image pipeline with task-aware terminal routing."""

    architecture: ClassVar[str] = "LLaDA2MoeModelLM"
    terminal_stages_fn: str | None = f"{_PKG}.routing.resolve_terminal_stages"
    total_gpu_memory_fraction_dict: ClassVar[dict[str, float]] = {
        IMAGE_STAGE: 0.1,
        THINKER_STAGE: 0.7,
        IMAGE_DECODE_STAGE: 0.2,
    }

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {THINKER_STAGE: THINKER_STAGE}

    @classmethod
    def sequence_parallel_policy(
        cls, *, stage_name: str
    ) -> SequenceParallelPolicy | None:
        if stage_name == IMAGE_DECODE_STAGE:
            return LLADA2_IMAGE_DECODER_SP_POLICY
        return super().sequence_parallel_policy(stage_name=stage_name)

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"thinker_max_seq_len": 8192},
            runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
            next=IMAGE_STAGE,
        ),
        StageConfig(
            name=IMAGE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(
                    total_gpu_memory_fraction=total_gpu_memory_fraction_dict[
                        IMAGE_STAGE
                    ]
                )
            ),
            next=THINKER_STAGE,
        ),
        StageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory_args={
                "thinker_max_seq_len": 8192,
                "dllm_algorithm": "LowConfidenceCFG",
                "server_args_overrides": {
                    "disable_cuda_graph": False,
                    "max_running_requests": 3,
                },
            },
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(
                    total_gpu_memory_fraction=total_gpu_memory_fraction_dict[
                        THINKER_STAGE
                    ]
                )
            ),
            next=[DECODE_STAGE, IMAGE_DECODE_STAGE],
            route_fn=f"{_PKG}.routing.thinker_next",
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
        StageConfig(
            name=IMAGE_DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_image_decode_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(
                    total_gpu_memory_fraction=total_gpu_memory_fraction_dict[
                        IMAGE_DECODE_STAGE
                    ]
                )
            ),
            terminal=True,
        ),
    ]


EntryClass = LLaDA2UniOmniPipelineConfig

Variants = {
    "text": LLaDA2UniPipelineConfig,
    "omni": LLaDA2UniOmniPipelineConfig,
}
