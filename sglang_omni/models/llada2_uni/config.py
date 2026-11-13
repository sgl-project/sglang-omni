# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for LLaDA2-Uni (Diffusion LLM)."""

from __future__ import annotations

import os
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from typing import ClassVar, Literal

import torch
import typer
from pydantic import Field, model_validator

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.pipeline.umm import (
    CONTROLLER_FACTORY_PATH,
    OUTPUT_FACTORY_PATH,
    ROUTE_FN_PATH,
    STREAM_DONE_FN_PATH,
    UMMPipelineConfig,
)

_PKG = "sglang_omni.models.llada2_uni"

PREPROCESSING_STAGE = "preprocessing"
IMAGE_STAGE = "image_encoder"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"
IMAGE_DECODE_STAGE = "image_decode"

DEFAULT_THINKER_MAX_NEW_TOKENS = 2048


class LLaDA2OffloadFactoryArgs(FactoryArgs):
    enable_serial_offload: bool = False


class LLaDA2ImageEncoderStageConfig(StageConfig):
    factory: LLaDA2OffloadFactoryArgs = Field(default_factory=LLaDA2OffloadFactoryArgs)


class LLaDA2ThinkerStageConfig(EngineStageConfig):
    factory: LLaDA2OffloadFactoryArgs = Field(default_factory=LLaDA2OffloadFactoryArgs)


class LLaDA2ImageDecoderFactoryArgs(LLaDA2OffloadFactoryArgs):
    backend: Literal["diffusers", "sglang"] = "diffusers"
    decode_mode: Literal["normal", "decoder-turbo"] = "normal"
    num_steps: int = Field(default=50, ge=1)
    resolution_multiplier: int = Field(default=2, ge=1)
    attention_backend: str = "torch_sdpa"


class LLaDA2ImageDecoderStageConfig(StageConfig):
    factory: LLaDA2ImageDecoderFactoryArgs = Field(
        default_factory=LLaDA2ImageDecoderFactoryArgs
    )


class LLaDA2UniPipelineConfig(PipelineConfig):
    """4-stage DLLM pipeline: preprocessing → image_encoder → thinker → decode."""

    architecture: ClassVar[str] = "LLaDA2MoeModelLM"

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        THINKER_STAGE: LLaDA2ThinkerStageConfig,
        IMAGE_STAGE: LLaDA2ImageEncoderStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_seq_len=8192),
            next=IMAGE_STAGE,
        ),
        LLaDA2ImageEncoderStageConfig(
            name=IMAGE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_image_encoder_executor",
            gpu=0,
            next=THINKER_STAGE,
        ),
        LLaDA2ThinkerStageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory=LLaDA2OffloadFactoryArgs(
                max_seq_len=8192, dllm_algorithm="LowConfidenceCFG"
            ),
            gpu=0,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
    ]

    @model_validator(mode="after")
    def validate_serial_offload(self) -> LLaDA2UniPipelineConfig:
        enabled = {
            stage.name
            for stage in self.stages
            if isinstance(stage.factory, LLaDA2OffloadFactoryArgs)
            and stage.factory.enable_serial_offload
        }
        if not enabled:
            return self
        elif not self.supports_image_api:
            raise ValueError(
                "Serial offload supports only the single-image omni pipeline"
            )
        elif enabled != {IMAGE_STAGE, THINKER_STAGE, IMAGE_DECODE_STAGE}:
            raise ValueError(
                "Serial offload must enable image_encoder, thinker and image_decode together"
            )
        else:
            pass
        stages = [self.stage_named(name) for name in sorted(enabled)]
        if (
            any(stage.process != stages[0].process for stage in stages)
            or not stages[0].process
        ):
            raise ValueError("Serial offload stages must share one process")
        elif any(
            stage.gpu != stages[0].gpu or stage.tp_size != 1 for stage in stages
        ) or not isinstance(stages[0].gpu, int):
            raise ValueError("Serial offload requires the same single GPU and TP=1")
        else:
            pass
        process = self.processes.get(stages[0].process)
        if process is not None and process.num_replicas != 1:
            raise ValueError("Serial offload requires one process replica")
        else:
            pass
        decoder = self.stage_named(IMAGE_DECODE_STAGE)
        assert isinstance(decoder, LLaDA2ImageDecoderStageConfig)
        if decoder.factory.backend != "diffusers":
            raise ValueError("Serial offload requires the diffusers decoder backend")
        else:
            pass
        return self


class LLaDA2UniOmniPipelineConfig(LLaDA2UniPipelineConfig):
    """LLaDA text, image generation and editing with a shared thinker."""

    supports_image_api: ClassVar[bool] = True

    def resolved_env_defaults(self) -> dict[str, str]:
        environment = super().resolved_env_defaults()
        thinker = self.stage_named(THINKER_STAGE)
        assert isinstance(thinker.factory, LLaDA2OffloadFactoryArgs)
        if not thinker.factory.enable_serial_offload:
            return environment
        else:
            pass
        package = find_spec("torch_memory_saver")
        if package is None or package.origin is None or torch.version.cuda is None:
            raise RuntimeError(
                "LLaDA2-Uni offload requires CUDA and torch_memory_saver==0.0.10"
            )
        elif version("torch_memory_saver") != "0.0.10":
            raise RuntimeError("LLaDA2-Uni offload requires torch_memory_saver==0.0.10")
        else:
            pass
        package_directory = Path(package.origin).parent
        cuda_major = torch.version.cuda.split(".")[0]
        pattern = f"torch_memory_saver_hook_mode_preload_cu{cuda_major}.*.so"
        libraries = [
            library
            for directory in (package_directory, package_directory.parent)
            for library in directory.glob(pattern)
        ]
        if len(libraries) != 1:
            raise RuntimeError(
                f"LLaDA2-Uni offload requires exactly one allocator library matching {pattern}"
            )
        else:
            pass
        preload = os.environ.get("LD_PRELOAD", "")
        if "torch_memory_saver" not in preload:
            environment["LD_PRELOAD"] = (
                f"{libraries[0]}:{preload}" if preload else str(libraries[0])
            )
        else:
            pass
        return environment

    def with_serial_offload(
        self, components: frozenset[str]
    ) -> LLaDA2UniOmniPipelineConfig:
        required = {"thinker", "decoder", "vae"}
        if components != required:
            raise typer.BadParameter(
                "LLaDA2-Uni --stage-offload-components requires all of: thinker,decoder,vae"
            )
        else:
            pass
        configuration = self.model_dump()
        thinker = self.stage_named(THINKER_STAGE)
        process = thinker.process or thinker.name
        for stage in configuration["stages"]:
            if stage["name"] in {IMAGE_STAGE, THINKER_STAGE, IMAGE_DECODE_STAGE}:
                stage["process"] = process
                stage["factory"]["enable_serial_offload"] = True
            else:
                pass
        return type(self)(**configuration)

    def image_decode_mode(self) -> Literal["normal", "decoder-turbo"]:
        stage = self.stage_named(IMAGE_DECODE_STAGE)
        assert isinstance(stage, LLaDA2ImageDecoderStageConfig)
        return stage.factory.decode_mode

    def image_decode_steps(self) -> int:
        stage = self.stage_named(IMAGE_DECODE_STAGE)
        assert isinstance(stage, LLaDA2ImageDecoderStageConfig)
        return stage.factory.num_steps

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        **LLaDA2UniPipelineConfig.stage_config_types,
        IMAGE_DECODE_STAGE: LLaDA2ImageDecoderStageConfig,
    }

    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_seq_len=8192),
            next=IMAGE_STAGE,
        ),
        LLaDA2ImageEncoderStageConfig(
            name=IMAGE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_image_encoder_executor",
            gpu=0,
            gpu_memory_fraction=0.05,
            next=THINKER_STAGE,
        ),
        LLaDA2ThinkerStageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory=LLaDA2OffloadFactoryArgs(
                max_seq_len=8192, dllm_algorithm="LowConfidenceCFG"
            ),
            engine=EngineArgs(mem_fraction_static=0.75),
            gpu=0,
            gpu_memory_fraction=0.75,
            next=[THINKER_STAGE, DECODE_STAGE, IMAGE_DECODE_STAGE],
            route_fn=f"{_PKG}.request_builders.thinker_next",
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
        LLaDA2ImageDecoderStageConfig(
            name=IMAGE_DECODE_STAGE,
            process=IMAGE_DECODE_STAGE,
            factory_path=f"{_PKG}.stages.create_image_decode_executor",
            gpu=0,
            gpu_memory_fraction=0.20,
            terminal=True,
        ),
    ]


class LLaDA2UniInterleavedPipelineConfig(UMMPipelineConfig, LLaDA2UniPipelineConfig):
    """Text and images in one answer, on the shared UMM controller.

    The orchestrator runs the LLaDA2-Uni adapter. Phase prepare runs on the
    preprocessing stage, think on the thinker, and decode on the image
    decoder, which decodes each finished image while the thinker writes on.
    """

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        **LLaDA2UniPipelineConfig.stage_config_types,
        IMAGE_DECODE_STAGE: LLaDA2ImageDecoderStageConfig,
    }
    entry_stage: str = "orchestrator"
    stages: list[StageConfig] = [
        StageConfig(
            name="orchestrator",
            process="orchestrator",
            factory_path=CONTROLLER_FACTORY_PATH,
            factory=FactoryArgs(
                adapter=f"{_PKG}.umm.LLaDA2UniUMMAdapter",
                phases={
                    "prepare": PREPROCESSING_STAGE,
                    "think": THINKER_STAGE,
                    "decode": IMAGE_DECODE_STAGE,
                },
                # Ten frames, the default max_frames: preparation, a text and
                # an image phase per frame, the closing text, one decode each.
                max_dispatches=32,
                # A text and an image per frame and the closing text.
                max_outputs=21,
                # Bounds orchestrator memory. The thinker engine queues the
                # requests it cannot run yet.
                max_active=32,
                # Ten frames as PNG data URLs at the largest grid the default
                # max_image_tokens allows (2048 pixels square, about 17 MiB
                # each), and their text.
                max_retained_bytes=256 * 1024 * 1024,
                timeout_s=3600.0,
            ),
            next=[PREPROCESSING_STAGE, THINKER_STAGE, IMAGE_DECODE_STAGE, "output"],
            route_fn=ROUTE_FN_PATH,
            stream_to=["output"],
            stream_done_to_fn=STREAM_DONE_FN_PATH,
        ),
        StageConfig(
            name="output",
            process="orchestrator",
            factory_path=OUTPUT_FACTORY_PATH,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_seq_len=8192, interleaved=True),
            next="orchestrator",
        ),
        LLaDA2ThinkerStageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_dllm_thinker_executor_from_config",
            factory=LLaDA2OffloadFactoryArgs(
                max_seq_len=8192, dllm_algorithm="LowConfidenceCFG"
            ),
            engine=EngineArgs(mem_fraction_static=0.75),
            gpu=0,
            gpu_memory_fraction=0.75,
            next="orchestrator",
        ),
        LLaDA2ImageDecoderStageConfig(
            name=IMAGE_DECODE_STAGE,
            process=IMAGE_DECODE_STAGE,
            factory_path=f"{_PKG}.stages.create_image_decode_executor",
            gpu=0,
            gpu_memory_fraction=0.20,
            next="orchestrator",
        ),
    ]


EntryClass = LLaDA2UniOmniPipelineConfig

Variants = {
    "text": LLaDA2UniPipelineConfig,
    "omni": LLaDA2UniOmniPipelineConfig,
    "interleaved": LLaDA2UniInterleavedPipelineConfig,
}
