# SPDX-License-Identifier: Apache-2.0
"""Ming 3 audio pipeline configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator

from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)

PACKAGE = "sglang_omni.models.ming3_audio"


def default_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="preprocessing",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            next=["audio_encoder", "talker"],
            route_fn=f"{PACKAGE}.routing.route_preprocessing",
        ),
        StageConfig(
            name="audio_encoder",
            process="audio_encoder",
            gpu=0,
            factory_path=f"{PACKAGE}.stages.create_audio_encoder_executor",
            next="thinker",
        ),
        EngineStageConfig(
            name="thinker",
            process="thinker",
            factory_path=f"{PACKAGE}.stages.create_thinker_executor",
            factory=FactoryArgs(max_seq_len=8192, enable_async_decode=False),
            engine=EngineArgs(
                mem_fraction_static=0.93,
                max_running_requests=8,
                disable_cuda_graph=True,
                enable_torch_compile=False,
            ),
            next=["decode", "talker"],
            route_fn=f"{PACKAGE}.routing.route_thinker",
        ),
        StageConfig(
            name="decode",
            process="preprocessing",
            factory_path=f"{PACKAGE}.stages.create_decode_executor",
            terminal=True,
        ),
        StageConfig(
            name="talker",
            process="talker",
            gpu=1,
            factory_path=f"{PACKAGE}.stages.create_talker_executor",
            factory=FactoryArgs(max_decode_steps=512, seed=123, enable_cuda_graph=True),
            next="decode",
        ),
    ]


class Ming3AudioPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "BailingMoeV3AudioForConditionalGeneration"

    @model_validator(mode="before")
    @classmethod
    def place_thinker_on_visible_gpus(cls, data: object) -> object:
        # note (zhaochenyang): StageConfig requires a GPU list when tp_size > 1.
        # The serve flag sets only tp_size; logical ids follow visible-device order.
        if not isinstance(data, dict):
            return data
        else:
            pass
        stages = data.get("stages")
        if not isinstance(stages, list):
            return data
        else:
            pass
        placed_stages: list[object] = []
        for stage in stages:
            tp_size = stage.get("tp_size", 1) if isinstance(stage, dict) else 1
            if (
                isinstance(stage, dict)
                and stage.get("name") == "thinker"
                and stage.get("gpu") is None
                and isinstance(tp_size, int)
                and tp_size > 1
            ):
                placed_stages.append({**stage, "gpu": list(range(tp_size))})
            else:
                placed_stages.append(stage)
        return {**data, "stages": placed_stages}

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "thinker": EngineStageConfig,
    }
    model_path: str
    entry_stage: str = "preprocessing"
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=default_stages)


EntryClass = Ming3AudioPipelineConfig
