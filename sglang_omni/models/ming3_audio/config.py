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
from sglang_omni.models.ming3_audio.text_processing import (
    DEFAULT_ENABLE_TEXT_SEGMENTATION,
    DEFAULT_MAX_FRAGMENT_CHARS,
    DEFAULT_MAX_FRAGMENTS,
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_SEMANTIC_LENGTH,
    DEFAULT_MAX_TOTAL_DECODE_STEPS,
    MAX_ALLOWED_FRAGMENT_CHARS,
    maximum_admitted_fragment_count,
)

PACKAGE = "sglang_omni.models.ming3_audio"


class Ming3TalkerFactoryArgs(FactoryArgs):
    """Validated constructor controls for the Ming 3 Talker stage."""

    max_decode_steps: int = Field(default=512, gt=0)
    seed: int = 123
    enable_cuda_graph: bool = True
    enable_text_segmentation: bool = DEFAULT_ENABLE_TEXT_SEGMENTATION
    max_semantic_length: int = Field(default=DEFAULT_MAX_SEMANTIC_LENGTH, ge=1)
    max_fragment_characters: int = Field(
        default=DEFAULT_MAX_FRAGMENT_CHARS,
        ge=1,
        le=MAX_ALLOWED_FRAGMENT_CHARS,
    )
    max_fragments: int = Field(default=DEFAULT_MAX_FRAGMENTS, ge=1)
    max_input_characters: int = Field(default=DEFAULT_MAX_INPUT_CHARS, ge=1)
    max_total_decode_steps: int = Field(default=DEFAULT_MAX_TOTAL_DECODE_STEPS, ge=1)

    @model_validator(mode="after")
    def validate_decode_budget(self) -> "Ming3TalkerFactoryArgs":
        if self.enable_text_segmentation:
            maximum_fragment_count = maximum_admitted_fragment_count(
                self.max_decode_steps, self.max_total_decode_steps
            )
            if self.max_fragments > maximum_fragment_count:
                raise ValueError(
                    "max_fragments must not exceed the conservative decode-budget "
                    f"ceiling of {maximum_fragment_count}"
                )
            else:
                pass
        else:
            pass
        return self


def default_stages() -> list[StageConfig]:
    talker_factory = Ming3TalkerFactoryArgs()
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
            factory=FactoryArgs(**talker_factory.model_dump(exclude_none=True)),
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

    @model_validator(mode="after")
    def validate_talker_factory(self) -> "Ming3AudioPipelineConfig":
        for stage in self.stages:
            if stage.name == "talker":
                validated_factory = Ming3TalkerFactoryArgs.model_validate(
                    stage.factory.model_dump()
                )
                normalized_factory_arguments = validated_factory.model_dump(
                    exclude_none=True
                )
                for name, value in (validated_factory.model_extra or {}).items():
                    if value is None:
                        normalized_factory_arguments[name] = None
                    else:
                        pass
                stage.factory = FactoryArgs(**normalized_factory_arguments)
            else:
                pass
        return self


EntryClass = Ming3AudioPipelineConfig
