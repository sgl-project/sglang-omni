# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for YuE2."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)

_PKG = "sglang_omni.models.yue2"


def stages(*, synth_gpu: int = 0) -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="yue2_synth",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            next="yue2_synth",
        ),
        StageConfig(
            name="yue2_synth",
            process="yue2_synth",
            factory_path=f"{_PKG}.stages.create_synth_executor",
            gpu=synth_gpu,
            terminal=True,
        ),
    ]


class Yue2PipelineConfig(PipelineConfig):
    """Preprocessing -> terminal YuE2 synthesis on a single GPU."""

    architecture: ClassVar[str] = "YuE2ForCausalLM"
    requires_model_capabilities: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "yue2_synth": StageConfig,
    }

    stages: list[StageConfig] = Field(default_factory=stages)
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({("preprocessing", "yue2_synth")})


EntryClass = Yue2PipelineConfig

Variants = {
    "default": Yue2PipelineConfig,
}
