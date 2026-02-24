# SPDX-License-Identifier: Apache-2.0
"""Generic pipeline config factory for omni models."""

from __future__ import annotations

from typing import Any, TypedDict

from sglang_omni.config import (
    ExecutorConfig,
    InputHandlerConfig,
    PipelineConfig,
    RelayConfig,
    StageConfig,
)


class StageSpec(TypedDict, total=False):
    """Declarative stage spec consumed by create_omni_pipeline_config."""

    name: str
    executor_factory: str
    executor_args: dict[str, Any]
    get_next: str
    relay_device: str
    input_handler: dict[str, Any]
    num_workers: int


def create_omni_pipeline_config(
    *,
    name: str,
    entry_stage: str,
    stages: list[StageSpec],
    relay_type: str = "shm",
    fused_stages: list[list[str]] | None = None,
) -> PipelineConfig:
    """Build a PipelineConfig from stage specs."""

    stage_configs: list[StageConfig] = []
    for spec in stages:
        stage_configs.append(
            StageConfig(
                name=spec["name"],
                executor=ExecutorConfig(
                    factory=spec["executor_factory"],
                    args=spec.get("executor_args", {}),
                ),
                get_next=spec["get_next"],
                input_handler=InputHandlerConfig(**spec.get("input_handler", {})),
                relay=RelayConfig(
                    type=relay_type,
                    device=spec.get("relay_device", "cpu"),
                ),
                num_workers=int(spec.get("num_workers", 1)),
            )
        )

    return PipelineConfig(
        name=name,
        entry_stage=entry_stage,
        fused_stages=fused_stages or [],
        stages=stage_configs,
    )
