# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni_v1.config import (
    PipelineConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
    build_stage_placement_plan,
    compile_pipeline_core,
    resolve_pipeline_process_mode,
    resolve_stage_gpu_ids,
)


_FACTORY = "tests.v1_dummy_factories.dummy_factory"


def _stage(
    name: str,
    *,
    gpu: int | list[int] | None = None,
    fraction: float | None = None,
    tp_size: int = 1,
    terminal: bool = False,
    next_stage: str | None = None,
) -> StageConfig:
    return StageConfig(
        name=name,
        factory=_FACTORY,
        gpu=gpu,
        tp_size=tp_size,
        runtime=StageRuntimeConfig(
            resources=StageResourceConfig(total_gpu_memory_fraction=fraction)
        ),
        next=next_stage,
        terminal=terminal,
    )


def test_same_gpu_colocation_requires_memory_fraction_for_all_stages() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.10, next_stage="thinker"),
            _stage("thinker", gpu=0, terminal=True),
        ],
    )

    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        build_stage_placement_plan(config)


def test_same_gpu_without_budget_stays_legacy_single_process() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("image_encoder", gpu=0, next_stage="thinker"),
            _stage("thinker", gpu=0, terminal=True),
        ],
    )

    plan = build_stage_placement_plan(config)

    assert plan.requires_multi_process is False
    assert resolve_pipeline_process_mode(config, plan) is False


def test_same_gpu_colocation_sums_budget_and_requires_multi_process() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.10, next_stage="thinker"),
            _stage("thinker", gpu=0, fraction=0.70, terminal=True),
        ],
    )

    plan = build_stage_placement_plan(config)

    assert plan.gpus[0].stage_names == ("preprocess", "thinker")
    assert plan.gpus[0].total_gpu_memory_fraction == 0.80
    assert plan.requires_multi_process is True
    assert resolve_pipeline_process_mode(config, plan) is True


def test_same_gpu_colocation_rejects_over_budget() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.35, next_stage="thinker"),
            _stage("thinker", gpu=0, fraction=0.75, terminal=True),
        ],
    )

    with pytest.raises(ValueError, match="exceeds placement limit"):
        build_stage_placement_plan(config)


def test_single_process_mode_rejects_colocated_gpu_plan() -> None:
    config = PipelineConfig(
        model_path="dummy",
        process={"mode": "single"},
        stages=[
            _stage("preprocess", gpu=0, fraction=0.10, next_stage="thinker"),
            _stage("thinker", gpu=0, fraction=0.70, terminal=True),
        ],
    )
    plan = build_stage_placement_plan(config)

    with pytest.raises(ValueError, match="process.mode='single'"):
        resolve_pipeline_process_mode(config, plan)


def test_single_process_compiler_rejects_colocated_gpu_plan() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.10, next_stage="thinker"),
            _stage("thinker", gpu=0, fraction=0.70, terminal=True),
        ],
    )

    with pytest.raises(ValueError, match="MultiProcessPipelineRunner"):
        compile_pipeline_core(config)


def test_tp_rank_gpu_ids_are_preserved() -> None:
    stage = _stage(
        "thinker",
        gpu=[0, 1],
        fraction=0.45,
        tp_size=2,
        terminal=True,
    )
    config = PipelineConfig(model_path="dummy", stages=[stage])

    plan = build_stage_placement_plan(config)

    assert resolve_stage_gpu_ids(plan, stage) == [0, 1]
    assert plan.requires_multi_process is True


def test_placement_policy_hook_runs_after_generic_plan() -> None:
    config = PipelineConfig(
        model_path="dummy",
        placement_policy="tests.v1_dummy_factories.RejectThinkerPlacementPolicy",
        stages=[_stage("thinker", gpu=0, terminal=True)],
    )

    with pytest.raises(ValueError, match="policy rejected thinker"):
        build_stage_placement_plan(config)
