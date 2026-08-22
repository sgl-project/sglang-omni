# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    StageMemoryConfig,
    StageResourceConfig,
    StageRuntimeConfig,
    build_stage_placement_plan,
    resolve_stage_gpu_ids,
)

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


def _stage(
    name: str,
    *,
    gpu: int | list[int] | None = None,
    fraction: float | None = None,
    kv_cache_bytes: int | None = None,
    tp_size: int = 1,
    terminal: bool = False,
    next_stage: str | None = None,
) -> StageConfig:
    return StageConfig(
        name=name,
        process="pipeline",
        factory=_FACTORY,
        gpu=gpu,
        tp_size=tp_size,
        runtime=StageRuntimeConfig(
            memory=StageMemoryConfig(kv_cache_bytes=kv_cache_bytes),
            resources=StageResourceConfig(total_gpu_memory_fraction=fraction),
        ),
        next=next_stage,
        terminal=terminal,
    )


def test_same_gpu_placement_records_missing_memory_fraction_stages() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.10, next_stage="thinker"),
            _stage("thinker", gpu=0, terminal=True),
        ],
    )

    plan = build_stage_placement_plan(config)

    assert plan.gpus[0].missing_fraction_stage_names == ("thinker",)


def test_same_gpu_without_budget_records_placement() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("image_encoder", gpu=0, next_stage="thinker"),
            _stage("thinker", gpu=0, terminal=True),
        ],
    )

    plan = build_stage_placement_plan(config)

    assert plan.gpus[0].stage_names == ("image_encoder", "thinker")
    assert plan.gpus[0].missing_fraction_stage_names == (
        "image_encoder",
        "thinker",
    )


def test_untyped_factory_budget_is_rejected_before_placement() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="thinker",
                process="pipeline",
                factory=_FACTORY,
                factory_args={"total_gpu_memory_fraction": 0.50},
                gpu=0,
                terminal=True,
            )
        ],
    )

    with pytest.raises(ValueError, match="runtime.resources.total_gpu_memory_fraction"):
        build_stage_placement_plan(config)


def test_untyped_runtime_override_budget_is_rejected_before_placement() -> None:
    config = PipelineConfig(
        model_path="dummy",
        runtime_overrides={"thinker": {"total_gpu_memory_fraction": 0.50}},
        stages=[_stage("thinker", gpu=0, terminal=True)],
    )

    with pytest.raises(ValueError, match="runtime.resources.total_gpu_memory_fraction"):
        build_stage_placement_plan(config)


def test_same_gpu_colocation_sums_budget() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.10, next_stage="thinker"),
            _stage("thinker", gpu=0, fraction=0.70, terminal=True),
        ],
    )

    plan = build_stage_placement_plan(config)

    assert plan.gpus[0].stage_names == ("preprocess", "thinker")
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.80)


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


def test_tp_memory_fraction_is_per_rank_per_assigned_gpu() -> None:
    stage = _stage(
        "thinker",
        gpu=[0, 1],
        fraction=0.30,
        tp_size=2,
        terminal=True,
    )
    config = PipelineConfig(model_path="dummy", stages=[stage])

    plan = build_stage_placement_plan(config)

    assert plan.gpus[0].stage_names == ("thinker",)
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.30)
    assert plan.gpus[1].stage_names == ("thinker",)
    assert plan.gpus[1].total_gpu_memory_fraction == pytest.approx(0.30)


def test_same_gpu_colocation_sums_kv_byte_budgets() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage(
                "preprocess",
                gpu=0,
                kv_cache_bytes=1024**3,
                next_stage="thinker",
            ),
            _stage(
                "thinker",
                gpu=0,
                kv_cache_bytes=2 * 1024**3,
                terminal=True,
            ),
        ],
    )

    plan = build_stage_placement_plan(config)

    assert plan.stages["preprocess"].kv_cache_bytes == 1024**3
    assert plan.stages["thinker"].kv_cache_bytes == 2 * 1024**3
    assert plan.gpus[0].total_kv_cache_bytes == 3 * 1024**3
    assert plan.gpus[0].missing_kv_cache_bytes_stage_names == ()


def test_tp_kv_cache_bytes_is_per_rank_per_assigned_gpu() -> None:
    stage = _stage(
        "thinker",
        gpu=[0, 1],
        kv_cache_bytes=2 * 1024**3,
        tp_size=2,
        terminal=True,
    )
    config = PipelineConfig(model_path="dummy", stages=[stage])

    plan = build_stage_placement_plan(config)

    assert plan.stages["thinker"].kv_cache_bytes == 2 * 1024**3
    assert plan.gpus[0].total_kv_cache_bytes == 2 * 1024**3
    assert plan.gpus[1].total_kv_cache_bytes == 2 * 1024**3


def test_fraction_only_stage_is_marked_missing_kv_byte_budget() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.10, next_stage="thinker"),
            _stage("thinker", gpu=0, kv_cache_bytes=2 * 1024**3, terminal=True),
        ],
    )

    plan = build_stage_placement_plan(config)

    assert plan.stages["preprocess"].kv_cache_bytes is None
    assert plan.gpus[0].total_kv_cache_bytes == 2 * 1024**3
    assert plan.gpus[0].missing_kv_cache_bytes_stage_names == ("preprocess",)


def test_placement_summary_includes_kv_cache_bytes(monkeypatch) -> None:
    pytest.importorskip("uvicorn")
    from sglang_omni.serve import launcher

    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.10, next_stage="thinker"),
            _stage("thinker", gpu=0, kv_cache_bytes=2 * 1024**3, terminal=True),
        ],
    )
    plan = build_stage_placement_plan(config)
    process_plan = SimpleNamespace(groups=(), tp_stage_to_processes={})

    monkeypatch.setattr(
        launcher,
        "get_gpu_device_info",
        lambda gpu_id: SimpleNamespace(
            device_id=gpu_id,
            name=f"gpu-{gpu_id}",
            total_memory_bytes=24 * 1024**3,
        ),
    )

    summary = launcher._placement_log_summary(
        plan,
        process_plan,
        config,
    )

    assert summary["stage_runtime"]["thinker"]["kv_cache_bytes"] == 2 * 1024**3
    assert summary["gpus"][0]["total_kv_cache_bytes"] == 2 * 1024**3
    assert summary["gpus"][0]["missing_kv_cache_bytes_stages"] == ["preprocess"]


def test_tp_scalar_gpu_is_rejected() -> None:
    stage = _stage(
        "thinker",
        gpu=0,
        fraction=0.45,
        tp_size=2,
        terminal=True,
    )
    config = PipelineConfig(model_path="dummy", stages=[stage])

    with pytest.raises(ValueError, match="requires a list"):
        build_stage_placement_plan(config)


def test_tp_duplicate_gpu_ids_are_rejected() -> None:
    stage = _stage(
        "thinker",
        gpu=[0, 0],
        fraction=0.45,
        tp_size=2,
        terminal=True,
    )
    config = PipelineConfig(model_path="dummy", stages=[stage])

    with pytest.raises(ValueError, match="unique GPU ids"):
        build_stage_placement_plan(config)


def test_placement_policy_hook_runs_after_generic_plan() -> None:
    config = PipelineConfig(
        model_path="dummy",
        placement_policy=(
            "tests.unit_test.fixtures.pipeline_fakes.RejectThinkerPlacementPolicy"
        ),
        stages=[_stage("thinker", gpu=0, terminal=True)],
    )

    with pytest.raises(ValueError, match="policy rejected thinker"):
        build_stage_placement_plan(config)
