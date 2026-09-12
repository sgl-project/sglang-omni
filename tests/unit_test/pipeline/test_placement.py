# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    PipelineConfig,
    StageConfig,
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
    total_reserve_bytes: int | None = None,
    tp_size: int = 1,
    terminal: bool = False,
    next_stage: str | None = None,
) -> StageConfig:
    # Note (Jiaxin Deng): the engine block only exists on engine stages, so a
    # byte-budgeted test stage must be an EngineStageConfig.
    if kv_cache_bytes is not None:
        return EngineStageConfig(
            name=name,
            process="pipeline",
            factory_path=_FACTORY,
            gpu=gpu,
            tp_size=tp_size,
            gpu_memory_fraction=fraction,
            total_reserve_bytes=total_reserve_bytes,
            engine=EngineArgs(kv_cache_bytes=kv_cache_bytes),
            next=next_stage,
            terminal=terminal,
        )
    return StageConfig(
        name=name,
        process="pipeline",
        factory_path=_FACTORY,
        gpu=gpu,
        tp_size=tp_size,
        gpu_memory_fraction=fraction,
        total_reserve_bytes=total_reserve_bytes,
        next=next_stage,
        terminal=terminal,
    )


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
    assert plan.gpus[0].total_reserve_bytes == 0


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


def test_mixed_fraction_and_byte_stages_sum_in_their_own_domains() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.10, next_stage="thinker"),
            _stage(
                "thinker",
                gpu=0,
                kv_cache_bytes=2 * 1024**3,
                total_reserve_bytes=8 * 1024**3,
                terminal=True,
            ),
        ],
    )

    plan = build_stage_placement_plan(config)

    assert plan.stages["preprocess"].kv_cache_bytes is None
    assert plan.gpus[0].total_kv_cache_bytes == 2 * 1024**3
    assert plan.gpus[0].total_reserve_bytes == 8 * 1024**3
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.10)


def test_placement_summary_includes_kv_cache_bytes(monkeypatch) -> None:
    pytest.importorskip("sglang")
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
    assert summary["gpus"][0]["total_reserve_bytes"] == 0


def test_gpu_capacity_check_rejects_combined_overcommit(monkeypatch) -> None:
    import sglang_omni.utils.gpu_memory as gpu_memory
    from sglang_omni.config.placement import validate_gpu_capacity

    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.90, next_stage="thinker"),
            _stage(
                "thinker",
                gpu=0,
                kv_cache_bytes=2 * 1024**3,
                total_reserve_bytes=20 * 1024**3,
                terminal=True,
            ),
        ],
    )
    plan = build_stage_placement_plan(config)
    monkeypatch.setattr(
        gpu_memory,
        "get_gpu_device_info",
        lambda gpu_id: SimpleNamespace(total_memory_bytes=80 * 1024**3),
    )

    with pytest.raises(ValueError, match="exceed physical memory"):
        validate_gpu_capacity(plan)


def test_gpu_capacity_check_passes_within_card(monkeypatch) -> None:
    import sglang_omni.utils.gpu_memory as gpu_memory
    from sglang_omni.config.placement import validate_gpu_capacity

    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("preprocess", gpu=0, fraction=0.50, next_stage="thinker"),
            _stage(
                "thinker",
                gpu=0,
                kv_cache_bytes=2 * 1024**3,
                total_reserve_bytes=20 * 1024**3,
                terminal=True,
            ),
        ],
    )
    plan = build_stage_placement_plan(config)
    monkeypatch.setattr(
        gpu_memory,
        "get_gpu_device_info",
        lambda gpu_id: SimpleNamespace(total_memory_bytes=80 * 1024**3),
    )

    validate_gpu_capacity(plan)


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


def test_tp_scalar_gpu_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires a list"):
        _stage(
            "thinker",
            gpu=0,
            fraction=0.45,
            tp_size=2,
            terminal=True,
        )


def test_tp_duplicate_gpu_ids_are_rejected() -> None:
    # Shape rules live on StageConfig now: the duplicate ids never survive
    # long enough to reach the placement planner.
    with pytest.raises(ValueError, match="unique GPU ids"):
        _stage(
            "thinker",
            gpu=[0, 0],
            fraction=0.45,
            tp_size=2,
            terminal=True,
        )


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


def test_gpu_capacity_check_rejects_kv_pools_over_vram(monkeypatch) -> None:
    import sglang_omni.utils.gpu_memory as gpu_memory
    from sglang_omni.config.placement import validate_gpu_capacity

    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("thinker", gpu=0, kv_cache_bytes=100 * 1024**3, terminal=True),
        ],
    )
    plan = build_stage_placement_plan(config)
    monkeypatch.setattr(
        gpu_memory,
        "get_gpu_device_info",
        lambda gpu_id: SimpleNamespace(total_memory_bytes=80 * 1024**3),
    )

    with pytest.raises(ValueError, match="KV pools alone exceed physical memory"):
        validate_gpu_capacity(plan)


def test_gpu_capacity_check_sums_replica_kv_pools(monkeypatch) -> None:
    import sglang_omni.utils.gpu_memory as gpu_memory
    from sglang_omni.config.placement import validate_gpu_capacity
    from sglang_omni.config.schema import ProcessConfig
    from sglang_omni.config.topology import compile_logical_processes
    from sglang_omni.pipeline.replicas import expand_replica_stages

    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("thinker", gpu=0, kv_cache_bytes=60 * 1024**3, terminal=True),
        ],
        processes={
            "pipeline": ProcessConfig(num_replicas=2, replica_devices=[0, 0]),
        },
    )
    logical_plan, stages_cfg = compile_logical_processes(config)
    stages_cfg, replica_topology = expand_replica_stages(stages_cfg, logical_plan)
    plan = build_stage_placement_plan(
        config,
        stages_cfg=stages_cfg,
        replica_instances=replica_topology.replicas,
    )
    monkeypatch.setattr(
        gpu_memory,
        "get_gpu_device_info",
        lambda gpu_id: SimpleNamespace(total_memory_bytes=80 * 1024**3),
    )

    assert plan.gpus[0].total_kv_cache_bytes == 120 * 1024**3
    with pytest.raises(ValueError, match="KV pools alone exceed physical memory"):
        validate_gpu_capacity(plan)
