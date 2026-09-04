# SPDX-License-Identifier: Apache-2.0
"""Unit tests for process topology planning.

Which OS process a stage runs in is plain configuration: the ``process``
field, set in the model's config class, a config file's ``stages:`` mapping,
or a dotted flag (``--vocoder.process vocoder``). These tests pin the
planning rules: process declarations, TP rank process naming, and the
per-GPU memory budgets that sharing a GPU requires.
"""

from __future__ import annotations

import pytest

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    build_process_topology_plan,
    build_stage_placement_plan,
    compile_logical_processes,
)
from sglang_omni.config.manager import ConfigManager
from sglang_omni.pipeline.replicas import expand_replica_stages

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


def _stage(
    name: str,
    *,
    gpu: int | list[int] | None = None,
    fraction: float | None = None,
    process: str | None = None,
    tp_size: int = 1,
    terminal: bool = False,
    next_stage: str | None = None,
) -> StageConfig:
    return StageConfig(
        name=name,
        factory_path=_FACTORY,
        gpu=gpu,
        process=process,
        tp_size=tp_size,
        gpu_memory_fraction=fraction,
        next=next_stage,
        terminal=terminal,
    )


def _topology(config: PipelineConfig):
    plan, stages = compile_logical_processes(config)
    stages, replica_topology = expand_replica_stages(stages, plan)
    gpu_placement = build_stage_placement_plan(
        config,
        stages_cfg=stages,
        replica_instances=replica_topology.replicas,
    )
    return build_process_topology_plan(config, gpu_placement, stages_cfg=stages)


def test_stage_process_parses_from_schema_and_dotted_overrides() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="old0", next_stage="b"),
            _stage("b", process="old1", terminal=True),
        ],
    )

    merged = ConfigManager(config).merge_config({"a.process": "p0", "b.process": "p1"})

    assert [stage.process for stage in merged.stages] == ["p0", "p1"]
    # The source config is not mutated: the resolver rebuilds.
    assert [stage.process for stage in config.stages] == ["old0", "old1"]


def test_non_tp_stages_must_declare_process() -> None:
    with pytest.raises(ValueError, match="Non-TP stages must declare process"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage("a", process="p0", next_stage="b"),
                _stage("b", terminal=True),
            ],
        )


def test_missing_non_tp_process_declaration_is_rejected() -> None:
    with pytest.raises(ValueError, match="Non-TP stages must declare process"):
        PipelineConfig(
            model_path="dummy",
            stages=[_stage("a", next_stage="b"), _stage("b", terminal=True)],
        )


def test_tp_process_names_are_derived_when_process_is_missing() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[_stage("thinker", gpu=[0, 1], tp_size=2, terminal=True)],
    )

    topology = _topology(config)

    assert topology.groups == ()
    assert topology.tp_stage_to_processes == {"thinker": ("thinker_tp0", "thinker_tp1")}


def test_tp_process_field_is_used_as_rank_process_prefix() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage(
                "thinker",
                gpu=[0, 1],
                tp_size=2,
                process="model",
                terminal=True,
            )
        ],
    )

    topology = _topology(config)

    assert topology.tp_stage_to_processes == {"thinker": ("model_tp0", "model_tp1")}


def test_same_process_same_gpu_does_not_require_memory_budgets() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, process="p0", next_stage="b"),
            _stage("b", gpu=0, process="p0", terminal=True),
        ],
    )

    topology = _topology(config)

    assert [
        (group.name, group.stage_names, group.gpu_id) for group in topology.groups
    ] == [("p0", ("a", "b"), 0)]


def test_same_gpu_multiple_processes_accepts_explicit_budgets() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, fraction=0.20, process="p0", next_stage="b"),
            _stage("b", gpu=0, fraction=0.30, process="p0", next_stage="c"),
            _stage("c", gpu=0, fraction=0.40, process="p1", terminal=True),
        ],
    )

    topology = _topology(config)

    assert [
        (group.name, group.stage_names, group.gpu_id) for group in topology.groups
    ] == [
        ("p0", ("a", "b"), 0),
        ("p1", ("c",), 0),
    ]


def test_same_gpu_multiple_processes_rejects_missing_budget() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, fraction=0.20, process="p0", next_stage="b"),
            _stage("b", gpu=0, process="p1", terminal=True),
        ],
    )
    with pytest.raises(ValueError, match="gpu_memory_fraction"):
        _topology(config)


def test_same_gpu_multiple_processes_rejects_over_budget() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, fraction=0.70, process="p0", next_stage="b"),
            _stage("b", gpu=0, fraction=0.40, process="p1", terminal=True),
        ],
    )

    with pytest.raises(ValueError, match="exceeds placement limit"):
        build_stage_placement_plan(config)


def test_one_process_group_cannot_span_multiple_gpus() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", gpu=0, process="p0", next_stage="b"),
            _stage("b", gpu=1, process="p0", terminal=True),
        ],
    )
    with pytest.raises(ValueError, match="spans multiple GPUs"):
        _topology(config)


def test_tp_process_names_must_not_collide_with_non_tp_process_group() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage("a", process="thinker_tp0", next_stage="thinker"),
            _stage("thinker", gpu=[0, 1], tp_size=2, terminal=True),
        ],
    )
    with pytest.raises(ValueError, match="collide"):
        _topology(config)


def test_tp_process_names_must_be_unique_across_tp_stages() -> None:
    with pytest.raises(ValueError, match="claimed by multiple TP stages"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage("a", gpu=[0, 1], tp_size=2, process="model", next_stage="b"),
                _stage("b", gpu=[2, 3], tp_size=2, process="model", terminal=True),
            ],
        )


def _byte_budget_stage(
    name: str,
    *,
    process: str,
    kv_cache_bytes: int | None,
    total_reserve_bytes: int | None,
    next_stage: str | None = None,
    terminal: bool = False,
) -> StageConfig:
    from sglang_omni.config import EngineArgs, EngineStageConfig

    return EngineStageConfig(
        name=name,
        factory_path=_FACTORY,
        gpu=0,
        process=process,
        total_reserve_bytes=total_reserve_bytes,
        engine=EngineArgs(kv_cache_bytes=kv_cache_bytes),
        next=next_stage,
        terminal=terminal,
    )


def test_colocation_accepts_byte_budgeted_stages_with_total_reserve() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _byte_budget_stage(
                "a",
                process="p1",
                kv_cache_bytes=1024**3,
                total_reserve_bytes=4 * 1024**3,
                next_stage="b",
            ),
            _byte_budget_stage(
                "b",
                process="p2",
                kv_cache_bytes=1024**3,
                total_reserve_bytes=4 * 1024**3,
                terminal=True,
            ),
        ],
    )

    _topology(config)


def test_colocation_rejects_kv_only_stage_without_total_reserve() -> None:
    """kv_cache_bytes covers the KV pool alone; sharing a GPU requires the
    stage's total footprint, so a byte-budgeted colocated stage must also
    declare total_reserve_bytes."""
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _byte_budget_stage(
                "a",
                process="p1",
                kv_cache_bytes=1024**3,
                total_reserve_bytes=None,
                next_stage="b",
            ),
            _byte_budget_stage(
                "b",
                process="p2",
                kv_cache_bytes=1024**3,
                total_reserve_bytes=4 * 1024**3,
                terminal=True,
            ),
        ],
    )

    with pytest.raises(ValueError, match="declared total footprint"):
        _topology(config)
