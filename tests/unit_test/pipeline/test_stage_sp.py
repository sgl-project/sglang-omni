# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.config import (
    ParallelismConfig,
    PipelineConfig,
    SequenceParallelPolicy,
    StageConfig,
    build_process_topology_plan,
    build_stage_placement_plan,
    resolve_stage_gpu_ids,
)
from sglang_omni.config.process_overrides import apply_stage_process_overrides

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


class _SequenceParallelPipelineConfig(PipelineConfig):
    @classmethod
    def sequence_parallel_policy(
        cls, *, stage_name: str
    ) -> SequenceParallelPolicy | None:
        if stage_name == "decode":
            return SequenceParallelPolicy(
                attention_heads=6,
                requires_power_of_two=True,
            )
        return None


def _stage(
    *,
    parallelism: ParallelismConfig | dict[str, int] | None = None,
    gpu: int | list[int] | None = None,
    process: str | None = None,
) -> StageConfig:
    kwargs = {}
    if parallelism is not None:
        kwargs["parallelism"] = parallelism
    return StageConfig(
        name="decode",
        factory=_FACTORY,
        gpu=gpu,
        process=process,
        terminal=True,
        **kwargs,
    )


def test_stage_schema_accepts_sequence_parallel_decomposition() -> None:
    stage = _stage(
        gpu=[0, 1, 2, 3],
        parallelism={"sp": 4, "ulysses_degree": 2, "ring_degree": 2},
    )

    assert stage.tp_size == 1
    assert stage.parallelism.sp == 4
    assert stage.parallelism.ulysses_degree == 2
    assert stage.parallelism.ring_degree == 2


def test_stage_schema_rejects_combined_tp_and_sp() -> None:
    with pytest.raises(ValueError, match="cannot enable TP and SP"):
        _stage(
            gpu=[0, 1, 2, 3],
            parallelism={"tp": 2, "sp": 2},
        )


def test_pipeline_rejects_sp_without_stage_policy() -> None:
    with pytest.raises(ValueError, match="does not support sequence parallelism"):
        PipelineConfig(
            model_path="dummy",
            stages=[_stage(gpu=[0, 1], parallelism={"sp": 2})],
        )


@pytest.mark.parametrize(
    ("parallelism", "message"),
    [
        ({"sp": 3, "ulysses_degree": 3}, "power of two"),
        ({"sp": 4, "ulysses_degree": 4}, "must divide.*6 attention heads"),
        (
            {"sp": 4, "ulysses_degree": 2, "ring_degree": 1},
            "sp must equal ulysses_degree.*ring_degree",
        ),
    ],
)
def test_pipeline_applies_generic_sequence_parallel_policy(
    parallelism: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _SequenceParallelPipelineConfig(
            model_path="dummy",
            stages=[_stage(gpu=[0, 1, 2, 3], parallelism=parallelism)],
        )


def test_sp_placement_preserves_one_gpu_per_rank() -> None:
    stage = _stage(
        gpu=[0, 1, 2, 3],
        parallelism={"sp": 4, "ulysses_degree": 2, "ring_degree": 2},
    )
    config = _SequenceParallelPipelineConfig(
        model_path="dummy",
        stages=[stage],
    )

    plan = build_stage_placement_plan(config)

    assert resolve_stage_gpu_ids(plan, stage) == [0, 1, 2, 3]
    assert plan.stages["decode"].sp_size == 4
    assert plan.stages["decode"].world_size == 4


@pytest.mark.parametrize(
    ("gpu", "message"),
    [
        (0, "one GPU id per SP rank"),
        ([0], "gpu has 1 entries but parallelism.sp=2"),
        ([0, 0], "SP placement requires unique GPU ids"),
    ],
)
def test_sp_placement_requires_one_unique_gpu_per_rank(
    gpu: int | list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config = _SequenceParallelPipelineConfig(
            model_path="dummy",
            stages=[_stage(gpu=gpu, parallelism={"sp": 2})],
        )
        build_stage_placement_plan(config)


def test_sp_topology_assigns_one_process_name_per_rank() -> None:
    stage = _stage(
        gpu=[0, 1],
        process="worker",
        parallelism={"sp": 2, "ulysses_degree": 2},
    )
    config = _SequenceParallelPipelineConfig(
        model_path="dummy",
        stages=[stage],
    )
    placement = build_stage_placement_plan(config)

    topology = build_process_topology_plan(config, placement)

    assert topology.groups == ()
    assert topology.sp_stage_to_processes == {"decode": ("worker_sp0", "worker_sp1")}
    assert topology.tp_stage_to_processes == {}


def test_non_sp_stage_keeps_single_process_contract() -> None:
    stage = _stage(gpu=0, process="worker")
    config = PipelineConfig(model_path="dummy", stages=[stage])
    placement = build_stage_placement_plan(config)

    topology = build_process_topology_plan(config, placement)

    assert topology.sp_stage_to_processes == {}
    assert topology.stage_to_process == {"decode": "worker"}


def test_process_override_rejects_sequence_parallel_stage() -> None:
    stage = _stage(gpu=[0, 1], parallelism={"sp": 2})
    config = _SequenceParallelPipelineConfig(model_path="dummy", stages=[stage])

    with pytest.raises(ValueError, match="one process per SP rank"):
        apply_stage_process_overrides(config, isolate_stages=["decode"])
