# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.config import (
    ParallelismConfig,
    PipelineConfig,
    PlacementConfig,
    SGLangServerArgsConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
    TensorRefEdgeConfig,
)

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


def _stage(**kwargs) -> StageConfig:
    data = {
        "name": "stage",
        "process": "pipeline",
        "factory": _FACTORY,
        "terminal": True,
    }
    data.update(kwargs)
    return StageConfig(**data)


def test_stage_runtime_schema_accepts_typed_runtime_values() -> None:
    stage = _stage(
        runtime=StageRuntimeConfig(
            resources=StageResourceConfig(total_gpu_memory_fraction=0.25),
            max_seq_len=8192,
            video_fps=2.0,
            sglang_server_args=SGLangServerArgsConfig(mem_fraction_static=0.7),
        ),
        runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
    )

    assert stage.runtime.resources.total_gpu_memory_fraction == 0.25
    assert stage.runtime.sglang_server_args.mem_fraction_static == 0.7
    assert stage.runtime_arg_map["max_seq_len"] == "thinker_max_seq_len"


def test_invalid_total_gpu_memory_fraction_raises() -> None:
    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        StageResourceConfig(total_gpu_memory_fraction=0.0)


def test_invalid_sglang_mem_fraction_static_raises() -> None:
    with pytest.raises(ValueError, match="mem_fraction_static"):
        SGLangServerArgsConfig(mem_fraction_static=1.0)


def test_invalid_stage_runtime_values_raise() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        StageRuntimeConfig(max_seq_len=0)
    with pytest.raises(ValueError, match="video_fps"):
        StageRuntimeConfig(video_fps=-1.0)


def test_stage_rejects_terminal_with_next() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage(name="source", next="sink", terminal=True),
                _stage(name="sink"),
            ],
        )


def test_tp_size_normalizes_into_parallelism_tp() -> None:
    stage = _stage(tp_size=2, gpu=[0, 1])

    assert stage.tp_size == 2
    assert stage.parallelism.tp == 2


def test_parallelism_tp_normalizes_back_to_tp_size() -> None:
    stage = _stage(parallelism=ParallelismConfig(tp=2), gpu=[0, 1])

    assert stage.tp_size == 2
    assert stage.parallelism.tp == 2


def test_conflicting_tp_size_and_parallelism_tp_raise() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        _stage(tp_size=2, parallelism=ParallelismConfig(tp=3), gpu=[0, 1])


def test_pipeline_accepts_placement_config() -> None:
    config = PipelineConfig(
        model_path="dummy",
        placement=PlacementConfig(max_total_gpu_memory_fraction_per_gpu=0.95),
        stages=[_stage()],
    )

    assert config.placement.max_total_gpu_memory_fraction_per_gpu == 0.95


def test_invalid_placement_limit_raises() -> None:
    with pytest.raises(ValueError, match="max_total_gpu_memory_fraction_per_gpu"):
        PlacementConfig(max_total_gpu_memory_fraction_per_gpu=1.1)


def test_tensor_ref_edge_must_be_declared_route_target() -> None:
    with pytest.raises(ValueError, match="not a declared next or stream_to target"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage(
                    name="source",
                    next="sink",
                    terminal=False,
                    tensor_ref_edges={
                        "consumer": TensorRefEdgeConfig(
                            consumer_stage="consumer",
                            paths=("embeds",),
                        )
                    },
                ),
                _stage(name="sink"),
                _stage(name="consumer"),
            ],
        )


def test_tensor_ref_edge_allows_direct_forwarder_and_later_consumer() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _stage(
                name="source",
                next="forwarder",
                terminal=False,
                tensor_ref_edges={
                    "forwarder": TensorRefEdgeConfig(
                        consumer_stage="consumer",
                        paths=("embeds",),
                    )
                },
            ),
            _stage(name="forwarder", next="consumer", terminal=False),
            _stage(name="consumer"),
        ],
    )

    assert config.stages[0].tensor_ref_edges["forwarder"].consumer_stage == "consumer"


def test_tensor_ref_edge_target_must_exist() -> None:
    with pytest.raises(ValueError, match="references unknown target stage"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage(
                    name="source",
                    next="sink",
                    terminal=False,
                    tensor_ref_edges={
                        "missing": TensorRefEdgeConfig(
                            consumer_stage="sink",
                            paths=("embeds",),
                        )
                    },
                ),
                _stage(name="sink"),
            ],
        )


def test_tensor_ref_edge_consumer_must_exist() -> None:
    with pytest.raises(ValueError, match="unknown consumer_stage"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                _stage(
                    name="source",
                    next="sink",
                    terminal=False,
                    tensor_ref_edges={
                        "sink": TensorRefEdgeConfig(
                            consumer_stage="missing",
                            paths=("embeds",),
                        )
                    },
                ),
                _stage(name="sink"),
            ],
        )
