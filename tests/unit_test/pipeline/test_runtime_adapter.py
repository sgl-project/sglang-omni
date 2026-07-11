# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.config import (
    PipelineConfig,
    SGLangServerArgsConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
    resolve_stage_factory_args,
)
from sglang_omni.config.runtime import resolve_stage_static_factory_args
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.runtime_factory"
_FACTORY_WITHOUT_TOTAL_BUDGET = (
    "tests.unit_test.fixtures.pipeline_fakes.runtime_factory_without_total_budget"
)


def _stage(**kwargs) -> StageConfig:
    data = {
        "name": "thinker",
        "process": "pipeline",
        "factory": _FACTORY,
        "terminal": True,
        "gpu": 1,
    }
    data.update(kwargs)
    return StageConfig(**data)


def test_typed_runtime_maps_to_factory_args_and_sglang_overrides() -> None:
    stage = _stage(
        runtime=StageRuntimeConfig(
            resources=StageResourceConfig(total_gpu_memory_fraction=0.25),
            max_seq_len=32768,
            video_fps=2.0,
            sglang_server_args=SGLangServerArgsConfig(mem_fraction_static=0.72),
        ),
        runtime_arg_map={
            "max_seq_len": "thinker_max_seq_len",
            "video_fps": "video_fps",
        },
    )
    config = PipelineConfig(
        model_path="dummy-model",
        stages=[stage],
        runtime_overrides={
            "thinker": {"server_args_overrides": {"disable_cuda_graph": True}}
        },
    )

    args = resolve_stage_factory_args(stage, config)

    assert args["model_path"] == "dummy-model"
    assert args["gpu_id"] == 1
    assert args["thinker_max_seq_len"] == 32768
    assert args["video_fps"] == 2.0
    assert args["server_args_overrides"] == {
        "disable_cuda_graph": True,
        "mem_fraction_static": 0.72,
    }
    assert args["total_gpu_memory_fraction"] == 0.25


def test_total_gpu_memory_fraction_is_not_injected_into_unrelated_factories() -> None:
    stage = _stage(
        factory=_FACTORY_WITHOUT_TOTAL_BUDGET,
        runtime=StageRuntimeConfig(
            resources=StageResourceConfig(total_gpu_memory_fraction=0.25),
            sglang_server_args=SGLangServerArgsConfig(mem_fraction_static=0.72),
        ),
    )
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    args = resolve_stage_factory_args(stage, config)

    assert "total_gpu_memory_fraction" not in args
    assert args["server_args_overrides"] == {"mem_fraction_static": 0.72}


def test_typed_sglang_runtime_rejects_compat_mem_fraction_duplicate() -> None:
    stage = _stage(
        runtime=StageRuntimeConfig(
            sglang_server_args=SGLangServerArgsConfig(mem_fraction_static=0.70)
        )
    )
    config = PipelineConfig(
        model_path="dummy-model",
        stages=[stage],
        runtime_overrides={
            "thinker": {
                "server_args_overrides": {
                    "disable_cuda_graph": True,
                    "mem_fraction_static": 0.85,
                }
            }
        },
    )

    with pytest.raises(ValueError, match="mem_fraction_static"):
        resolve_stage_factory_args(stage, config)


def test_typed_sglang_runtime_rejects_factory_mem_fraction_duplicate() -> None:
    stage = _stage(
        factory_args={"server_args_overrides": {"mem_fraction_static": 0.85}},
        runtime=StageRuntimeConfig(
            sglang_server_args=SGLangServerArgsConfig(mem_fraction_static=0.70)
        ),
    )
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    with pytest.raises(ValueError, match="mem_fraction_static"):
        resolve_stage_factory_args(stage, config)


def test_typed_total_memory_fraction_rejects_override_duplicate() -> None:
    stage = _stage(
        runtime=StageRuntimeConfig(
            resources=StageResourceConfig(total_gpu_memory_fraction=0.25)
        )
    )
    config = PipelineConfig(
        model_path="dummy-model",
        stages=[stage],
        runtime_overrides={"thinker": {"total_gpu_memory_fraction": 0.40}},
    )

    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        resolve_stage_factory_args(stage, config)


def test_typed_total_memory_fraction_rejects_factory_duplicate() -> None:
    stage = _stage(
        factory_args={"total_gpu_memory_fraction": 0.40},
        runtime=StageRuntimeConfig(
            resources=StageResourceConfig(total_gpu_memory_fraction=0.25)
        ),
    )
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        resolve_stage_factory_args(stage, config)


def test_untyped_total_memory_fraction_factory_arg_is_rejected() -> None:
    stage = _stage(factory_args={"total_gpu_memory_fraction": 0.40})
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    with pytest.raises(ValueError, match="runtime.resources.total_gpu_memory_fraction"):
        resolve_stage_factory_args(stage, config)


def test_untyped_total_memory_fraction_runtime_override_is_rejected() -> None:
    stage = _stage()
    config = PipelineConfig(
        model_path="dummy-model",
        stages=[stage],
        runtime_overrides={"thinker": {"total_gpu_memory_fraction": 0.40}},
    )

    with pytest.raises(ValueError, match="runtime.resources.total_gpu_memory_fraction"):
        resolve_stage_factory_args(stage, config)


def test_gpu_id_factory_arg_is_rejected() -> None:
    stage = _stage(factory_args={"gpu_id": 0})
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    with pytest.raises(ValueError, match="owned by placement"):
        resolve_stage_static_factory_args(stage, config)


def test_gpu_id_runtime_override_is_rejected() -> None:
    stage = _stage()
    config = PipelineConfig(
        model_path="dummy-model",
        stages=[stage],
        runtime_overrides={"thinker": {"gpu_id": 0}},
    )

    with pytest.raises(ValueError, match="owned by placement"):
        resolve_stage_static_factory_args(stage, config)


def test_mapped_runtime_field_requires_stage_arg_mapping() -> None:
    stage = _stage(runtime=StageRuntimeConfig(max_seq_len=8192))
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    with pytest.raises(ValueError, match="runtime_arg_map"):
        resolve_stage_factory_args(stage, config)


def test_rank_gpu_id_can_be_supplied_by_launch_planner() -> None:
    stage = _stage(gpu=0)
    config = PipelineConfig(model_path="dummy-model", stages=[stage])

    args = resolve_stage_factory_args(stage, config, gpu_id=3)

    assert args["gpu_id"] == 3


def test_runtime_override_wins_over_qwen_model_default() -> None:
    config = Qwen3OmniSpeechPipelineConfig(
        model_path="dummy-model",
        runtime_overrides={"thinker": {"thinker_max_seq_len": 16384}},
    )
    thinker = next(stage for stage in config.stages if stage.name == "thinker")

    args = resolve_stage_factory_args(thinker, config)

    assert args["thinker_max_seq_len"] == 16384


def test_explicit_typed_runtime_rejects_runtime_override_duplicate() -> None:
    config = Qwen3OmniSpeechPipelineConfig(
        model_path="dummy-model",
        runtime_overrides={"thinker": {"thinker_max_seq_len": 16384}},
    )
    thinker = next(stage for stage in config.stages if stage.name == "thinker")
    thinker.runtime.max_seq_len = 32768

    with pytest.raises(ValueError, match="thinker_max_seq_len"):
        resolve_stage_factory_args(thinker, config)


def test_typed_runtime_can_override_static_model_factory_default() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy-model")
    thinker = next(stage for stage in config.stages if stage.name == "thinker")
    assert thinker.factory_args["thinker_max_seq_len"] == 8192
    thinker.runtime.max_seq_len = 32768

    args = resolve_stage_factory_args(thinker, config)

    assert args["thinker_max_seq_len"] == 32768
