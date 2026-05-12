# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni_v1.config import (
    PipelineConfig,
    SGLangServerArgsConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
    resolve_stage_factory_args,
)
from sglang_omni_v1.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.runtime_factory"


def _stage(**kwargs) -> StageConfig:
    data = {
        "name": "thinker",
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
    assert "total_gpu_memory_fraction" not in args


def test_typed_sglang_runtime_rejects_legacy_mem_fraction_duplicate() -> None:
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


def test_legacy_runtime_override_wins_over_qwen_model_default() -> None:
    config = Qwen3OmniSpeechPipelineConfig(
        model_path="dummy-model",
        runtime_overrides={"thinker": {"thinker_max_seq_len": 16384}},
    )
    thinker = next(stage for stage in config.stages if stage.name == "thinker")

    args = resolve_stage_factory_args(thinker, config)

    assert args["thinker_max_seq_len"] == 16384


def test_explicit_typed_runtime_rejects_legacy_runtime_override_duplicate() -> None:
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
