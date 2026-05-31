# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.config import build_process_topology_plan, build_stage_placement_plan
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
    Variants,
)


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _set_colocated_runtime(
    config: Qwen3OmniSpeechColocatedPipelineConfig,
    *,
    include_mem_fraction: bool = True,
    conflicting_mem_fraction: bool = False,
) -> None:
    fractions = {
        "image_encoder": 0.025,
        "audio_encoder": 0.025,
        "thinker": 0.75,
        "talker_ar": 0.12,
        "code2wav": 0.02,
    }
    for stage_name, fraction in fractions.items():
        _stage(config, stage_name).runtime.resources.total_gpu_memory_fraction = (
            fraction
        )
    if include_mem_fraction:
        _stage(config, "thinker").runtime.sglang_server_args.mem_fraction_static = (
            0.74 if conflicting_mem_fraction else 0.75
        )
        _stage(config, "talker_ar").runtime.sglang_server_args.mem_fraction_static = (
            0.11 if conflicting_mem_fraction else 0.12
        )


def test_default_speech_topology_stays_disaggregated() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    assert len(config.stages) == 8
    assert _stage(config, "thinker").gpu == 0
    assert _stage(config, "talker_ar").gpu == 1
    assert _stage(config, "code2wav").gpu == 1
    assert _stage(config, "talker_ar").factory_args["enable_partial_start"] is True
    assert config.placement.require_memory_fraction_for_colocation is False
    assert {stage.name: stage.process for stage in config.stages} == {
        "preprocessing": "preprocessing",
        "image_encoder": "image_encoder",
        "audio_encoder": "audio_encoder",
        "mm_aggregate": "mm_aggregate",
        "thinker": "thinker",
        "decode": "decode",
        "talker_ar": "talker_ar",
        "code2wav": "code2wav",
    }
    assert "code_predictor" not in {stage.name for stage in config.stages}

    plan = build_stage_placement_plan(config)
    topology = build_process_topology_plan(config, plan)

    assert [group.name for group in topology.groups] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
        "talker_ar",
        "code2wav",
    ]


def test_colocated_topology_is_opt_in_and_uses_one_gpu() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")

    assert Variants["speech-colocated"] is Qwen3OmniSpeechColocatedPipelineConfig
    assert _stage(config, "talker_ar").factory_args["enable_partial_start"] is False
    for stage_name in (
        "image_encoder",
        "audio_encoder",
        "thinker",
        "talker_ar",
        "code2wav",
    ):
        assert _stage(config, stage_name).gpu == 0
        assert _stage(config, stage_name).process == stage_name


def test_colocated_config_passes_with_explicit_budgets_without_ar_mem_fraction() -> (
    None
):
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config, include_mem_fraction=False)

    plan = build_stage_placement_plan(config)
    topology = build_process_topology_plan(config, plan)

    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.94)
    assert [group.name for group in topology.groups] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
        "talker_ar",
        "code2wav",
    ]


def test_colocated_config_marks_same_gpu_stream_targets() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config)

    plan = build_stage_placement_plan(config)

    assert plan.same_gpu_stream_targets["thinker"] == frozenset({"talker_ar"})
    assert plan.same_gpu_stream_targets["talker_ar"] == frozenset({"code2wav"})


def test_default_speech_marks_only_talker_to_code2wav_same_gpu_stream() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    plan = build_stage_placement_plan(config)

    assert "thinker" not in plan.same_gpu_stream_targets
    assert plan.same_gpu_stream_targets["talker_ar"] == frozenset({"code2wav"})


def test_colocated_config_rejects_conflicting_ar_mem_fraction() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config, conflicting_mem_fraction=True)

    with pytest.raises(ValueError, match="conflicting memory fractions"):
        build_stage_placement_plan(config)


def test_colocated_config_rejects_missing_stage_budgets() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")

    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        build_stage_placement_plan(config)


@pytest.mark.parametrize("stage_name", ["talker_ar", "code2wav"])
def test_colocated_config_rejects_moving_gpu_stage_away(stage_name: str) -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config)
    _stage(config, stage_name).gpu = 1

    with pytest.raises(ValueError, match="share one GPU"):
        build_stage_placement_plan(config)


def test_colocated_config_rejects_topology_override_before_runtime_validation() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config)
    _stage(config, "talker_ar").gpu = 1

    with pytest.raises(ValueError, match="share one GPU"):
        build_stage_placement_plan(config)


def test_default_speech_rejects_same_gpu_thinker_and_talker_colocation() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    _stage(config, "talker_ar").gpu = 0
    _stage(config, "code2wav").gpu = 0
    for stage_name in (
        "image_encoder",
        "audio_encoder",
        "thinker",
        "talker_ar",
        "code2wav",
    ):
        _stage(config, stage_name).runtime.resources.total_gpu_memory_fraction = 0.10

    with pytest.raises(ValueError, match="Qwen3OmniSpeechColocatedPipelineConfig"):
        build_stage_placement_plan(config)


def test_default_speech_allows_thinker_tp_placement() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    thinker = _stage(config, "thinker")
    thinker.tp_size = 2
    thinker.parallelism.tp = 2
    thinker.gpu = [0, 1]

    plan = build_stage_placement_plan(config)

    assert plan.stages["thinker"].gpu_ids == (0, 1)


def test_colocated_config_rejects_thinker_tp() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    _set_colocated_runtime(config)
    thinker = _stage(config, "thinker")
    thinker.tp_size = 2
    thinker.parallelism.tp = 2
    thinker.gpu = [0, 1]

    with pytest.raises(ValueError, match="thinker TP"):
        build_stage_placement_plan(config)
