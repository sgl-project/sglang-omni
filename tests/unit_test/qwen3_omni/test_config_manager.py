# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.config import build_stage_placement_plan, resolve_stage_factory_args
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni import config as qwen3_omni_config
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from tests.unit_test.pipeline.helpers import build_compiled_process_topology

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def test_config_manager_parses_dotted_fraction_overrides_as_numbers() -> None:
    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))
    extra_args = manager.parse_extra_args(
        [
            "--image_encoder.gpu_memory_fraction",
            "0.05",
            "--audio_encoder.gpu_memory_fraction",
            "0.05",
            "--thinker.gpu_memory_fraction",
            "0.35",
            "--thinker.engine.mem_fraction_static",
            "0.35",
            "--talker_ar.gpu_memory_fraction",
            "0.35",
            "--talker_ar.engine.mem_fraction_static",
            "0.35",
            "--code2wav.gpu_memory_fraction",
            "0.05",
        ]
    )

    merged = manager.merge_config(extra_args)
    plan = build_stage_placement_plan(merged)

    assert _stage(merged, "thinker").gpu_memory_fraction == pytest.approx(0.35)
    assert _stage(merged, "thinker").engine.mem_fraction_static == pytest.approx(0.35)
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.85)


def test_config_manager_applies_dotted_tp_size_override() -> None:
    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))
    merged = manager.merge_config({"thinker.tp_size": 2, "thinker.gpu": [0, 1]})
    thinker = _stage(merged, "thinker")

    assert thinker.tp_size == 2
    assert thinker.gpu == [0, 1]


def test_config_manager_sets_tp_size_directly() -> None:
    """tp_size is the only spelling; the parallelism.tp mirror is gone."""
    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))
    merged = manager.merge_config({"thinker.tp_size": 2, "thinker.gpu": [0, 1]})
    thinker = _stage(merged, "thinker")

    assert thinker.tp_size == 2
    assert thinker.gpu == [0, 1]


def test_config_manager_rejects_trailing_key_without_value() -> None:
    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))

    with pytest.raises(ValueError, match="Missing value"):
        manager.parse_extra_args(
            [
                "--thinker.gpu_memory_fraction",
                "0.35",
                "--thinker.engine.mem-fraction-static",
            ]
        )


def test_qwen3_omni_h20_colocated_example_config_loads_and_plans() -> None:
    config_path = _REPO_ROOT / "examples" / "configs" / "qwen3_omni_colocated_h20.yaml"

    manager = ConfigManager.from_file(str(config_path))
    config = manager.config
    plan = build_stage_placement_plan(config)
    topology = build_compiled_process_topology(config)

    assert isinstance(config, Qwen3OmniSpeechColocatedPipelineConfig)
    assert config.name == "qwen3-omni-colocated-h20"
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.94)
    assert [group.name for group in topology.groups] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "thinker",
        "decode",
        "talker_ar",
        "code2wav",
    ]
    assert _stage(config, "thinker").engine.mem_fraction_static is None
    assert _stage(config, "talker_ar").engine.mem_fraction_static is None
    assert {
        stage.name: stage.gpu
        for stage in config.stages
        if stage.name
        in {
            "image_encoder",
            "audio_encoder",
            "thinker",
            "talker_ar",
            "code2wav",
        }
    } == {
        "image_encoder": 0,
        "audio_encoder": 0,
        "thinker": 0,
        "talker_ar": 0,
        "code2wav": 0,
    }


def test_qwen3_omni_mmsu_example_config_uses_text_pipeline() -> None:
    config_path = _REPO_ROOT / "examples" / "configs" / "qwen3_omni_mmsu.yaml"

    manager = ConfigManager.from_file(str(config_path))
    config = manager.config
    plan = build_stage_placement_plan(config)
    thinker_args = resolve_stage_factory_args(_stage(config, "thinker"), config)

    assert isinstance(config, Qwen3OmniPipelineConfig)
    assert config.name == "qwen3-omni-mmsu"
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
    ]
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert "talker_ar" not in {stage.name for stage in config.stages}
    assert "code2wav" not in {stage.name for stage in config.stages}
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.8)
    assert thinker_args["total_gpu_memory_fraction"] == pytest.approx(0.75)
    assert thinker_args["server_args_overrides"]["max_running_requests"] == 4


def test_qwen_preprocessing_model_video_fps_resolves_to_factory_arg() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    merged = ConfigManager(config).merge_config(
        [("preprocessing.factory.video_fps", "2.0")]
    )

    args = resolve_stage_factory_args(_stage(merged, "preprocessing"), merged)

    assert args["video_fps"] == 2.0


def test_h20_colocated_example_reserve_keeps_raw_budget_in_resolved_config() -> None:
    config_path = _REPO_ROOT / "examples" / "configs" / "qwen3_omni_colocated_h20.yaml"
    config = ConfigManager.from_file(str(config_path)).config

    merged = ConfigManager(config).merge_config(
        [("thinker.factory.encoder_mem_reserve", "0.05")]
    )
    plan = build_stage_placement_plan(merged)
    thinker = _stage(merged, "thinker")
    thinker_args = resolve_stage_factory_args(thinker, merged)

    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.94)
    assert thinker.gpu_memory_fraction == pytest.approx(0.75)
    assert thinker_args["total_gpu_memory_fraction"] == pytest.approx(0.75)
    assert thinker_args["encoder_mem_reserve"] == pytest.approx(0.05)


def test_config_manager_rejects_unknown_stage_entry(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_colocated.yaml"
    config_path.write_text(
        """
config_cls: Qwen3OmniSpeechColocatedPipelineConfig
model_path: dummy
stages:
  missing_stage:
    gpu_memory_fraction: 0.05
"""
    )

    # Stage topology lives in the model's config class; an unknown name in
    # the stages: mapping is refused, not created.
    with pytest.raises(Exception, match="no stage named"):
        ConfigManager.from_file(str(config_path))


def test_config_manager_rejects_removed_stage_overrides_block(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "bad_colocated.yaml"
    config_path.write_text(
        """
config_cls: Qwen3OmniSpeechColocatedPipelineConfig
model_path: dummy
stage_overrides:
  thinker:
    gpu: 0
"""
    )

    with pytest.raises(ValueError, match="stages: mapping"):
        ConfigManager.from_file(str(config_path))


def test_config_manager_validates_stage_entry_values(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "bad_colocated.yaml"
    config_path.write_text(
        """
config_cls: Qwen3OmniSpeechColocatedPipelineConfig
model_path: dummy
stages:
  image_encoder:
    gpu_memory_fraction: 1.5
"""
    )

    with pytest.raises(ValueError, match="gpu_memory_fraction"):
        ConfigManager.from_file(str(config_path))


def test_qwen3_omni_h100_bf16_config_enables_speech_prefill_graph() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    config_path = (
        repo_root / "examples" / "configs" / "qwen3_omni_colocated_h100_bf16.yaml"
    )

    config = ConfigManager.from_file(str(config_path)).config
    overrides = _stage(config, "thinker").engine.overrides()

    assert isinstance(config, Qwen3OmniSpeechColocatedPipelineConfig)
    assert "disable_radix_cache" not in overrides
    assert overrides["cuda_graph_backend_prefill"] == "breakable"
    assert "cuda_graph_bs_prefill" not in overrides
    assert overrides["cuda_graph_max_bs_prefill"] == 2048


def test_qwen3_omni_gfx950_bf16_config_uses_colocated_budgets() -> None:
    config_path = (
        _REPO_ROOT / "examples" / "configs" / "qwen3_omni_colocated_gfx950_bf16.yaml"
    )

    config = ConfigManager.from_file(str(config_path)).config
    plan = build_stage_placement_plan(config)
    overrides = _stage(config, "thinker").engine.overrides()

    assert isinstance(config, Qwen3OmniSpeechColocatedPipelineConfig)
    assert config.name == "qwen3-omni-colocated-gfx950-bf16"
    assert "prefill_attention_backend" not in overrides
    assert overrides["cuda_graph_backend_prefill"] == "disabled"
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.94)
    assert {
        name: _stage(config, name).gpu_memory_fraction
        for name in (
            "image_encoder",
            "audio_encoder",
            "thinker",
            "talker_ar",
            "code2wav",
        )
    } == {
        "image_encoder": pytest.approx(0.02),
        "audio_encoder": pytest.approx(0.02),
        "thinker": pytest.approx(0.78),
        "talker_ar": pytest.approx(0.10),
        "code2wav": pytest.approx(0.02),
    }


@pytest.mark.parametrize(
    ("is_rocm", "expected_env"),
    [
        (True, {"SGLANG_DISABLE_AITER_GREEDY_SAMPLE": "1"}),
        (False, {}),
    ],
)
def test_qwen3_omni_talker_stage_keeps_greedy_selection_off_aiter_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
    is_rocm: bool,
    expected_env: dict[str, str],
) -> None:
    """aiter's greedy_sample is wrong below 16384 vocab entries; the Talker
    codec head has 3072, so the ROCm Talker process falls back to torch.argmax."""
    monkeypatch.setattr(qwen3_omni_config.current_platform, "is_rocm", lambda: is_rocm)

    for config_cls in (
        Qwen3OmniSpeechPipelineConfig,
        Qwen3OmniSpeechColocatedPipelineConfig,
    ):
        config = config_cls(model_path="dummy")

        assert _stage(config, "talker_ar").env == expected_env
        assert _stage(config, "thinker").env == {}
