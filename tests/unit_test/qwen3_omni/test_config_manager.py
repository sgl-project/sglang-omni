# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.cli.serve import apply_encoder_mem_reserve_cli_override
from sglang_omni.config import (
    build_process_topology_plan,
    build_stage_placement_plan,
    resolve_stage_factory_args,
)
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechColocatedPipelineConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def test_config_manager_parses_dotted_fraction_overrides_as_numbers() -> None:
    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))
    extra_args = manager.parse_extra_args(
        [
            "--stages.1.runtime.resources.total-gpu-memory-fraction",
            "0.05",
            "--stages.2.runtime.resources.total-gpu-memory-fraction",
            "0.05",
            "--stages.4.runtime.resources.total-gpu-memory-fraction",
            "0.35",
            "--stages.4.runtime.sglang-server-args.mem-fraction-static",
            "0.35",
            "--stages.6.runtime.resources.total-gpu-memory-fraction",
            "0.35",
            "--stages.6.runtime.sglang-server-args.mem-fraction-static",
            "0.35",
            "--stages.7.runtime.resources.total-gpu-memory-fraction",
            "0.05",
        ]
    )

    merged = manager.merge_config(extra_args)
    plan = build_stage_placement_plan(merged)

    assert _stage(
        merged, "thinker"
    ).runtime.resources.total_gpu_memory_fraction == pytest.approx(0.35)
    assert _stage(
        merged, "thinker"
    ).runtime.sglang_server_args.mem_fraction_static == pytest.approx(0.35)
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.85)


def test_config_manager_dotted_tp_size_override_updates_parallelism_alias() -> None:
    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))
    merged = manager.merge_config({"stages.4.tp_size": 2, "stages.4.gpu": [0, 1]})
    thinker = _stage(merged, "thinker")

    assert thinker.tp_size == 2
    assert thinker.parallelism.tp == 2
    assert thinker.gpu == [0, 1]


def test_config_manager_dotted_parallelism_override_updates_tp_size_alias() -> None:
    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))
    merged = manager.merge_config(
        {"stages.4.parallelism.tp": 2, "stages.4.gpu": [0, 1]}
    )
    thinker = _stage(merged, "thinker")

    assert thinker.tp_size == 2
    assert thinker.parallelism.tp == 2
    assert thinker.gpu == [0, 1]


def test_config_manager_rejects_trailing_key_without_value() -> None:
    manager = ConfigManager(Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy"))

    with pytest.raises(ValueError, match="Missing value"):
        manager.parse_extra_args(
            [
                "--stages.4.runtime.resources.total-gpu-memory-fraction",
                "0.35",
                "--stages.4.runtime.sglang-server-args.mem-fraction-static",
            ]
        )


def test_qwen3_omni_h20_colocated_example_config_loads_and_plans(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_OMNI_QWEN3_FUSE_TALKER_CODE2WAV", raising=False)
    config_path = _REPO_ROOT / "examples" / "configs" / "qwen3_omni_colocated_h20.yaml"
    config_text = config_path.read_text()

    manager = ConfigManager.from_file(str(config_path))
    config = manager.config
    plan = build_stage_placement_plan(config)
    topology = build_process_topology_plan(config, plan)

    assert "stages:" not in config_text
    assert "factory:" not in config_text
    assert isinstance(config, Qwen3OmniSpeechColocatedPipelineConfig)
    assert config.name == "qwen3-omni-colocated-h20"
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
    assert (
        _stage(config, "thinker").runtime.sglang_server_args.mem_fraction_static is None
    )
    assert (
        _stage(config, "talker_ar").runtime.sglang_server_args.mem_fraction_static
        is None
    )
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


def test_qwen3_omni_h200_fused_example_config_loads_and_plans() -> None:
    config_path = (
        _REPO_ROOT / "examples" / "configs" / "qwen3_omni_colocated_h200_fused.yaml"
    )

    manager = ConfigManager.from_file(str(config_path))
    config = manager.config
    plan = build_stage_placement_plan(config)
    topology = build_process_topology_plan(config, plan)

    assert isinstance(config, Qwen3OmniSpeechColocatedPipelineConfig)
    assert config.name == "qwen3-omni-colocated-h200-fused"
    assert config.fused_stages == [["talker_ar", "code2wav"]]
    assert [group.name for group in topology.groups] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
        "fused_talker_ar_code2wav",
    ]


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


def test_qwen_preprocessing_runtime_video_fps_resolves_to_factory_arg() -> None:
    config = Qwen3OmniSpeechColocatedPipelineConfig(model_path="dummy")
    preprocessing = _stage(config, "preprocessing")
    preprocessing.runtime.video_fps = 2.0

    args = resolve_stage_factory_args(preprocessing, config)

    assert args["video_fps"] == 2.0


def test_h20_colocated_example_reserve_keeps_raw_budget_in_resolved_config() -> None:
    config_path = _REPO_ROOT / "examples" / "configs" / "qwen3_omni_colocated_h20.yaml"
    config = ConfigManager.from_file(str(config_path)).config

    apply_encoder_mem_reserve_cli_override(
        config,
        encoder_mem_reserve=0.05,
        mem_fraction_static=None,
        thinker_mem_fraction_static=None,
    )
    plan = build_stage_placement_plan(config)
    thinker = _stage(config, "thinker")
    thinker_args = resolve_stage_factory_args(thinker, config)

    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.94)
    assert thinker.runtime.resources.total_gpu_memory_fraction == pytest.approx(0.75)
    assert thinker_args["total_gpu_memory_fraction"] == pytest.approx(0.75)
    assert thinker_args["encoder_mem_reserve"] == pytest.approx(0.05)


def test_config_manager_rejects_unknown_stage_override(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_colocated.yaml"
    config_path.write_text(
        """
config_cls: Qwen3OmniSpeechColocatedPipelineConfig
model_path: dummy
stage_overrides:
  missing_stage:
    runtime:
      resources:
        total_gpu_memory_fraction: 0.05
"""
    )

    with pytest.raises(ValueError, match="unknown stage"):
        ConfigManager.from_file(str(config_path))


def test_config_manager_rejects_unsupported_stage_override_key(
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

    with pytest.raises(ValueError, match="supports only runtime"):
        ConfigManager.from_file(str(config_path))


def test_config_manager_rejects_non_mapping_stage_overrides(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "bad_colocated.yaml"
    config_path.write_text(
        """
config_cls: Qwen3OmniSpeechColocatedPipelineConfig
model_path: dummy
stage_overrides:
"""
    )

    with pytest.raises(ValueError, match="stage_overrides must be a mapping"):
        ConfigManager.from_file(str(config_path))


def test_config_manager_validates_stage_override_runtime_values(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "bad_colocated.yaml"
    config_path.write_text(
        """
config_cls: Qwen3OmniSpeechColocatedPipelineConfig
model_path: dummy
stage_overrides:
  image_encoder:
    runtime:
      resources:
        total_gpu_memory_fraction: 1.5
"""
    )

    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        ConfigManager.from_file(str(config_path))
