# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni_v1.config import build_stage_placement_plan
from sglang_omni_v1.config.manager import ConfigManager
from sglang_omni_v1.models.qwen3_omni.config import (
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
            "0.70",
            "--stages.6.runtime.resources.total-gpu-memory-fraction",
            "0.35",
            "--stages.6.runtime.sglang-server-args.mem-fraction-static",
            "0.65",
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
    ).runtime.sglang_server_args.mem_fraction_static == pytest.approx(0.70)
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(0.85)


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


def test_qwen3_omni_colocated_example_config_loads_and_plans() -> None:
    config_path = _REPO_ROOT / "examples" / "configs" / "qwen3_omni_colocated.yaml"
    config_text = config_path.read_text()

    manager = ConfigManager.from_file(str(config_path))
    config = manager.config
    plan = build_stage_placement_plan(config)

    assert "stages:" not in config_text
    assert "factory:" not in config_text
    assert isinstance(config, Qwen3OmniSpeechColocatedPipelineConfig)
    assert config.name == "qwen3-omni-colocated"
    assert config.process.mode == "multi"
    assert plan.requires_multi_process is True
    assert plan.gpus[0].total_gpu_memory_fraction == pytest.approx(1.0)
    assert _stage(
        config, "thinker"
    ).runtime.sglang_server_args.mem_fraction_static == pytest.approx(0.85)
    assert _stage(
        config, "talker_ar"
    ).runtime.sglang_server_args.mem_fraction_static == pytest.approx(0.40)
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
