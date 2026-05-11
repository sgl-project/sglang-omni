# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni_v1.config import build_stage_placement_plan
from sglang_omni_v1.config.manager import ConfigManager
from sglang_omni_v1.models.qwen3_omni.config import (
    Qwen3OmniSpeechColocatedPipelineConfig,
)


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
