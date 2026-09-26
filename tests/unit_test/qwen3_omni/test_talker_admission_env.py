# SPDX-License-Identifier: Apache-2.0
"""Talker admission defaults stay scoped to the speech worker process."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.models.qwen3_omni import config as qwen3_omni_config
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni.pipeline.mp_runner import build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.pipeline.stage_workers import (
    StageLaunchConfig,
    StageWorkerProcessSpec,
    patched_spawn_env,
)
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext

CLIP_ENV = "SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION"
REPO_ROOT = Path(__file__).resolve().parents[3]


def stage_processes(config: PipelineConfig) -> list[StageWorkerProcessSpec]:
    prepared = prepare_pipeline_runtime(config)
    groups = build_stage_groups(
        config,
        ctx=FakeMpContext(),
        stages_cfg=prepared.stages_cfg,
        endpoints=prepared.endpoints,
        placement_plan=prepared.placement_plan,
        process_plan=prepared.process_plan,
    )
    return [process for group in groups for process in group.process_specs]


@pytest.mark.parametrize("variant", ["colocated-h100", "speech"])
def test_talker_admission_clip_reaches_only_the_talker_process(
    monkeypatch: pytest.MonkeyPatch, variant: Literal["colocated-h100", "speech"]
) -> None:
    monkeypatch.setattr(qwen3_omni_config.current_platform, "is_rocm", lambda: False)
    monkeypatch.delenv(CLIP_ENV, raising=False)
    if variant == "colocated-h100":
        config_path = (
            REPO_ROOT / "examples" / "configs" / "qwen3_omni_colocated_h100_bf16.yaml"
        )
        config = ConfigManager.from_file(str(config_path)).config
    else:
        config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")

    processes = stage_processes(config)
    talker_process = next(
        process
        for process in processes
        if any(stage.stage_name == "talker_ar" for stage in process.stage_specs)
    )
    assert all(stage.stage_name != "thinker" for stage in talker_process.stage_specs)
    for process in processes:
        with patched_spawn_env(process):
            if process is talker_process:
                assert os.environ[CLIP_ENV] == "256"
            else:
                assert CLIP_ENV not in os.environ
        assert CLIP_ENV not in os.environ


def test_text_pipeline_sets_no_admission_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CLIP_ENV, raising=False)
    for process in stage_processes(Qwen3OmniPipelineConfig(model_path="dummy")):
        with patched_spawn_env(process):
            assert CLIP_ENV not in os.environ


def test_spawn_env_rejects_thinker_and_talker_clips_in_one_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CLIP_ENV, raising=False)
    process = StageWorkerProcessSpec(
        process_name="pipeline",
        stage_specs=[
            StageLaunchConfig(stage_name="thinker", env_defaults={CLIP_ENV: "32"}),
            StageLaunchConfig(stage_name="talker_ar", env_defaults={CLIP_ENV: "256"}),
        ],
    )

    with pytest.raises(AssertionError, match="conflicting env default"):
        with patched_spawn_env(process):
            pass
