# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni import config as qwen3_omni_config
from sglang_omni.models.qwen3_omni.config import (
    TALKER_MAX_NEW_TOKENS_ESTIMATION,
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.pipeline.stage_workers import (
    StageLaunchConfig,
    StageWorkerProcessSpec,
    _patched_spawn_env,
)
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext

_CLIP_ENV = "SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _colocated_h100_config():
    config_path = (
        _REPO_ROOT / "examples" / "configs" / "qwen3_omni_colocated_h100_bf16.yaml"
    )
    return ConfigManager.from_file(str(config_path)).config


def _stage_specs_by_name(config) -> dict[str, tuple[str, StageLaunchConfig]]:
    prep = prepare_pipeline_runtime(config)
    groups = _build_stage_groups(
        config,
        ctx=FakeMpContext(),
        stages_cfg=prep.stages_cfg,
        endpoints=prep.endpoints,
        placement_plan=prep.placement_plan,
        process_plan=prep.process_plan,
    )
    return {
        stage_spec.stage_name: (process_spec.process_name, stage_spec)
        for group in groups
        for process_spec in group.process_specs
        for stage_spec in process_spec.stage_specs
    }


@pytest.mark.parametrize(
    "make_config",
    [
        pytest.param(_colocated_h100_config, id="colocated-h100"),
        pytest.param(
            lambda: Qwen3OmniSpeechPipelineConfig(model_path="dummy"), id="speech"
        ),
    ],
)
def test_talker_admission_clip_reaches_only_the_talker_process(
    monkeypatch: pytest.MonkeyPatch, make_config
) -> None:
    monkeypatch.setattr(qwen3_omni_config.current_platform, "is_rocm", lambda: False)
    specs = _stage_specs_by_name(make_config())

    talker_process, talker = specs["talker_ar"]
    thinker_process, thinker = specs["thinker"]

    assert talker_process != thinker_process
    assert talker.env_defaults[_CLIP_ENV] == TALKER_MAX_NEW_TOKENS_ESTIMATION
    assert _CLIP_ENV not in thinker.env_defaults


def test_text_pipeline_sets_no_admission_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_omni_config.current_platform, "is_rocm", lambda: False)
    specs = _stage_specs_by_name(Qwen3OmniPipelineConfig(model_path="dummy"))

    assert all(_CLIP_ENV not in spec.env_defaults for _, spec in specs.values())


def test_spawn_env_rejects_thinker_and_talker_clips_in_one_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_CLIP_ENV, raising=False)
    spec = StageWorkerProcessSpec(
        process_name="pipeline",
        stage_specs=[
            StageLaunchConfig(stage_name="thinker", env_defaults={_CLIP_ENV: "32"}),
            StageLaunchConfig(
                stage_name="talker_ar",
                env_defaults={_CLIP_ENV: TALKER_MAX_NEW_TOKENS_ESTIMATION},
            ),
        ],
    )

    with pytest.raises(AssertionError, match="conflicting env default"):
        with _patched_spawn_env(spec):
            pass
