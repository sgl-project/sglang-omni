# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o pipeline config: stage-to-process placement and executor knobs."""

from __future__ import annotations

import pytest

from sglang_omni.models.minicpm_o.config import (
    CODE2WAV_MAX_CONCURRENCY,
    PREPROCESSING_MAX_CONCURRENCY,
    MiniCPMOPipelineConfig,
    MiniCPMOSpeechPipelineConfig,
)


def _process_by_stage(cfg) -> dict[str, str | None]:
    return {stage.name: stage.process for stage in cfg.stages}


@pytest.mark.parametrize("cls", [MiniCPMOPipelineConfig, MiniCPMOSpeechPipelineConfig])
def test_cpu_and_encoder_stages_leave_the_thinker_process(cls):
    procs = _process_by_stage(cls(model_path="dummy"))
    thinker_process = procs["thinker"]
    # trivial hops stay with the thinker so the merged prompt never crosses a
    # process boundary
    assert procs["mm_aggregate"] == thinker_process
    assert procs["decode"] == thinker_process
    # heavy non-AR work gets its own interpreter
    for stage in ("preprocessing", "image_encoder", "audio_encoder"):
        assert procs[stage] != thinker_process, stage
    assert (
        len({procs["preprocessing"], procs["image_encoder"], procs["audio_encoder"]})
        == 3
    )


def test_speech_stages_are_isolated_too():
    procs = _process_by_stage(MiniCPMOSpeechPipelineConfig(model_path="dummy"))
    assert procs["talker"] not in (procs["thinker"], procs["code2wav"])
    assert procs["code2wav"] != procs["thinker"]


@pytest.mark.parametrize("cls", [MiniCPMOPipelineConfig, MiniCPMOSpeechPipelineConfig])
def test_env_defaults_bound_per_process_thread_pools(cls):
    cfg = cls(model_path="dummy")
    assert cfg.env_defaults == {
        "OMP_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
    }


@pytest.mark.parametrize("cls", [MiniCPMOPipelineConfig, MiniCPMOSpeechPipelineConfig])
def test_colocated_processes_need_no_memory_fractions(cls):
    cfg = cls(model_path="dummy")
    assert cfg.placement.require_memory_fraction_for_colocation is False


def test_executor_concurrency_knobs_reach_the_factories():
    stages = {
        s.name: s for s in MiniCPMOSpeechPipelineConfig(model_path="dummy").stages
    }
    assert (
        stages["preprocessing"].factory.max_concurrency == PREPROCESSING_MAX_CONCURRENCY
    )
    assert PREPROCESSING_MAX_CONCURRENCY > 1
    assert stages["code2wav"].factory.max_concurrency == CODE2WAV_MAX_CONCURRENCY
    assert CODE2WAV_MAX_CONCURRENCY > 1
