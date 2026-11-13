# SPDX-License-Identifier: Apache-2.0
"""--stage-offload-components ar,dit CLI wiring."""

from __future__ import annotations

import os
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest
import torch
import typer

from sglang_omni.cli.serve import apply_stage_offload_cli_overrides
from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.minimax_music3.config import (
    MiniMaxMusic3DualGPUPipelineConfig,
    MiniMaxMusic3SingleGPUPipelineConfig,
)
from sglang_omni.pipeline.stage_workers import (
    StageLaunchConfig,
    StageWorkerProcessSpec,
    patched_spawn_env,
)


def stage_named(config: PipelineConfig, name: str) -> StageConfig:
    return next(stage for stage in config.stages if stage.name == name)


def test_absent_flag_is_a_no_op() -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    result = apply_stage_offload_cli_overrides(config, stage_offload_components=None)

    assert result is config
    assert (
        stage_named(result, "minimax_music3_ar").factory.enable_serial_offload is None
    )


@pytest.mark.parametrize("components", ["ar,dit", " AR , Dit "])
def test_ar_and_dit_colocates_both_stages_and_flags_their_factory_args(
    components: str,
) -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    result = apply_stage_offload_cli_overrides(
        config, stage_offload_components=components
    )

    ar_stage = stage_named(result, "minimax_music3_ar")
    dit_stage = stage_named(result, "dit_dav")
    assert ar_stage.process == dit_stage.process
    assert ar_stage.factory.enable_serial_offload is True
    assert dit_stage.factory.enable_serial_offload is True
    assert (
        stage_named(config, "minimax_music3_ar").factory.enable_serial_offload is None
    )


@pytest.mark.parametrize(
    ("components", "message"),
    [
        ("ar", "requires all of"),
        ("ar,dit,talker", "does not support"),
        (" , ", "must not be empty"),
    ],
)
def test_invalid_component_list_is_rejected(components: str, message: str) -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    with pytest.raises(typer.BadParameter, match=message):
        apply_stage_offload_cli_overrides(config, stage_offload_components=components)


def test_unsupported_pipeline_is_rejected() -> None:
    config = HiggsTtsPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="not supported"):
        apply_stage_offload_cli_overrides(config, stage_offload_components="ar,dit")


def test_mismatched_gpu_placement_is_rejected() -> None:
    config = MiniMaxMusic3DualGPUPipelineConfig(model_path="/models/minimax")
    assert (
        stage_named(config, "minimax_music3_ar").gpu
        != stage_named(config, "dit_dav").gpu
    )

    with pytest.raises(typer.BadParameter, match="same GPU"):
        apply_stage_offload_cli_overrides(config, stage_offload_components="ar,dit")


def test_offload_allocator_reaches_workers_with_existing_preload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    library = tmp_path / "torch_memory_saver_hook_mode_preload_cu13.test.so"
    library.touch()
    package = ModuleSpec(
        "torch_memory_saver", loader=None, origin=str(tmp_path / "__init__.py")
    )
    monkeypatch.setattr(
        "sglang_omni.models.minimax_music3.config.find_spec", lambda name: package
    )
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setenv("LD_PRELOAD", "/lib/profiler.so")
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")
    config = apply_stage_offload_cli_overrides(
        config, stage_offload_components="ar,dit"
    )
    environment = config.resolved_env_defaults()
    worker = StageWorkerProcessSpec(
        process_name="music3",
        stage_specs=[
            StageLaunchConfig(stage_name=name, env_defaults=environment)
            for name in ("minimax_music3_ar", "dit_dav")
        ],
    )
    with patched_spawn_env(worker):
        assert os.environ["LD_PRELOAD"] == f"{library}:/lib/profiler.so"
    assert os.environ["LD_PRELOAD"] == "/lib/profiler.so"
