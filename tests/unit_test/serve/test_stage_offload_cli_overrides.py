# SPDX-License-Identifier: Apache-2.0
"""--stage-offload-components ar,dit CLI wiring."""

from __future__ import annotations

import pytest
import typer

from sglang_omni.cli.serve import apply_stage_offload_cli_overrides
from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.minimax_music3.config import (
    MiniMaxMusic3DualGPUPipelineConfig,
    MiniMaxMusic3SingleGPUPipelineConfig,
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


def test_ar_and_dit_colocates_both_stages_and_flags_their_factory_args() -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    result = apply_stage_offload_cli_overrides(
        config, stage_offload_components="ar,dit"
    )

    ar_stage = stage_named(result, "minimax_music3_ar")
    dit_stage = stage_named(result, "dit_dav")
    assert ar_stage.process == dit_stage.process
    assert ar_stage.factory.enable_serial_offload is True
    assert dit_stage.factory.enable_serial_offload is True
    assert (
        stage_named(config, "minimax_music3_ar").factory.enable_serial_offload is None
    )


def test_whitespace_and_case_are_normalized() -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    result = apply_stage_offload_cli_overrides(
        config, stage_offload_components=" AR , Dit "
    )

    assert (
        stage_named(result, "dit_dav").process
        == stage_named(result, "minimax_music3_ar").process
    )


def test_partial_component_list_is_rejected() -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    with pytest.raises(typer.BadParameter, match="requires all of"):
        apply_stage_offload_cli_overrides(config, stage_offload_components="ar")


def test_unknown_component_is_rejected() -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    with pytest.raises(typer.BadParameter, match="does not support"):
        apply_stage_offload_cli_overrides(
            config, stage_offload_components="ar,dit,talker"
        )


def test_empty_flag_value_is_rejected() -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    with pytest.raises(typer.BadParameter, match="must not be empty"):
        apply_stage_offload_cli_overrides(config, stage_offload_components=" , ")


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
