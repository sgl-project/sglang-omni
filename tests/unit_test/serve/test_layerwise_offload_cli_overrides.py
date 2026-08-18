# SPDX-License-Identifier: Apache-2.0
"""--layerwise-offload-components ar,dit CLI wiring."""

from __future__ import annotations

import pytest
import typer

from sglang_omni.cli.serve import apply_layerwise_offload_cli_overrides
from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.minimax_music3.config import (
    MiniMaxMusic3DualGPUPipelineConfig,
    MiniMaxMusic3SingleGPUPipelineConfig,
)


def _stage(config: PipelineConfig, name: str) -> StageConfig:
    return next(stage for stage in config.stages if stage.name == name)


def test_absent_flag_is_a_no_op() -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    result = apply_layerwise_offload_cli_overrides(
        config, layerwise_offload_components=None
    )

    assert result is config
    assert "enable_serial_offload" not in _stage(result, "minimax_music3_ar").factory_args


def test_ar_and_dit_colocates_both_stages_and_flags_their_factory_args() -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    result = apply_layerwise_offload_cli_overrides(
        config, layerwise_offload_components="ar,dit"
    )

    ar_stage = _stage(result, "minimax_music3_ar")
    dit_stage = _stage(result, "dit_dav")
    assert ar_stage.process == dit_stage.process
    assert ar_stage.factory_args["enable_serial_offload"] is True
    assert dit_stage.factory_args["enable_serial_offload"] is True
    # The override deep-copies, matching every other apply_*_cli_overrides.
    assert "enable_serial_offload" not in _stage(config, "minimax_music3_ar").factory_args


def test_whitespace_and_case_are_normalized() -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    result = apply_layerwise_offload_cli_overrides(
        config, layerwise_offload_components=" AR , Dit "
    )

    assert _stage(result, "dit_dav").process == _stage(result, "minimax_music3_ar").process


def test_partial_component_list_is_rejected() -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    with pytest.raises(typer.BadParameter, match="requires all of"):
        apply_layerwise_offload_cli_overrides(config, layerwise_offload_components="ar")


def test_unknown_component_is_rejected() -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    with pytest.raises(typer.BadParameter, match="does not support"):
        apply_layerwise_offload_cli_overrides(
            config, layerwise_offload_components="ar,dit,talker"
        )


def test_empty_flag_value_is_rejected() -> None:
    config = MiniMaxMusic3SingleGPUPipelineConfig(model_path="/models/minimax")

    with pytest.raises(typer.BadParameter, match="must not be empty"):
        apply_layerwise_offload_cli_overrides(config, layerwise_offload_components=" , ")


def test_unsupported_pipeline_is_rejected() -> None:
    config = HiggsTtsPipelineConfig(model_path="dummy")

    with pytest.raises(typer.BadParameter, match="not supported"):
        apply_layerwise_offload_cli_overrides(config, layerwise_offload_components="ar,dit")


def test_mismatched_gpu_placement_is_rejected() -> None:
    config = MiniMaxMusic3DualGPUPipelineConfig(model_path="/models/minimax")
    assert _stage(config, "minimax_music3_ar").gpu != _stage(config, "dit_dav").gpu

    with pytest.raises(typer.BadParameter, match="same GPU"):
        apply_layerwise_offload_cli_overrides(config, layerwise_offload_components="ar,dit")
