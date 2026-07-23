# SPDX-License-Identifier: Apache-2.0
"""CLI override + config-default tests for Higgs TTS decode mode.

Higgs TTS defaults to async decode for throughput, while ``--decode-mode
async|sync`` can force or disable that mode explicitly. Omitting
``--decode-mode`` preserves the pipeline default.
"""

from __future__ import annotations

import pytest
import typer

from sglang_omni.cli.serve import apply_decode_mode_cli_overrides
from sglang_omni.config import PipelineConfig, StageConfig, resolve_stage_factory_args
from sglang_omni.models.fun_asr.config import FunASRPipelineConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.moss_transcribe_diarize.config import (
    MossTranscribeDiarizePipelineConfig,
)
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig


def _tts_engine_args(config: PipelineConfig) -> dict[str, object]:
    stage = next(s for s in config.stages if s.name == "tts_engine")
    return resolve_stage_factory_args(stage, config)


def test_decode_mode_default_config_is_async():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    assert _tts_engine_args(config)["enable_async_decode"] is True


def test_decode_mode_cli_can_force_sync_and_async():
    config = HiggsTtsPipelineConfig(model_path="dummy")

    apply_decode_mode_cli_overrides(
        config, decode_mode="sync", async_lookahead_min_batch_size=None
    )
    assert _tts_engine_args(config)["enable_async_decode"] is False

    apply_decode_mode_cli_overrides(
        config, decode_mode="async", async_lookahead_min_batch_size=None
    )
    assert _tts_engine_args(config)["enable_async_decode"] is True


def test_decode_mode_absent_preserves_config_default():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    apply_decode_mode_cli_overrides(
        config, decode_mode=None, async_lookahead_min_batch_size=None
    )
    assert _tts_engine_args(config)["enable_async_decode"] is True


def test_async_lookahead_min_batch_size_override_applies_without_mode_toggle():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    apply_decode_mode_cli_overrides(
        config, decode_mode=None, async_lookahead_min_batch_size=4
    )
    args = _tts_engine_args(config)
    assert args["enable_async_decode"] is True
    assert args["async_decode_min_batch_size"] == 4


def test_async_lookahead_min_batch_size_must_be_positive():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_decode_mode_cli_overrides(
            config, decode_mode="async", async_lookahead_min_batch_size=0
        )


def test_async_lookahead_min_batch_size_rejected_with_sync_mode():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter, match="cannot be combined"):
        apply_decode_mode_cli_overrides(
            config, decode_mode="sync", async_lookahead_min_batch_size=4
        )


def test_decode_mode_cli_invalid_mode_rejected():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_decode_mode_cli_overrides(
            config, decode_mode="bogus", async_lookahead_min_batch_size=None
        )


def test_decode_mode_cli_rejects_unsupported_config():
    config = Qwen3TTSPipelineConfig(model_path="dummy")
    with pytest.raises(
        typer.BadParameter,
        match="Higgs TTS, MOSS-TTS-Local, MOSS-Transcribe-Diarize, and Fun-ASR",
    ):
        apply_decode_mode_cli_overrides(
            config, decode_mode="sync", async_lookahead_min_batch_size=None
        )


def test_decode_mode_cli_applies_to_moss_transcribe_diarize_asr_stage():
    # MOSS-TD's async-decode stage is named "asr", not "tts_engine": the CLI
    # override matches stages by factory, so it must reach it all the same.
    config = MossTranscribeDiarizePipelineConfig(model_path="dummy")
    apply_decode_mode_cli_overrides(
        config, decode_mode="async", async_lookahead_min_batch_size=4
    )
    stage = next(s for s in config.stages if s.name == "asr")
    args = resolve_stage_factory_args(stage, config)
    assert args["enable_async_decode"] is True
    assert args["async_decode_min_batch_size"] == 4


def test_decode_mode_cli_applies_to_fun_asr_stage():
    config = FunASRPipelineConfig(model_path="dummy")
    apply_decode_mode_cli_overrides(
        config, decode_mode="async", async_lookahead_min_batch_size=4
    )
    stage = next(s for s in config.stages if s.name == "asr")
    args = resolve_stage_factory_args(stage, config)
    assert args["enable_async_decode"] is True
    assert args["async_decode_min_batch_size"] == 4

    apply_decode_mode_cli_overrides(
        config, decode_mode="sync", async_lookahead_min_batch_size=None
    )
    args = resolve_stage_factory_args(stage, config)
    assert args["enable_async_decode"] is False


def test_decode_mode_cli_absent_is_noop_without_tts_engine_stage():
    # serve() calls this for every model; pipelines with no tts_engine stage
    # (e.g. Qwen3-Omni, Ming) must serve unaffected when decode mode and the
    # advanced async-lookahead threshold are left unspecified.
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="thinker",
                process="pipeline",
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                terminal=True,
            )
        ],
    )
    result = apply_decode_mode_cli_overrides(
        config, decode_mode=None, async_lookahead_min_batch_size=None
    )
    assert result is config
    assert all(
        "enable_async_decode" not in (stage.factory_args or {})
        for stage in result.stages
    )


def test_async_lookahead_min_batch_size_without_supported_stage_fails_fast():
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="thinker",
                process="pipeline",
                factory="tests.unit_test.fixtures.pipeline_fakes.dummy_factory",
                terminal=True,
            )
        ],
    )
    with pytest.raises(typer.BadParameter, match="supported factory"):
        apply_decode_mode_cli_overrides(
            config, decode_mode=None, async_lookahead_min_batch_size=4
        )
