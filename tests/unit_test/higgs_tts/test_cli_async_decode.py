# SPDX-License-Identifier: Apache-2.0
"""CLI override + config-default tests for Higgs TTS async-decode.

Mirrors the ``--talker-partial-start`` tri-state contract in
``tests/unit_test/qwen3_omni/test_cli.py``: async-decode defaults to ON for
Higgs TTS, and ``--async-decode default|on|off`` can preserve, force, or
disable it. The full-set SeedTTS WER/SIM validation showed the default flip is
quality-neutral, so the off-switch exists for opt-out, not correctness.
"""

from __future__ import annotations

import pytest
import typer
import typer.main

from sglang_omni.cli import app
from sglang_omni.cli.serve import (
    _resolve_async_decode_flag,
    apply_async_decode_cli_overrides,
)
from sglang_omni.config import PipelineConfig, StageConfig, resolve_stage_factory_args
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig


def _tts_engine_args(config):
    stage = next(s for s in config.stages if s.name == "tts_engine")
    return resolve_stage_factory_args(stage, config)


def test_async_decode_default_is_on():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    assert _tts_engine_args(config)["enable_async_decode"] is True


def test_async_decode_cli_override_can_disable_and_enable():
    config = HiggsTtsPipelineConfig(model_path="dummy")

    apply_async_decode_cli_overrides(
        config, async_decode="off", async_decode_min_batch_size=None
    )
    assert _tts_engine_args(config)["enable_async_decode"] is False

    apply_async_decode_cli_overrides(
        config, async_decode="on", async_decode_min_batch_size=None
    )
    assert _tts_engine_args(config)["enable_async_decode"] is True


def test_async_decode_cli_default_preserves_config_default():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    apply_async_decode_cli_overrides(
        config, async_decode="default", async_decode_min_batch_size=None
    )
    assert _tts_engine_args(config)["enable_async_decode"] is True


def test_async_decode_min_batch_size_override_applies_without_toggle():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    apply_async_decode_cli_overrides(
        config, async_decode="default", async_decode_min_batch_size=4
    )
    args = _tts_engine_args(config)
    assert args["enable_async_decode"] is True
    assert args["async_decode_min_batch_size"] == 4


def test_async_decode_min_batch_size_must_be_positive():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_async_decode_cli_overrides(
            config, async_decode="on", async_decode_min_batch_size=0
        )


def test_async_decode_cli_invalid_mode_rejected():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter):
        apply_async_decode_cli_overrides(
            config, async_decode="bogus", async_decode_min_batch_size=None
        )


def test_async_decode_cli_rejects_unsupported_config():
    config = Qwen3TTSPipelineConfig(model_path="dummy")
    with pytest.raises(typer.BadParameter, match="currently supports only Higgs TTS"):
        apply_async_decode_cli_overrides(
            config, async_decode="off", async_decode_min_batch_size=None
        )


def test_async_decode_cli_default_is_noop_without_tts_engine_stage():
    # serve() calls this for every model; pipelines with no tts_engine stage
    # (e.g. Qwen3-Omni, Ming) must serve unaffected when the mode is left at
    # 'default'. The override is gated behind an explicit on/off/min-batch ask,
    # so 'default' must not reach the tts_engine stage lookup at all.
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
    result = apply_async_decode_cli_overrides(
        config, async_decode="default", async_decode_min_batch_size=None
    )
    assert result is config
    assert all(
        "enable_async_decode" not in (stage.factory_args or {})
        for stage in result.stages
    )


def test_enable_async_decode_alias_stays_registered_option():
    # serve uses ignore_unknown_options=True, so a removed flag would fall
    # through to ConfigManager.parse_extra_args and crash existing scripts with
    # "Missing value for argument". Keep both spellings as real options.
    serve_cmd = typer.main.get_command(app).commands["serve"]
    opt_names = {
        opt for param in serve_cmd.params for opt in getattr(param, "opts", [])
    }
    assert "--async-decode" in opt_names
    assert "--enable-async-decode" in opt_names


def test_resolve_async_decode_flag_maps_deprecated_alias_to_on():
    assert _resolve_async_decode_flag("default", enable_async_decode=True) == "on"
    assert _resolve_async_decode_flag("on", enable_async_decode=True) == "on"


def test_resolve_async_decode_flag_passthrough_when_alias_absent():
    assert _resolve_async_decode_flag("default", enable_async_decode=False) == "default"
    assert _resolve_async_decode_flag("off", enable_async_decode=False) == "off"


def test_resolve_async_decode_flag_conflict_rejected():
    with pytest.raises(typer.BadParameter, match="cannot be combined"):
        _resolve_async_decode_flag("off", enable_async_decode=True)


def test_async_decode_min_batch_size_without_tts_engine_fails_fast():
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
    with pytest.raises(typer.BadParameter, match="tts_engine"):
        apply_async_decode_cli_overrides(
            config, async_decode="default", async_decode_min_batch_size=4
        )
