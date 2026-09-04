# SPDX-License-Identifier: Apache-2.0
"""Async decode is plain per-stage scheduler configuration.

The dedicated ``--decode-mode`` flag is gone: the knobs live in the stage's
``scheduler`` group like every other executor setting, so they are set with
the same dotted spelling (``--tts_engine.factory.enable_async_decode``)
or under the stage's ``stages:`` entry in YAML, and reach the factory only
if its signature accepts them. Higgs TTS defaults to async decode for
throughput; these tests pin the default and the dotted override.
"""

from __future__ import annotations

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig


def _scheduler(config, stage_name: str):
    return config.stage_named(stage_name).factory


def test_decode_mode_default_config_is_async():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    assert _scheduler(config, "tts_engine").enable_async_decode is True


def test_dotted_flag_can_force_sync_and_async():
    config = HiggsTtsPipelineConfig(model_path="dummy")

    forced_sync = ConfigManager(config).merge_config(
        [("tts_engine.factory.enable_async_decode", "false")]
    )
    assert _scheduler(forced_sync, "tts_engine").enable_async_decode is False

    forced_async = ConfigManager(config).merge_config(
        [("tts_engine.factory.enable_async_decode", "true")]
    )
    assert _scheduler(forced_async, "tts_engine").enable_async_decode is True


def test_async_lookahead_min_batch_size_applies_without_mode_toggle():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    resolved = ConfigManager(config).merge_config(
        [("tts_engine.factory.async_decode_min_batch_size", "4")]
    )
    scheduler = _scheduler(resolved, "tts_engine")
    assert scheduler.enable_async_decode is True
    assert scheduler.async_decode_min_batch_size == 4


def test_async_lookahead_min_batch_size_must_be_positive():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(ValueError, match="async_decode_min_batch_size"):
        ConfigManager(config).merge_config(
            [("tts_engine.factory.async_decode_min_batch_size", "0")]
        )


def test_a_non_boolean_decode_toggle_is_refused():
    config = HiggsTtsPipelineConfig(model_path="dummy")
    with pytest.raises(Exception, match="bool"):
        ConfigManager(config).merge_config(
            [("tts_engine.factory.enable_async_decode", "bogus")]
        )
