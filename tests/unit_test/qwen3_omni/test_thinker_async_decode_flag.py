# SPDX-License-Identifier: Apache-2.0
"""The thinker stage defaults to async decode; the dotted flag overrides it.

Async decode is on by default; --thinker.factory.enable_async_decode false
turns it off like any other scheduler setting.
"""
from __future__ import annotations

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig


def _thinker_scheduler(cfg):
    return cfg.stage_named("thinker").factory


def test_thinker_async_on_by_default():
    cfg = Qwen3OmniPipelineConfig(model_path="dummy")
    assert _thinker_scheduler(cfg).enable_async_decode is True


def test_dotted_flag_disables_thinker_async():
    cfg = Qwen3OmniPipelineConfig(model_path="dummy")
    resolved = ConfigManager(cfg).merge_config(
        [("thinker.factory.enable_async_decode", "false")]
    )
    assert _thinker_scheduler(resolved).enable_async_decode is False


def test_dotted_flag_keeps_thinker_async():
    cfg = Qwen3OmniPipelineConfig(model_path="dummy")
    resolved = ConfigManager(cfg).merge_config(
        [("thinker.factory.enable_async_decode", "true")]
    )
    assert _thinker_scheduler(resolved).enable_async_decode is True
