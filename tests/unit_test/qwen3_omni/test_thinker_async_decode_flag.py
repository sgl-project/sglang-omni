# SPDX-License-Identifier: Apache-2.0
"""The dotted flag explicitly selects the thinker's decode scheduling mode."""
from __future__ import annotations

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig


def thinker_scheduler(cfg):
    return cfg.stage_named("thinker").factory


def test_dotted_flag_disables_thinker_async():
    cfg = Qwen3OmniPipelineConfig(model_path="dummy")
    resolved = ConfigManager(cfg).merge_config(
        [("thinker.factory.enable_async_decode", "false")]
    )
    assert thinker_scheduler(resolved).enable_async_decode is False
