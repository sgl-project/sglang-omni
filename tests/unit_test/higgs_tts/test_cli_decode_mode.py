# SPDX-License-Identifier: Apache-2.0
"""Async decode is plain per-stage scheduler configuration.

The dedicated ``--decode-mode`` flag is gone: the knobs live in the stage's
``scheduler`` group like every other executor setting, so they are set with
the same dotted spelling (``--tts_engine.factory.enable_async_decode``)
or under the stage's ``stages:`` entry in YAML, and reach the factory only
if its signature accepts them. Higgs TTS defaults to async decode for
throughput; these tests pin the values the schema refuses for the knobs.
"""

from __future__ import annotations

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig


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
