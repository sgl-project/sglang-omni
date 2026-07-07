# SPDX-License-Identifier: Apache-2.0
"""The thinker stage defaults to async decode, and --decode-mode overrides it.

Async decode is on by default (parity with Higgs); --decode-mode sync must route
the override to the thinker stage (resolved by factory) and turn it off.
"""
from __future__ import annotations

from sglang_omni.cli.serve import apply_decode_mode_cli_overrides
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig


def _thinker_factory_args(cfg):
    return next(s for s in cfg.stages if s.name == "thinker").factory_args


def test_thinker_async_on_by_default():
    cfg = Qwen3OmniPipelineConfig(model_path="dummy")
    assert _thinker_factory_args(cfg)["enable_async_decode"] is True


def test_decode_mode_sync_disables_thinker_async():
    cfg = Qwen3OmniPipelineConfig(model_path="dummy")
    apply_decode_mode_cli_overrides(
        cfg, decode_mode="sync", async_lookahead_min_batch_size=None
    )
    assert _thinker_factory_args(cfg)["enable_async_decode"] is False


def test_decode_mode_async_keeps_thinker_async():
    cfg = Qwen3OmniPipelineConfig(model_path="dummy")
    apply_decode_mode_cli_overrides(
        cfg, decode_mode="async", async_lookahead_min_batch_size=None
    )
    assert _thinker_factory_args(cfg)["enable_async_decode"] is True
