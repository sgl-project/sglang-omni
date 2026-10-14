# SPDX-License-Identifier: Apache-2.0
"""The thinker stage defaults to async decode; the dotted flag overrides it.

Async decode is on by default; --thinker.factory.enable_async_decode false
turns it off like any other scheduler setting.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.qwen3_omni.bootstrap import create_thinker_scheduler
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.scheduling import generation_batch_policy


def thinker_scheduler(cfg):
    return cfg.stage_named("thinker").factory


def test_thinker_async_on_by_default():
    cfg = Qwen3OmniPipelineConfig(model_path="dummy")
    assert thinker_scheduler(cfg).enable_async_decode is True


def test_dotted_flag_disables_thinker_async():
    cfg = Qwen3OmniPipelineConfig(model_path="dummy")
    resolved = ConfigManager(cfg).merge_config(
        [("thinker.factory.enable_async_decode", "false")]
    )
    assert thinker_scheduler(resolved).enable_async_decode is False


def test_dotted_flag_keeps_thinker_async():
    cfg = Qwen3OmniPipelineConfig(model_path="dummy")
    resolved = ConfigManager(cfg).merge_config(
        [("thinker.factory.enable_async_decode", "true")]
    )
    assert thinker_scheduler(resolved).enable_async_decode is True


def test_spec_v2_requires_compatible_native_scheduler_configuration(monkeypatch):
    monkeypatch.setattr(
        generation_batch_policy,
        "get_prefill_cuda_graph_backend",
        lambda args: generation_batch_policy.CudaGraphBackend.DISABLED,
    )
    arguments = SimpleNamespace(
        speculative_algorithm="DFLASH", disable_overlap_schedule=True
    )
    with pytest.raises(ValueError, match="disable_overlap_schedule=false"):
        create_thinker_scheduler(arguments, enable_spec_v2_overlap=True)
    arguments.speculative_algorithm = None
    with pytest.raises(ValueError, match="requires DFLASH"):
        create_thinker_scheduler(arguments, enable_spec_v2_overlap=True)
