# SPDX-License-Identifier: Apache-2.0
"""Optional encoder-cache configuration and namespace isolation."""

from __future__ import annotations

import json
import sys
import types

import pytest
import torch

from sglang_omni.scheduling.encoder_cache import create_encoder_output_cache
from sglang_omni.scheduling.stage_cache import StageOutputCache


BASE = dict(
    model_path="model",
    stage="image_encoder",
    dtype="bfloat16",
    max_entries=4,
    max_bytes=4096,
)


def test_default_uses_stage_cache_without_lmcache():
    cache = create_encoder_output_cache(**BASE)
    assert isinstance(cache, StageOutputCache)
    cache.put("key", torch.ones(2, 3))
    assert torch.equal(cache.get("key"), torch.ones(2, 3))


def test_explicit_namespace_required():
    with pytest.raises(ValueError, match="namespace"):
        create_encoder_output_cache(**BASE, lmcache_config_file="ec.yaml")
    with pytest.raises(ValueError, match="requires"):
        create_encoder_output_cache(**BASE, lmcache_namespace="rev1")


def test_stage_model_dtype_namespace_are_passed_to_adapter(monkeypatch):
    module = types.ModuleType("lmcache.integration.sglang_omni.encoder_cache")
    calls = []
    sentinel = object()

    def create(path, namespace):
        calls.append((path, json.loads(namespace)))
        return sentinel

    module.create_encoder_cache = create
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = create_encoder_output_cache(
        **BASE, lmcache_config_file="ec.yaml", lmcache_namespace="rev1/processor-v1"
    )
    assert result is sentinel
    assert calls == [
        (
            "ec.yaml",
            ["qwen3-omni", "model", "image_encoder", "bfloat16", "rev1/processor-v1"],
        )
    ]
