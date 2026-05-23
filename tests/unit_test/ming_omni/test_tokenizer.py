# SPDX-License-Identifier: Apache-2.0
"""Thinker tokenizer loading tests."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace


def test_load_ming_tokenizer_handles_custom_config_mapping_keyerror(monkeypatch):
    hub_module = ModuleType("huggingface_hub")
    hub_module.hf_hub_download = lambda *args, **kwargs: None
    hub_module.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)

    sentinel = object()
    calls: list[str] = []

    def fake_auto_from_pretrained(path: str, trust_remote_code: bool = False, **kwargs):
        calls.append(f"auto:{path}")
        raise KeyError("BailingMM2Config")

    def fake_fast_from_pretrained(path: str, **kwargs):
        calls.append(f"fast:{path}")
        if path == "org/model":
            return sentinel
        raise OSError("tokenizer not found")

    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = SimpleNamespace(
        from_pretrained=fake_auto_from_pretrained
    )
    transformers_module.PreTrainedTokenizerFast = SimpleNamespace(
        from_pretrained=fake_fast_from_pretrained
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    weight_loader_module = ModuleType("sglang_omni.models.weight_loader")
    weight_loader_module.resolve_model_path = lambda model_path: model_path
    monkeypatch.setitem(
        sys.modules, "sglang_omni.models.weight_loader", weight_loader_module
    )

    from sglang_omni.models.ming_omni.components import common

    common = importlib.reload(common)

    tokenizer = common.load_ming_tokenizer("org/model")

    assert tokenizer is sentinel
    assert calls == ["auto:org/model", "fast:org/model"]
