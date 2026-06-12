# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest import mock

import sglang.srt.models.registry as registry_mod

import sglang_omni.model_runner.sglang_model_runner as runner_mod


def test_register_omni_model_skips_unimportable(monkeypatch):
    registry = SimpleNamespace(models={})
    monkeypatch.setattr(registry_mod, "ModelRegistry", registry)

    def fake_import(name, *args, **kwargs):
        if "higgs_tts" in name:
            raise ModuleNotFoundError("No module named 'qwen_vl_utils'")
        return mock.MagicMock()

    monkeypatch.setattr(importlib, "import_module", fake_import)

    runner_mod.SGLModelRunner._register_omni_model(object())

    assert "MossTTSDelaySGLangModel" in registry.models
    assert "HiggsMultimodalQwen3ForConditionalGeneration" not in registry.models
