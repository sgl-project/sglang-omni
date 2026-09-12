# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from sglang_omni.model_runner import mlx_model_worker as registry


def test_factories_are_lazy_and_preserve_asr(monkeypatch):
    calls = []
    factory = lambda: object

    def import_module(name):
        calls.append(name)
        return SimpleNamespace(
            make_qwen3_asr_mlx_runner_class=factory, make_higgs_mlx_runner_class=factory
        )

    monkeypatch.setattr(registry, "import_module", import_module)
    assert not calls
    assert (
        registry.resolve_mlx_runner_factory("Qwen3ASRForConditionalGeneration")
        is factory
    )
    assert calls == ["sglang_omni.models.qwen3_asr.mlx.runner"]
    assert registry.resolve_mlx_runner_factory("HiggsTTSModel") is factory
    assert calls[-1] == "sglang_omni.models.higgs_tts.mlx.runner"


def test_unknown_architecture_does_not_import(monkeypatch):
    monkeypatch.setattr(
        registry, "import_module", lambda name: pytest.fail("unexpected import")
    )
    with pytest.raises(NotImplementedError, match="supported architectures"):
        registry.resolve_mlx_runner_factory("unknown")


def test_register_extension(monkeypatch):
    monkeypatch.setattr(registry, "_MLX_RUNNER_FACTORIES", {})
    factory = lambda: object
    monkeypatch.setattr(
        registry, "import_module", lambda name: SimpleNamespace(factory=factory)
    )
    registry.register_mlx_runner_factory("custom", "example:factory")
    assert registry.resolve_mlx_runner_factory("custom") is factory
