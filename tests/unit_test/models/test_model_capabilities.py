# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from dataclasses import MISSING, FrozenInstanceError, fields
from types import ModuleType

import pytest

from sglang_omni.models.model_capabilities import (
    ModelCapabilities,
    get_model_capabilities,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

EXPECTED_MODEL_CAPABILITIES = {
    "FishOmniModel": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=False,
        supports_breakable_prefill_cuda_graph=True,
        supports_full_prefill_cuda_graph=False,
    ),
    "DotsTTSForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=False,
        supports_torch_compile=True,
        supports_breakable_prefill_cuda_graph=False,
        supports_full_prefill_cuda_graph=False,
    ),
    "MiniMaxMusic3ForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=False,
        supports_batch_vocoder=False,
        supports_streaming_vocoder=False,
        supports_cuda_graph=True,
        supports_torch_compile=True,
        supports_breakable_prefill_cuda_graph=False,
        supports_full_prefill_cuda_graph=False,
    ),
    "AudarTTSForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=False,
        supports_streaming_vocoder=False,
        supports_cuda_graph=False,
        supports_torch_compile=False,
        supports_breakable_prefill_cuda_graph=False,
        supports_full_prefill_cuda_graph=False,
    ),
    "Qwen3TTSForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=False,
        supports_breakable_prefill_cuda_graph=True,
        supports_full_prefill_cuda_graph=True,
    ),
    "HiggsMultimodalQwen3ForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=True,
        supports_breakable_prefill_cuda_graph=True,
        supports_full_prefill_cuda_graph=False,
    ),
    "MossTTSDelayModel": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=False,
        supports_breakable_prefill_cuda_graph=True,
        supports_full_prefill_cuda_graph=False,
    ),
    "MossTTSLocalModel": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=True,
        supports_breakable_prefill_cuda_graph=False,
        supports_full_prefill_cuda_graph=False,
    ),
    "FishQwen3OmniForCausalLM": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=True,
        supports_breakable_prefill_cuda_graph=False,
        supports_full_prefill_cuda_graph=False,
    ),
    "BailingMMNativeForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=False,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=False,
        supports_breakable_prefill_cuda_graph=False,
        supports_full_prefill_cuda_graph=False,
    ),
    "VoxtralTTSForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=False,
        supports_batch_vocoder=False,
        supports_streaming_vocoder=False,
        supports_cuda_graph=True,
        supports_torch_compile=True,
        supports_breakable_prefill_cuda_graph=False,
        supports_full_prefill_cuda_graph=False,
    ),
    "Zonos2ForCausalLM": ModelCapabilities(
        supports_reference_audio=True,
        supports_batch_vocoder=True,
        supports_streaming_vocoder=True,
        supports_cuda_graph=True,
        supports_torch_compile=True,
        supports_breakable_prefill_cuda_graph=False,
        supports_full_prefill_cuda_graph=False,
    ),
    "MossTranscribeDiarizeForConditionalGeneration": ModelCapabilities(
        supports_reference_audio=False,
        supports_batch_vocoder=False,
        supports_streaming_vocoder=False,
        supports_cuda_graph=True,
        supports_torch_compile=True,
        supports_breakable_prefill_cuda_graph=True,
        supports_full_prefill_cuda_graph=False,
    ),
}


def capability_required_architectures() -> set[str]:
    return {
        config_cls.architecture
        for config_cls in set(PIPELINE_CONFIG_REGISTRY.configs.values())
        if getattr(config_cls, "requires_model_capabilities", False)
    }


def test_expected_capabilities_cover_registered_required_configs() -> None:
    assert capability_required_architectures() == set(EXPECTED_MODEL_CAPABILITIES)


def test_required_model_capability_configs_resolve_capabilities() -> None:
    for architecture in sorted(capability_required_architectures()):
        assert get_model_capabilities(architecture) is not None


def test_model_capabilities_are_frozen_and_explicit() -> None:
    for field in fields(ModelCapabilities):
        assert field.type in (bool, "bool")
        assert field.default is MISSING
        assert field.default_factory is MISSING

    with pytest.raises(TypeError):
        ModelCapabilities()

    capabilities = next(iter(EXPECTED_MODEL_CAPABILITIES.values()))
    with pytest.raises(FrozenInstanceError):
        capabilities.supports_reference_audio = False


@pytest.mark.parametrize("architecture", EXPECTED_MODEL_CAPABILITIES)
def test_get_model_capabilities_for_registered_architecture(architecture: str) -> None:
    assert (
        get_model_capabilities(architecture)
        == EXPECTED_MODEL_CAPABILITIES[architecture]
    )


def test_get_model_capabilities_for_non_tts_and_unknown_architectures() -> None:
    assert get_model_capabilities("Qwen3OmniMoeForConditionalGeneration") is None
    assert get_model_capabilities("UnknownArchitecture") is None


def test_get_model_capabilities_rejects_malformed_capabilities_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = "tests.unit_test.models.fake_bad_capabilities"
    fake_package = ModuleType(package_name)
    fake_package.CAPABILITIES = object()

    class FakeConfig:
        pass

    FakeConfig.__module__ = f"{package_name}.config"

    monkeypatch.setitem(sys.modules, package_name, fake_package)
    monkeypatch.setitem(
        PIPELINE_CONFIG_REGISTRY.configs,
        "MalformedCapabilitiesModel",
        FakeConfig,
    )

    with pytest.raises(TypeError, match="must be a ModelCapabilities instance"):
        get_model_capabilities("MalformedCapabilitiesModel")


def test_get_model_capabilities_resolves_registered_alias() -> None:
    assert (
        get_model_capabilities("MossTTSDelay")
        == EXPECTED_MODEL_CAPABILITIES["MossTTSDelayModel"]
    )


def test_launcher_model_capabilities_warning_isolated(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from sglang_omni.serve import launcher

    def fail_summary(_pipeline_config: object) -> None:
        raise RuntimeError("capability lookup failed")

    monkeypatch.setattr(launcher, "model_capabilities_log_summary", fail_summary)
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config("Qwen3TTSForConditionalGeneration")

    with caplog.at_level("WARNING", logger="sglang_omni.serve.launcher"):
        launcher.log_model_capabilities(config_cls(model_path="dummy"))

    assert "Failed to resolve model capabilities for startup log" in caplog.text
