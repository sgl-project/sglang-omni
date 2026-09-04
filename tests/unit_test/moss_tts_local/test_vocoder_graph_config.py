# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest
import torch

from sglang_omni.models.moss_tts_local import config as config_module


@pytest.mark.parametrize(
    "hip_version,dxg_device,dxg_detection,expected",
    [
        ("7.2.0", True, "1", True),
        ("7.2.0", True, None, True),
        ("7.2.0", True, "0", False),
        ("7.2.0", False, "1", False),
        (None, True, "1", False),
    ],
)
def test_rocm_wsl_dxg_detection_uses_hip_and_dxg_signals(
    monkeypatch,
    hip_version,
    dxg_device,
    dxg_detection,
    expected,
) -> None:
    monkeypatch.setattr(torch.version, "hip", hip_version)
    monkeypatch.setattr(
        config_module.os.path,
        "exists",
        lambda path: dxg_device and path == "/dev/dxg",
    )
    if dxg_detection is None:
        monkeypatch.delenv("HSA_ENABLE_DXG_DETECTION", raising=False)
    else:
        monkeypatch.setenv("HSA_ENABLE_DXG_DETECTION", dxg_detection)

    assert config_module._uses_rocm_wsl_dxg() is expected


def test_rocm_wsl_dxg_detection_without_torch_is_false(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    assert config_module._uses_rocm_wsl_dxg() is False


def test_rocm_wsl_dxg_default_disables_vocoder_graph_before_factory(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config_module, "_uses_rocm_wsl_dxg", lambda: True)
    config = config_module.MossTTSLocalPipelineConfig(model_path="x")

    assert config.cuda_graph is None
    assert config.stage_factory_kwargs("vocoder")["cuda_graph"] is False


def test_non_dxg_default_keeps_vocoder_graph_enabled(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "_uses_rocm_wsl_dxg", lambda: False)
    config = config_module.MossTTSLocalPipelineConfig(model_path="x")

    assert config.cuda_graph is None
    assert config.stage_factory_kwargs("vocoder")["cuda_graph"] is True


def test_rocm_wsl_dxg_explicit_vocoder_graph_contract(monkeypatch) -> None:
    """DXG accepts explicit eager mode but rejects the unsafe force-on value."""
    monkeypatch.setattr(config_module, "_uses_rocm_wsl_dxg", lambda: True)

    disabled = config_module.MossTTSLocalPipelineConfig(
        model_path="x", cuda_graph=False
    )
    assert disabled.stage_factory_kwargs("vocoder")["cuda_graph"] is False
    with pytest.raises(ValueError, match="cannot be enabled on ROCm WSL/DXG"):
        config_module.MossTTSLocalPipelineConfig(model_path="x", cuda_graph=True)


@pytest.mark.parametrize("configured", [False, True])
def test_non_dxg_preserves_explicit_vocoder_graph_override(
    monkeypatch, configured
) -> None:
    monkeypatch.setattr(config_module, "_uses_rocm_wsl_dxg", lambda: False)

    config = config_module.MossTTSLocalPipelineConfig(
        model_path="x", cuda_graph=configured
    )

    assert config.stage_factory_kwargs("vocoder")["cuda_graph"] is configured


def test_unset_vocoder_graph_survives_config_rebuild(monkeypatch) -> None:
    """The resolver's dump/rebuild must not turn the platform default into an override."""
    monkeypatch.setattr(config_module, "_uses_rocm_wsl_dxg", lambda: True)
    config = config_module.MossTTSLocalPipelineConfig(model_path="x")

    rebuilt = type(config)(**config.model_dump())

    assert rebuilt.cuda_graph is None
    assert rebuilt.stage_factory_kwargs("vocoder")["cuda_graph"] is False
