# SPDX-License-Identifier: Apache-2.0
"""Intel XPU policy tests for MOSS-TTS."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

import sglang_omni.models.moss_tts.vocoder as vocoder_module
from sglang_omni import platforms
from sglang_omni.models.moss_tts import engine_builder, stages
from sglang_omni.models.moss_tts.vocoder import _autocast_if_supported


def _mock_xpu_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(
        platforms.current_platform, "device_type", "xpu", raising=False
    )


def test_moss_tts_codec_device_follows_xpu_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_xpu_platform(monkeypatch)

    assert stages._resolve_codec_device(None, 2) == "xpu:2"
    assert stages._resolve_codec_device("cpu", 2) == "cpu"


def test_moss_tts_xpu_defaults_to_eager_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_xpu_platform(monkeypatch)

    defaults = engine_builder.MossTtsEngineBuilder().generation_defaults(
        dtype="bfloat16"
    )

    assert defaults["disable_cuda_graph"] is True


def test_moss_tts_vocoder_enables_bfloat16_autocast_on_xpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.dtype]] = []

    def fake_autocast(*, device_type: str, dtype: torch.dtype):
        calls.append((device_type, dtype))
        return nullcontext()

    monkeypatch.setattr(vocoder_module.torch, "autocast", fake_autocast)

    with _autocast_if_supported(torch.device("xpu:0"), torch.bfloat16):
        pass

    assert calls == [("xpu", torch.bfloat16)]
