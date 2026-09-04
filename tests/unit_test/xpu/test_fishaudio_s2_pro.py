# SPDX-License-Identifier: Apache-2.0
"""Intel XPU policy tests for FishAudio S2-Pro."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from sglang_omni import platforms
from sglang_omni.models.fishaudio_s2_pro import engine_builder, stages
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.audio_decoder import (
    flash_attn_kvcache_op,
)


def _mock_xpu_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(platforms.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(platforms.current_platform, "device_type", "xpu", raising=False)


def test_s2pro_xpu_engine_defaults_skip_cuda_backend_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_xpu_platform(monkeypatch)
    monkeypatch.setattr(
        engine_builder,
        "get_visible_gpu_sm_version",
        lambda _gpu_id: pytest.fail("XPU must not query CUDA compute capability"),
    )
    builder = engine_builder.FishS2ProEngineBuilder(max_new_tokens=16, ras_window=4)
    builder.gpu_id = 0

    defaults = builder.generation_defaults(dtype="bfloat16")
    overrides: dict[str, object] = {}
    builder.adjust_overrides(overrides)

    assert defaults["disable_cuda_graph"] is True
    assert defaults["enable_torch_compile"] is False
    assert "attention_backend" not in overrides


def test_s2pro_vocoder_follows_xpu_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_xpu_platform(monkeypatch)
    codec = SimpleNamespace()
    loaded: list[tuple[str, str]] = []
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)

    def load_codec(checkpoint: str, device: str) -> object:
        loaded.append((checkpoint, device))
        return codec

    monkeypatch.setattr(stages, "_load_codec", load_codec)
    monkeypatch.setattr(
        "sglang_omni.models.fishaudio_s2_pro.streaming_vocoder.S2ProVocoderScheduler",
        lambda _codec, **kwargs: SimpleNamespace(codec=_codec, **kwargs),
    )

    stages.create_vocoder_executor("model", gpu_id=2)
    stages.create_vocoder_executor("model", device="cpu", gpu_id=2)
    stages.create_vocoder_executor("model", device="xpu:0", gpu_id=2)

    assert loaded == [
        ("model", "xpu:2"),
        ("model", "cpu"),
        ("model", "xpu:2"),
    ]


def test_fish_fast_ar_xpu_sdpa_matches_kv_cache_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_xpu_platform(monkeypatch)
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    first_k = torch.tensor([[[[1.0, 0.0]]]])
    first_v = torch.tensor([[[[2.0, 3.0]]]])
    second_k = torch.tensor([[[[0.0, 1.0]]]])
    second_v = torch.tensor([[[[5.0, 7.0]]]])
    k_cache = torch.zeros(1, 3, 1, 2)
    v_cache = torch.zeros_like(k_cache)

    first = flash_attn_kvcache_op(
        q,
        k_cache,
        v_cache,
        k=first_k,
        v=first_v,
        causal=True,
        cache_position=0,
    )
    second = flash_attn_kvcache_op(
        q,
        k_cache,
        v_cache,
        k=second_k,
        v=second_v,
        causal=True,
        cache_position=1,
    )

    expected_k = torch.cat((first_k, second_k), dim=1)
    expected_v = torch.cat((first_v, second_v), dim=1)
    repeated_k = expected_k.transpose(1, 2).repeat_interleave(2, dim=1)
    repeated_v = expected_v.transpose(1, 2).repeat_interleave(2, dim=1)
    query = q.transpose(1, 2)
    weights = torch.softmax(
        torch.matmul(query, repeated_k.transpose(-1, -2)) / math.sqrt(q.shape[-1]),
        dim=-1,
    )
    expected_second = torch.matmul(weights, repeated_v).transpose(1, 2)

    torch.testing.assert_close(first, first_v.repeat_interleave(2, dim=2))
    torch.testing.assert_close(second, expected_second)
    torch.testing.assert_close(k_cache[:, :2], expected_k)
    torch.testing.assert_close(v_cache[:, :2], expected_v)
