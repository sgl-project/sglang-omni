# SPDX-License-Identifier: Apache-2.0
"""NPU platform adaptation tests for FishAudio S2-Pro."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic import (
    audio_decoder as fish_audio_decoder,
)


def test_npu_kvcache_attention_uses_fused_infer_attention(
    monkeypatch,
) -> None:
    q = torch.randn(2, 1, 4, 8)
    k = torch.randn(2, 1, 4, 8)
    v = torch.randn(2, 1, 4, 8)
    k_cache = torch.zeros(2, 11, 4, 8)
    v_cache = torch.zeros(2, 11, 4, 8)
    calls = []

    def fake_fused_attention(query, key, value, **kwargs):
        calls.append((query, key, value, kwargs))
        return torch.full_like(query, 7), torch.empty(0)

    monkeypatch.setattr(
        fish_audio_decoder.torch.ops,
        "npu",
        SimpleNamespace(
            npu_fused_infer_attention_score=fake_fused_attention,
        ),
        raising=False,
    )

    out = fish_audio_decoder.npu_kvcache_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        cache_position=3,
    )

    assert torch.equal(out, torch.full_like(q, 7))
    assert torch.equal(k_cache[:, 3:4], k)
    assert torch.equal(v_cache[:, 3:4], v)
    assert len(calls) == 1
    query, key, value, kwargs = calls[0]
    assert query is q
    assert key.shape == value.shape == (2, 4, 4, 8)
    assert kwargs == {
        "num_heads": 4,
        "num_key_value_heads": 4,
        "input_layout": "BSND",
        "scale": 1.0 / math.sqrt(8),
    }


def test_npu_kvcache_attention_requires_fused_infer_attention(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        fish_audio_decoder.torch.ops,
        "npu",
        SimpleNamespace(),
        raising=False,
    )
    q = torch.randn(1, 1, 4, 8)
    k_cache = torch.zeros(1, 11, 4, 8)
    v_cache = torch.zeros(1, 11, 4, 8)

    with pytest.raises(RuntimeError, match="npu_fused_infer_attention_score"):
        fish_audio_decoder.npu_kvcache_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=q,
            v=q,
            cache_position=0,
        )


def test_npu_kvcache_attention_requires_single_token_query(monkeypatch) -> None:
    monkeypatch.setattr(
        fish_audio_decoder.torch.ops,
        "npu",
        SimpleNamespace(npu_fused_infer_attention_score=lambda *args, **kwargs: None),
        raising=False,
    )
    q = torch.randn(1, 2, 4, 8)
    k_cache = torch.zeros(1, 11, 4, 8)
    v_cache = torch.zeros(1, 11, 4, 8)

    with pytest.raises(ValueError, match="single-token query"):
        fish_audio_decoder.npu_kvcache_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=q,
            v=q,
            cache_position=0,
        )


def test_fast_ar_attention_rejects_unsupported_device_clearly() -> None:
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    k_cache = torch.zeros(1, 11, 4, 8)
    v_cache = torch.zeros(1, 11, 4, 8)

    with pytest.raises(RuntimeError, match="supports CUDA and NPU"):
        fish_audio_decoder.flash_attn_kvcache_op(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=v,
            cache_position=0,
        )


def test_npu_kvcache_attention_requires_kv_and_cache_position() -> None:
    q = torch.randn(1, 1, 4, 8)
    k_cache = torch.zeros(1, 11, 4, 8)
    v_cache = torch.zeros(1, 11, 4, 8)
    k = torch.randn(1, 1, 4, 8)

    with pytest.raises(ValueError, match="requires k and v"):
        fish_audio_decoder.npu_kvcache_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=None,
            v=None,
            cache_position=0,
        )
    with pytest.raises(ValueError, match="requires cache_position"):
        fish_audio_decoder.npu_kvcache_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=k,
            cache_position=-1,
        )


def test_fish_engine_builder_npu_defaults(monkeypatch) -> None:
    from sglang_omni.models.fishaudio_s2_pro import engine_builder as fish_engine

    fake_npu = SimpleNamespace(is_npu=lambda: True, device_type="npu")
    monkeypatch.setattr(fish_engine, "current_platform", fake_npu)

    builder = fish_engine.FishS2ProEngineBuilder(max_new_tokens=256, ras_window=16)
    builder.gpu_id = 0

    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["disable_cuda_graph"] is False
    assert defaults["cuda_graph_backend_decode"] == "full"
    assert defaults["max_running_requests"] == 16
    assert defaults["mem_fraction_static"] == 0.75
    assert defaults["enable_torch_compile"] is False
    assert defaults["dtype"] == "bfloat16"

    # NPU bounds decode graph buckets to keep eager-prefill headroom.
    overrides: dict = {}
    builder.adjust_overrides(overrides)
    assert overrides["cuda_graph_bs"] == [1, 2, 4, 8, 16]
    assert overrides["cuda_graph_max_bs"] == 16

    assert fish_engine.resolve_fast_ar_attention_backend(gpu_id=0) == "ascend"
