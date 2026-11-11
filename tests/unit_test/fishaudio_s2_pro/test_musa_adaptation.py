# SPDX-License-Identifier: Apache-2.0
"""MUSA platform adaptation tests for FishAudio S2-Pro."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic import (
    audio_decoder as fish_audio_decoder,
)


def test_fast_ar_attention_backend_uses_fa3_on_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.fishaudio_s2_pro import engine_builder

    monkeypatch.setattr(
        engine_builder,
        "current_platform",
        SimpleNamespace(is_npu=lambda: False, is_musa=lambda: True),
    )

    backend = engine_builder.resolve_fast_ar_attention_backend(gpu_id=0)

    assert backend == "fa3"


def test_musa_fast_ar_kvcache_reuses_sglang_flash_attn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def flash_attn_with_kvcache(**kwargs: object) -> torch.Tensor:
        calls.append(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(
        fish_audio_decoder,
        "load_fast_ar_flash_attn_kvcache",
        lambda: flash_attn_with_kvcache,
    )
    monkeypatch.setattr(
        fish_audio_decoder,
        "get_exec",
        lambda: SimpleNamespace(
            kernel=SimpleNamespace(attention_backend="fa3"),
        ),
    )

    q = SimpleNamespace(device=SimpleNamespace(index=0))
    k = torch.randn(2, 1, 2, 8)
    v = torch.randn(2, 1, 2, 8)
    k_cache = torch.zeros(2, 11, 2, 8)
    v_cache = torch.zeros(2, 11, 2, 8)
    cache_seqlens = torch.tensor([4, 4], dtype=torch.int32)

    out = fish_audio_decoder.fast_ar_kvcache_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        causal=True,
        num_splits=1,
        cache_position=-1,
    )

    assert out is q
    assert len(calls) == 1
    assert calls[0] == {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k": k,
        "v": v,
        "cache_seqlens": cache_seqlens,
        "causal": True,
        "num_splits": 1,
    }
